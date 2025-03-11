import os
import torch
import torch.nn as nn
import torch.optim as optim
from loss.sidloss import SIDLoss
from loss.yolo_loss import Loss
from .mytrainer import BaseCLODYOLO
from ultralytics import YOLO
import math
from copy import deepcopy
import micromind as mm
from micromind.networks.yolo import Darknet, Yolov8Neck, DetectionHead, SPPF
from sid_nn.sidyolo import DetectionHeadSID
from micromind.utils.yolo import get_variant_multiples
import os
from validation.validator import DetectionValidator
from copy import deepcopy
from micromind.utils.helpers import get_logger

from collections import OrderedDict

logger = get_logger()

# This is used ONLY if you are not using argparse to get the hparams
default_cfg = {
    "output_folder": "results",
    "experiment_name": "micromind_exp",
    "opt": "adam",  # this is ignored if you are overriding the configure_optimizers
    "lr": 0.001,  # this is ignored if you are overriding the configure_optimizers
    "debug": False,
}

class YOLOv8Backbone(torch.nn.Module):

    def __init__(self, version="n"):
        super().__init__()
        model = "yolov8"+version+"-cls.pt"
        classifier = YOLO(model)

        self.sequential = classifier.model.model

        self.ps_indices = [4, 6, 8]
        self.num_blocks = 9

    def forward(self, x):

        ps = []
        for i in range(self.num_blocks):
            if i in self.ps_indices:
                ps.append(self.sequential[i](x))
                x = ps[-1]
            else:
                x = self.sequential[i](x)

        return ps  

""" SID  """
class YOLOSID(BaseCLODYOLO):

    def __init__(self, m_cfg, hparams, data_cfg_path_val, exp_folder, teacher_dict=None, logger=None, old_classes = None, new_classes=None, *args, **kwargs):
        """Initializes the YOLO model."""
        super().__init__(m_cfg, hparams, data_cfg_path_val, exp_folder, logger, *args, **kwargs)
        self.hparams = deepcopy(hparams)
        self.exp_folder = exp_folder
        self.hparams.data_cfg_val = data_cfg_path_val
        self.m_cfg = m_cfg
        self.old_classes = old_classes
        self.new_classes = new_classes

        w, r, d = get_variant_multiples(hparams.model_size)

        self.modules["backbone"] = YOLOv8Backbone()
        self.modules["sppf"] = SPPF(int(512 * w * r), int(512 * w * r))
        self.modules["neck"] = Yolov8Neck(
            filters=[int(256 * w), int(512 * w), int(512 * w * r)],
            heads=hparams.heads,
            d=d,
        )
        self.modules["head"] = DetectionHeadSID(
            hparams.num_classes,
            filters=(int(256 * w), int(512 * w), int(512 * w * r)),
            heads=hparams.heads,
        )

        self.is_first_task = teacher_dict is None

        # load student (previous model)
        if teacher_dict is not None:
            self.modules.load_state_dict(teacher_dict, strict=False)

        if teacher_dict is not None:
        # teacher
            self.modules["teacher_backbone"] = Darknet(w, r, d)
            self.modules["teacher_backbone"] = YOLOv8Backbone()
            self.modules["teacher_sppf"] = SPPF(int(512 * w * r), int(512 * w * r))
            self.modules["teacher_neck"] = Yolov8Neck(
                filters=[int(256 * w), int(512 * w), int(512 * w * r)],
                heads=hparams.heads,
                d=d,
            )
            self.modules["teacher_head"] = DetectionHeadSID(
                hparams.num_classes,
                filters=(int(256 * w), int(512 * w), int(512 * w * r)),
                heads=hparams.heads,
            )
        if teacher_dict is not None:
            #print(self.modules["head"].cv3_3)
            # modify state dict to match teacher keys
            teacher_state_dict = OrderedDict([("teacher_"+k, v) for k,v in teacher_dict.items()])
            # load just teacher modules
            self.modules.load_state_dict(teacher_state_dict, strict=False)

        self.output_int_teacher = None # used to save low level output teacher
        self.output_int_student = None # used to save low level student output
        self.output_teacher = None

        self.sid_consts = hparams.sid_consts
        self.sid_cls_out = self.sid_consts[1] if len(self.sid_consts)>1 else None

        # temp fix
        new_classes = self.m_cfg.classes if self.sid_cls_out is not None else []
        #self.new_classes = new_classes

        if self.is_first_task:
            self.criterion = Loss(self.m_cfg, self.modules["head"], self.device)
        else: 
            self.criterion = SIDLoss(self.m_cfg, self.modules["head"], self.device, self.sid_consts[0], self.sid_cls_out, new_classes, old_classes=self.old_classes)

        # logger for mAPs
        self.logger = logger

        print("Number of parameters for each module:")
        print(self.compute_params())

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        preprocessed_batch = {}
        preprocessed_batch["img"] = (
            batch["img"].to(self.device, non_blocking=True).float() / 255
        )
        for k in batch:
            if isinstance(batch[k], torch.Tensor) and k != "img":
                preprocessed_batch[k] = batch[k].to(self.device)

        return preprocessed_batch
    

    def forward(self, batch):
        """Runs the forward method by calling every module."""
        if self.modules.training:   

            preprocessed_batch = self.preprocess_batch(batch)
            backbone = self.modules["backbone"](
                preprocessed_batch["img"].to(self.device)
            )
            if not self.is_first_task:
                self.output_int_teacher = []
                self.output_int_student = []
                with torch.no_grad():
                    backbone_teacher = self.modules["teacher_backbone"](
                        preprocessed_batch["img"].to(self.device)
                    )
        else:

            if torch.is_tensor(batch):
                backbone = self.modules["backbone"](batch)
                if "sppf" in self.modules.keys():
                    neck_input = list(backbone)[0:2]
                    neck_input.append(self.modules["sppf"](backbone[2]))
                else:
                    neck_input = backbone
                neck = self.modules["neck"](*neck_input)
                head = self.modules["head"](neck)
                return head

            backbone = self.modules["backbone"](batch["img"] / 255)
            if not self.is_first_task:
                with torch.no_grad():
                    backbone_teacher = self.modules["teacher_backbone"](batch["img"] / 255)

        if "sppf" in self.modules.keys():
            neck_input = list(backbone)[0:2]
            neck_input.append(self.modules["sppf"](backbone[2]))
            if not self.is_first_task:
                neck_input_teacher = list(backbone_teacher)[0:2]
                neck_input_teacher.append(self.modules["teacher_sppf"](backbone_teacher[2]))
        else:
            neck_input = backbone
            if not self.is_first_task:
                neck_input_teacher = backbone_teacher

        neck = self.modules["neck"](*neck_input)

        if not self.is_first_task:    

            with torch.no_grad():
                neck_teacher = self.modules["teacher_neck"](*neck_input_teacher)
                
            head_teacher = self.modules["teacher_head"](neck_teacher)

            # save intermediate teacher output
            for i in range(self.modules["teacher_head"].nl):
                self.output_int_teacher.append(self.modules["teacher_head"].second_reg_conv[i])
                self.output_int_teacher.append(self.modules["teacher_head"].second_cls_conv[i])

            if self.sid_cls_out is not None:
                self.output_teacher = head_teacher

        head = self.modules["head"](neck)

        if not self.is_first_task:
            for i in range(self.modules["head"].nl):
                self.output_int_student.append(self.modules["head"].second_reg_conv[i])
                self.output_int_student.append(self.modules["head"].second_cls_conv[i])

        return head

    def compute_loss(self, pred, batch):
        """Computes the loss."""
        preprocessed_batch = self.preprocess_batch(batch)

        if self.is_first_task:
            lossi_sum, loss = self.criterion(pred, preprocessed_batch)
            return lossi_sum

        lossi_sum, lossi = self.criterion(
            pred,
            preprocessed_batch,
            self.output_int_student,
            self.output_int_teacher,
            self.output_teacher
        )

        return lossi_sum
    

    def save_last_model(self, path, task):
        """Save just student model"""

        state_dict = self.modules.state_dict()

        list_params_student = []
        # remove teacher and save just student: remove (key, value) if key has "teacher"
        for k,v in state_dict.items():
            if "teacher" in k:
                continue
            list_params_student.append((k,v))

        student_state_dict = OrderedDict(list_params_student)

        torch.save(student_state_dict, path+f"/model_task_{task}.pt")


    def load_model_prev_task(self, state_dict):
        """load student net from previous task"""
        self.modules.load_state_dict(state_dict, strict=False)

    
    def on_train_end(self):
        
        if self.is_first_task or self.current_epoch == 0:
            return super().on_train_end()

        for i, conv in enumerate(self.modules["head"].cv3_3):
            with torch.no_grad():
                conv.weight[self.old_classes,:,:,:] = self.modules["teacher_head"].cv3_3[i].weight[self.old_classes,:,:,:]

        args = dict(
            model="yolov8n.pt", data=self.hparams.data_cfg_val, verbose=False, plots=False
        )
        validator = DetectionValidator(args=args)

        validator(model=self)

        # log mAPs
        self.log_maps(validator, epoch=self.current_epoch+1)