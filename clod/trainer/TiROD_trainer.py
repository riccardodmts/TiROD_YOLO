"""
YOLO training.

This code allows you to train an object detection model with the YOLOv8 neck and loss.

To run this script, you can start it with:
    python train_yolov8.py cfg/<cfg_file>.py

Authors:
    - Matteo Beltrami, 2024
    - Francesco Paissan, 2024
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from loss.yolo_loss import Loss
import math
from copy import deepcopy
from tqdm import tqdm
import numpy as np

import micromind as mm
from micromind.networks.yolo import Darknet, Yolov8Neck, DetectionHead, SPPF
from sid_nn.sidyolo import DetectionHeadSID
from micromind.utils.yolo import get_variant_multiples
import os
from validation.validator import DetectionValidator, Pseudolabel, PseudolabelReplay
from copy import deepcopy
from micromind.utils.helpers import get_logger
from micromind.utils.checkpointer import Checkpointer
from micromind.core import Metric, Stage
from typing import Dict, List, Optional
from data.OCDM import OCDM

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
class BaseCLODYOLO(mm.MicroMind):

    def __init__(self, m_cfg, hparams, data_cfg_path_val, exp_folder, logger=None, logmAP50=True, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.logmAP50 = logmAP50
        self.hparams = deepcopy(hparams)
        self.exp_folder = exp_folder
        self.hparams.data_cfg_val = data_cfg_path_val
        self.m_cfg = m_cfg
        w, r, d = get_variant_multiples(hparams.model_size)


        self.modules["backbone"] = Darknet(w, r, d)
        self.modules["neck"] = Yolov8Neck(
            filters=[int(256 * w), int(512 * w), int(512 * w * r)],
            heads=hparams.heads,
            d=d,
        )
        self.modules["head"] = DetectionHead(
            hparams.num_classes,
            filters=(int(256 * w), int(512 * w), int(512 * w * r)),
            heads=hparams.heads,
        )

        self.criterion = Loss(self.m_cfg, self.modules["head"], self.device)

        # logger for mAPs
        self.logger = logger


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

        if "sppf" in self.modules.keys():
            neck_input = list(backbone)[0:2]
            neck_input.append(self.modules["sppf"](backbone[2]))
        else:
            neck_input = backbone
        neck = self.modules["neck"](*neck_input)
        head = self.modules["head"](neck)

        return head

    def compute_loss(self, pred, batch):
        """Computes the loss."""
        preprocessed_batch = self.preprocess_batch(batch)

        lossi_sum, lossi = self.criterion(
            pred,
            preprocessed_batch,
        )

        return lossi_sum

    def build_optimizer(
        self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e6
    ):
        """
        Constructs an optimizer for the given model, based on the specified optimizer
        name, learning rate, momentum, weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the
                optimizer is selected based on the number of iterations.
                Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer.
                Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines
                the optimizer if name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(
            v for k, v in nn.__dict__.items() if "Norm" in k
        )  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            print(
                f"optimizer: 'optimizer=auto' found, "
                f"ignoring 'lr0={lr}' and 'momentum={momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", self.hparams.num_classes)  # number of classes
            lr_fit = round(
                0.002 * 5 / (4 + nc), 6
            )  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("AdamW", lr_fit, 0.9)
            lr *= 10
            # self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in ("Adam", "Adamax", "AdamW", "NAdam", "RAdam"):
            optimizer = getattr(optim, name, optim.Adam)(
                g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit"
                "https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group(
            {"params": g[0], "weight_decay": decay}
        )  # add g0 with weight_decay
        optimizer.add_param_group(
            {"params": g[1], "weight_decay": 0.0}
        )  # add g1 (BatchNorm2d weights)
        print(
            f"{optimizer:} {type(optimizer).__name__}(lr={lr}, "
            f"momentum={momentum}) with parameter groups"
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} "
            f"weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer, lr

    def _setup_scheduler(self, opt, lrf=0.01, lr0=0.01, cos_lr=True):
        """Initialize training learning rate scheduler."""

        def one_cycle(y1=0.0, y2=1.0, steps=100):
            """Returns a lambda function for sinusoidal ramp from y1 to y2
            https://arxiv.org/pdf/1812.01187.pdf."""
            return (
                lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1)
                + y1
            )

        #lrf *= lr0

        if cos_lr:
            self.lf = one_cycle(1, lrf, 350)  # cosine 1->hyp['lrf']
        else:
            self.lf = (
                lambda x: max(1 - x / self.epochs, 0) * (1.0 - lrf) + lrf
            )  # linear
        return optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.lf)

    def configure_optimizers(self):
        """Configures the optimizer and the scheduler."""
        # opt = torch.optim.SGD(self.modules.parameters(), lr=1e-2, weight_decay=0.0005)
        # opt = torch.optim.AdamW(
        #     self.modules.parameters(), lr=0.000119, weight_decay=0.0
        # )
        #opt, lr = self.build_optimizer(self.modules, name="auto", lr=0.01, momentum=0.9)
        opt, lr = self.build_optimizer(self.modules, name="SGD", lr=self.hparams.lr0, momentum=self.hparams.momentum)
        sched = self._setup_scheduler(opt, self.hparams.lrf, self.hparams.lr0, cos_lr=False)

        return opt, sched
    
    def train(
        self,
        epochs: int = 1,
        warmup: bool = False,
        datasets: Dict = {},
        metrics: List[Metric] = [],
        checkpointer: Optional[Checkpointer] = None,
        max_norm=10.0,
        debug: Optional[bool] = False,
        skip=False
    ):
        self.epochs = epochs
        if not warmup:
            logger.info("No warmup!")
            super().train(epochs, datasets, metrics, checkpointer, max_norm, debug)
        else:
            warmup_finished = False
            self.datasets = datasets
            self.metrics = metrics
            self.checkpointer = checkpointer
            assert "train" in self.datasets, "Training dataloader was not specified."
            assert epochs > 0, "You must specify at least one epoch."
            self.epochs -= self.hparams.warmup_epochs
            self.debug = debug

            self.on_train_start()
            
            if skip:
                return None

            if self.accelerator.is_local_main_process:
                logger.info(
                    f"Starting from epoch {self.start_epoch + 1}."
                    + f" Training is scheduled for {epochs} epochs."
                )
            
            warmup_epochs = self.hparams.warmup_epochs
            warmup_bias_lr = self.hparams.warmup_bias_lr
            warmup_lrf = self.hparams.lr0
            warmup_momentum = self.hparams.warmup_momentum
            warmup_f_momentum = self.hparams.momentum

            nb = len(self.datasets["train"])
            nw = max(round(warmup_epochs * nb), 100) if warmup_epochs > 0 else -1  # warmup iterations
            
            for e in range(self.start_epoch + 1, epochs + 1):
                self.current_epoch = e
                pbar = tqdm(
                    self.datasets["train"],
                    unit="batches",
                    ascii=True,
                    dynamic_ncols=True,
                    disable=not self.accelerator.is_local_main_process,
                )
                loss_epoch = 0
                pbar.set_description(f"Running epoch {self.current_epoch}/{epochs}")
                self.modules.train()
                for idx, batch in enumerate(pbar):
                    
                    # warmup
                    ni = idx + nb * (e - 1)
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        for j, x in enumerate(self.opt.param_groups):
                            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [warmup_bias_lr if j == 0 else 0.0, warmup_lrf]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [warmup_momentum, warmup_f_momentum])
                    else:
                        warmup_finished = True



                    if isinstance(batch, list):
                        batch = [b.to(self.device) for b in batch]

                    self.opt.zero_grad()

                    with self.accelerator.autocast():
                        model_out = self(batch)
                        loss = self.compute_loss(model_out, batch)
                        loss_epoch += loss.item()

                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(
                        self.modules.parameters(), max_norm=max_norm
                    )
                    self.opt.step()

                    # loss_epoch += loss.item()

                    for m in self.metrics:
                        if (
                            self.current_epoch + 1
                        ) % m.eval_period == 0 and not m.eval_only:
                            m(model_out, batch, Stage.train, self.device)

                    running_train = {}
                    for m in self.metrics:
                        if (
                            self.current_epoch + 1
                        ) % m.eval_period == 0 and not m.eval_only:
                            running_train["train_" + m.name] = m.reduce(Stage.train)

                    running_train.update({"train_loss": loss_epoch / (idx + 1)})

                    pbar.set_postfix(**running_train)

                    if self.debug and idx > 10:
                        break

                pbar.close()

                train_metrics = {}
                for m in self.metrics:
                    if (self.current_epoch + 1) % m.eval_period == 0 and not m.eval_only:
                        train_metrics["train_" + m.name] = m.reduce(Stage.train, True)

                train_metrics.update({"train_loss": loss_epoch / (idx + 1)})

                if "val" in datasets:
                    val_metrics = self.validate()
                else:
                    train_metrics.update({"val_loss": loss_epoch / (idx + 1)})
                    val_metrics = train_metrics

                self.on_train_epoch_end()

                if self.accelerator.is_local_main_process and self.checkpointer is not None:
                    self.checkpointer(
                        self,
                        train_metrics,
                        val_metrics,
                    )

                if e >= 1 and self.debug:
                    break

                if hasattr(self, "lr_sched") and warmup_finished:
                    # ok for cos_lr
                    # self.lr_sched.step(val_metrics["val_loss"])
                    print(f"sched step - old LR={self.lr_sched.get_lr()}")
                    self.lr_sched.step()
                    print(f"sched step - new LR={self.lr_sched.get_lr()}")

            self.on_train_end()
        return None
    
    
    @torch.no_grad()
    def on_train_epoch_end(self):
        """
        Computes the mean average precision (mAP) at the end of the training epoch
        and logs the metrics in `metrics.txt` inside the experiment folder.
        The `verbose` argument if set to `True` prints details regarding the
        number of images, instances and metrics for each class of the dataset.
        The `plots` argument, if set to `True`, saves in the `runs/detect/train`
        folder the plots of the confusion matrix, the F1-Confidence,
        Precision-Confidence, Precision-Recall, Recall-Confidence curves and the
        predictions and labels of the first three batches of images.
        """
        args = dict(
            model="yolov8n.pt", data=self.hparams.data_cfg_val, verbose=False, plots=False
        )
        validator = DetectionValidator(args=args)

        validator(model=self)

        val_metrics = [
            validator.metrics.box.map * 100,
            validator.metrics.box.map50 * 100,
            validator.metrics.box.map75 * 100,
        ]
        metrics_file = os.path.join(self.exp_folder, "val_log.txt")
        metrics_info = (
            f"Epoch {self.current_epoch}: "
            f"mAP50-95(B): {round(val_metrics[0], 3)}%; "
            f"mAP50(B): {round(val_metrics[1], 3)}%; "
            f"mAP75(B): {round(val_metrics[2], 3)}%\n"
        )


        with open(metrics_file, "a") as file:
            file.write(metrics_info)

    
        # initialize logger for current task if current epoch is the first one
        if self.current_epoch == 1:
            self.logger.on_task_start()

        # log mAPs
        self.log_maps(validator)

        return
    

    def save_last_model(self, path, task):

        torch.save(self.modules.state_dict(), path+f"/model_task_{task}.pt")


    def load_model_prev_task(self, path, prev_task):

        self.modules.load_state_dict(torch.load(path+f"/model_task_{prev_task}.pt"))
    

    def log_maps(self, validator):
        """log mAPs using the clod logger"""


        # temp solution. TODO
        nc_seen = self.m_cfg.classes[-1] + 1  # number of classes seen up to now

        mAP_per_class = validator.metrics.box.maps
        
        sum_mAP = 0.0

        sum_mAP50 = 0.0

        # compute mAP50-95 and mAP50 considering classes seen
        if len(mAP_per_class):
            for i in range(nc_seen):
                sum_mAP += mAP_per_class[i]
        else:
            mAP_per_class = np.zeros(13)

        self.logger.log(self.current_epoch, mAP_per_class, sum_mAP/nc_seen, None, sum_mAP50/nc_seen)

    @torch.no_grad()
    def evaluate(self, path_yaml):

        args = dict(
            model="yolov8n.pt", data=path_yaml, verbose=False, plots=False
        )

        validator = DetectionValidator(args=args)

        validator(model=self)

        try:
            return validator.metrics.box.maps, validator.metrics.box.all_ap[:,0]
        except:
            return np.zeros(self.modules["head"].nc), np.zeros(self.modules["head"].nc)


from ultralytics import YOLO

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

class YOLOOurs(BaseCLODYOLO):
    def __init__(self, m_cfg, hparams, data_cfg_path_val, exp_folder, logger=None, oldlabels=False, *args, **kwargs):
        """Initializes the YOLO model."""
        super().__init__(m_cfg, hparams, data_cfg_path_val, exp_folder, logger, *args, **kwargs)
        self.hparams = deepcopy(hparams)
        self.exp_folder = exp_folder
        self.hparams.data_cfg_val = data_cfg_path_val
        self.m_cfg = m_cfg
        w, r, d = get_variant_multiples(hparams.model_size)

        self.modules["backbone"] = Darknet(w, r, d)
        self.modules["backbone"] = YOLOv8Backbone()
        self.modules["sppf"] = SPPF(int(512 * w * r), int(512 * w * r))
        self.modules["neck"] = Yolov8Neck(
            filters=[int(256 * w), int(512 * w), int(512 * w * r)],
            heads=hparams.heads,
            d=d,
        )
        print(hparams.num_classes)
        self.modules["head"] = DetectionHead(
            hparams.num_classes,
            filters=(int(256 * w), int(512 * w), int(512 * w * r)),
            heads=hparams.heads,
        )

        """
        if oldlabels and self.m_cfg.classes[0]>0:
            old_classes = [i for i in range(self.m_cfg.classes[0])]
            classes = old_classes + self.m_cfg.classes
        else:
            classes = self.m_cfg.classes
        """
        self.criterion = Loss(self.m_cfg, self.modules["head"], self.device)

        # logger for mAPs
        self.logger = logger

        print("Number of parameters for each module:")
        print(self.compute_params())

    def add_pseudo_labels(self, data, classes):

        dataloader = deepcopy(data)

        # disable temp augmentation
        dataloader.dataset.augment = False
        dataloader.dataset.transforms = dataloader.dataset.build_transforms(hyp=dataloader.dataset.hyp)

        pseudolabel = Pseudolabel(classes=classes, dataloader=dataloader, ths=self.hparams.inference_ths)
        # add labels to dataset
        pseudolabel(data, model=self)
    
    def add_pseudo_lables_replay_memory(self, replay_memory):


        is_ocdm = isinstance(replay_memory, OCDM)

        # create loader
        loader = DataLoader(
            replay_memory,
            batch_size=16,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=getattr(replay_memory, "collate_fnv2", OCDM.collate_fn),
        )

        pseudolabel = PseudolabelReplay(dataloader=loader, ths=self.hparams.inference_ths, ocdm=is_ocdm)
        pseudolabel(model=self, classes=self.m_cfg.classes)

        return replay_memory
    
from loss.LwFloss import LwFLoss, LwFLossV2

class YOLOLwF(BaseCLODYOLO):

    def __init__(self, m_cfg, hparams, data_cfg_path_val, exp_folder, teacher_dict, logger=None, *args, **kwargs):
        """Initializes the YOLO model."""
        super().__init__(m_cfg, hparams, data_cfg_path_val, exp_folder, logger, *args, **kwargs)
        self.hparams = deepcopy(hparams)
        self.exp_folder = exp_folder
        self.hparams.data_cfg_val = data_cfg_path_val
        self.m_cfg = m_cfg

        w, r, d = get_variant_multiples(hparams.model_size)

        self.modules["backbone"] = Darknet(w, r, d)
        self.modules["backbone"] = YOLOv8Backbone()
        self.modules["sppf"] = SPPF(int(512 * w * r), int(512 * w * r))
        self.modules["neck"] = Yolov8Neck(
            filters=[int(256 * w), int(512 * w), int(512 * w * r)],
            heads=hparams.heads,
            d=d,
        )
        self.modules["head"] = DetectionHead(
            hparams.num_classes,
            filters=(int(256 * w), int(512 * w), int(512 * w * r)),
            heads=hparams.heads,
        )

        # load student (previous model)
        self.modules.load_state_dict(teacher_dict)

        # teacher
        self.modules["teacher_backbone"] = Darknet(w, r, d)
        self.modules["teacher_backbone"] = YOLOv8Backbone()
        self.modules["teacher_sppf"] = SPPF(int(512 * w * r), int(512 * w * r))
        self.modules["teacher_neck"] = Yolov8Neck(
            filters=[int(256 * w), int(512 * w), int(512 * w * r)],
            heads=hparams.heads,
            d=d,
        )
        self.modules["teacher_head"] = DetectionHead(
            hparams.num_classes,
            filters=(int(256 * w), int(512 * w), int(512 * w * r)),
            heads=hparams.heads,
        )

        # modify state dict to match teacher keys
        teacher_state_dict = OrderedDict([("teacher_"+k, v) for k,v in teacher_dict.items()])
        # load just teacher modules
        self.modules.load_state_dict(teacher_state_dict, strict=False)

        self.output_teacher = None # used to save output teacher

        self.lwf_params = self.hparams.lwf

        if len(self.lwf_params) > 2:
            # temp fix
            old_classes = [i for i in range(self.m_cfg.classes[0])]
            
            # yolov8 loss + custom lwf
            self.criterion = LwFLossV2(self.m_cfg, self.modules["head"], self.device,
                                       c1 = self.lwf_params[0], c2 = self.lwf_params[1],
                                       old_classes=old_classes, c3=self.lwf_params[2], classes=old_classes+self.m_cfg.classes)
        else:
            # yolov8 loss + l2 for lwf
            self.criterion = LwFLoss(self.m_cfg, self.modules["head"], self.device,
                                      lwf=self.lwf_params[0], new_classes=self.m_cfg.classes)

        # logger for mAPs
        self.logger = logger

        print("Number of parameters for each module:")
        print(self.compute_params())

    def forward(self, batch):
        """Runs the forward method by calling every module."""
        if self.modules.training:   

            preprocessed_batch = self.preprocess_batch(batch)
            backbone = self.modules["backbone"](
                preprocessed_batch["img"].to(self.device)
            )
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
            with torch.no_grad():
                backbone_teacher = self.modules["teacher_backbone"](batch["img"] / 255)

        if "sppf" in self.modules.keys():
            neck_input = list(backbone)[0:2]
            neck_input.append(self.modules["sppf"](backbone[2]))
            neck_input_teacher = list(backbone_teacher)[0:2]
            neck_input_teacher.append(self.modules["teacher_sppf"](backbone_teacher[2]))
        else:
            neck_input = backbone
            neck_input_teacher = backbone_teacher

        neck = self.modules["neck"](*neck_input)

        with torch.no_grad():
            neck_teacher = self.modules["teacher_neck"](*neck_input_teacher)
            self.output_teacher = self.modules["head"](neck_teacher)

        head = self.modules["head"](neck)

        return head

    def compute_loss(self, pred, batch):
        """Computes the loss."""
        preprocessed_batch = self.preprocess_batch(batch)

        lossi_sum, lossi = self.criterion(
            pred,
            preprocessed_batch,
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