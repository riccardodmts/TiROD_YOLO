import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

"""YOLOv8 loss + SID loss"""
class SIDLoss(v8DetectionLoss):
    def __init__(self, h, m, device, sid_int=1.0, sid_cls_out=None, new_classes=[], replay=False, old_classes = None):  # model must be de-paralleled
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.sid_int = sid_int
        self.sid_cls_out = sid_cls_out if sid_cls_out is not None else 0
        self.new_classes = new_classes

        self.old_classes = old_classes
        old_classes = [clas for clas in self.old_classes if clas not in self.new_classes]
        self.old_classes = old_classes
        if len(old_classes) == 0:
            self.sid_cls_out = 0
        self.replay = False  # True, if a replay memory is used

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # SID loss: L2
        self.sid_loss = torch.nn.MSELoss()

        self.last_int_loss = 0
        self.last_yolo_loss = 0
        self.last_out_loss = 0

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        Preprocesses the target counts and matches with the input batch size
        to output a tensor.
        """
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """
        Decode predicted object bounding box coordinates from anchor points and
        distribution.
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, student_int_output, teacher_int_output, teacher_output=None):
        """
        Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        teacher_output = teacher_output[1] if isinstance(teacher_output, tuple) else teacher_output
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )

        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.clone().detach().sigmoid(),
            (pred_bboxes.clone().detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE


        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor


            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        """Compute SID intermediate loss"""
        sid_int_loss = 0
        batch_idx = batch_size//2 if self.replay else 0 

        for student_item, teacher_item in zip(student_int_output, teacher_int_output):
            sid_int_loss += self.sid_loss(student_item[batch_idx:], teacher_item[batch_idx:])


        # Inter-Related Distillation
        inter_related_distill_loss = 0

        idxs = np.random.randint(student_int_output[0].shape[0], size=2)

        if (student_int_output[0].shape[0]) > 1:
            for i in range(3):

                temp_student_reg = self.sid_loss(student_int_output[i*2][idxs[0]], student_int_output[i*2][idxs[1]])
                temp_student_cls = self.sid_loss(student_int_output[i*2+1][idxs[0]], student_int_output[i*2+1][idxs[1]])

                temp_teacher_reg = self.sid_loss(teacher_int_output[i*2][idxs[0]], teacher_int_output[i*2][idxs[1]])
                temp_teacher_cls = self.sid_loss(teacher_int_output[i*2+1][idxs[0]], teacher_int_output[i*2+1][idxs[1]])

                inter_related_distill_loss += (temp_teacher_reg - temp_student_reg)**2 + (temp_student_cls - temp_teacher_cls)**2


        cls_idx = self.reg_max * 4
        filter_idx = self.reg_max * 4 + self.new_classes[0]

        lwf_loss_cls = 0  # lwf classification output

        if teacher_output is not None and self.sid_cls_out>0:

            for i in range(3):
                if self.old_classes:
                    lwf_loss_cls += self.sid_loss(feats[i][batch_idx:, self.old_classes, :,:], teacher_output[i][batch_idx:, self.old_classes, :,:].detach())
                else:
                    lwf_loss_cls += self.sid_loss(feats[i][batch_idx:, cls_idx : filter_idx, :,:], teacher_output[i][batch_idx:, cls_idx : filter_idx, :,:].detach())
            
            lwf_loss_cls /= 3

        sid_int_loss /= len(teacher_int_output)  # mean over several intermediate outputs
        #print(f"{loss.sum()},  {sid_int_loss},  {lwf_loss_cls}")
        self.last_yolo_loss = loss.sum().item()
        self.last_int_loss = sid_int_loss.item()
        self.last_out_loss = lwf_loss_cls.item() if lwf_loss_cls else 0

        total_loss = loss.sum() * batch_size + (self.sid_int * sid_int_loss * batch_size + self.sid_cls_out * lwf_loss_cls * batch_size + inter_related_distill_loss * 2) *5


        return total_loss, loss.detach()