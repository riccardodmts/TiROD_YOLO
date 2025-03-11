import torch
import torch.nn as nn
import torch.nn.functional as F

from micromind.utils.yolo import autopad, dist2bbox, make_anchors
from micromind.networks.yolo import Yolov8Neck, Darknet, DFL, Conv



class DetectionHeadSID(nn.Module):
    """Implements YOLOv8's detection head for SID.

    Arguments
    ---------
    nc : int
        Number of classes to predict.
    filters : tuple
        Number of channels of the three inputs of the detection head.
    heads : list, optional
        List indicating whether each detection head is active.
        Default: [True, True, True].
    """

    def __init__(self, nc=80, filters=(), heads=[True, True, True]):
        super().__init__()
        self.reg_max = 16
        self.nc = nc
        # filters = [f for f, h in zip(filters, heads) if h]
        self.nl = len(filters)
        self.no = nc + self.reg_max * 4
        self.stride = torch.tensor([8.0, 16.0, 32.0], dtype=torch.float16)
        assertion_error = """Expected at least one head to be active. \
            Please change the `heads` parameter to a valid configuration. \
            Every configuration other than [False, False, False] is a valid option."""
        assert heads != [False, False, False], " ".join(assertion_error.split())
        self.stride = self.stride[torch.tensor(heads)]
        c2, c3 = max((16, filters[0] // 4, self.reg_max * 4)), max(
            filters[0], min(self.nc, 104)
        )  # channels
        

        self.cv2_1 = nn.ModuleList(Conv(x, c2, 3) for x in filters)
        self.first_reg_conv = []  # list to store output first reg conv, one item per head

        self.cv2_2 = nn.ModuleList(Conv(c2, c2, 3) for i in range(self.nl))
        self.second_reg_conv = []  # list to store output second reg conv, one item per head

        self.cv2_3 = nn.ModuleList(nn.Conv2d(c2, 4 * self.reg_max, 1) for i in range(self.nl))

        """
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in filters
        )"""


        self.cv3_1 = nn.ModuleList(Conv(x, c3, 3) for x in filters)
        self.first_cls_conv = []  # list to store output first cls conv, one item per head

        self.cv3_2 = nn.ModuleList(Conv(c3, c3, 3) for i in range(self.nl))
        self.second_cls_conv = []  # list to store output second cls conv, one item per head

        self.cv3_3 = nn.ModuleList(nn.Conv2d(c3, self.nc, 1) for i in range(self.nl))

        """
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in filters
        )
        """

        self.dfl = DFL(self.reg_max)

    def forward(self, x):
        """Executes YOLOv8 detection head.

        Arguments
        ---------
        x : list[torch.Tensor]
            Input to the detection head.
            In the YOLOv8 standard implementation it contains the three outputs of
            the neck. In a more general case it contains as many tensors as the number
            of active heads in the initialization.

        Returns
        -------
            Output of the detection head : torch.Tensor
        """
        """
        for i in range(self.nl):
            a = self.cv2[i](x[i])
            b = self.cv3[i](x[i])
            x[i] = torch.cat((a, b), dim=1)
        """

        self.first_reg_conv = []
        self.second_reg_conv = []
        self.first_cls_conv = []
        self.second_cls_conv = []

        # first conv for each head (both cls and reg)
        for i in range(self.nl):
            self.first_reg_conv.append(self.cv2_1[i](x[i]))
            self.first_cls_conv.append(self.cv3_1[i](x[i]))
        # first conv for each head (both cls and reg) 
        for i in range(self.nl):
            self.second_reg_conv.append(self.cv2_2[i](self.first_reg_conv[i]))
            self.second_cls_conv.append(self.cv3_2[i](self.first_cls_conv[i]))

        for i in range(self.nl):
            a = self.cv2_3[i](self.second_reg_conv[i])
            b = self.cv3_3[i](self.second_cls_conv[i])
            x[i] = torch.cat((a,b), dim=1)


        # this is needed for DDP, automatically set with .eval() and .train()
        if not self.training:
            self.anchors, self.strides = (
                xl.transpose(0, 1) for xl in make_anchors(x, self.stride, 0.5)
            )

            y = [(i.reshape(x[0].shape[0], self.no, -1)) for i in x]
            x_cat = torch.cat(y, dim=2)
            box, cls = x_cat[:, : self.reg_max * 4], x_cat[:, self.reg_max * 4 :]
            dbox = (
                dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1)
                * self.strides
            )
            z = torch.cat((dbox, nn.Sigmoid()(cls)), dim=1)
            return z, x
        else:
            return x
        

class YOLOv8SID(nn.Module):
    """Implements YOLOv8 network.

    Arguments
    ---------
    w : float
        Width multiple of the Darknet.
    r : float
        Ratio multiple of the Darknet.
    d : float
        Depth multiple of the Darknet.
    num_classes : int
        Number of classes to predict.
    """

    def __init__(self, w, r, d, num_classes=80, heads=[True, True, True]):
        super().__init__()
        self.net = Darknet(w, r, d)
        self.fpn = Yolov8Neck(
            filters=[int(256 * w), int(512 * w), int(512 * w * r)], heads=heads, d=d
        )
        self.head = DetectionHeadSID(
            num_classes,
            filters=(int(256 * w), int(512 * w), int(512 * w * r)),
            heads=heads,
        )

    def forward(self, x):
        """Executes YOLOv8 network.

        Arguments
        ---------
        x : torch.Tensor
            Input to the YOLOv8 network.

        Returns
        -------
            Output of the YOLOv8 network : torch.Tensor
        """
        x = self.net(x)
        x = self.fpn(*x)
        x = self.head(x)
        return x