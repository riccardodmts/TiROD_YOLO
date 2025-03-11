import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import resnet
from collections import OrderedDict
from argparse import Namespace
from torchvision import transforms
import math
import PIL
import yaml
import os
import cv2
from pathlib import Path
from copy import deepcopy
import numpy as np

from ultralytics.utils import LOGGER
from ultralytics.data.augment import Compose, Format, Instances, LetterBox, v8_transforms
from ultralytics.data.utils import LOGGER
from ultralytics.data.utils import LOGGER
from torchvision.models import resnet50
from sklearn.cluster import KMeans

class KmeansMemory:

    def __init__(self, capacity, first_dataset, device = "cpu"):

        namespace = Namespace(**{"data" : "imagenet", "arch" : "ResNet50"})

        self.capacity = capacity
        self.ntasks = 0

        self.img_paths_per_task = []
        self.labels_per_task = []

        self.samples_per_task = []

        self.augment = False
        self.hyp = first_dataset.hyp
        self.imgsz = first_dataset.hyp.imgsz
        self.use_segments = False
        self.use_keypoints = False
        self.use_obb = False
        self.prefix = ""
        self.rect = False
        self.data = first_dataset.data

        self.preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        self.model = resnet50(pretrained=True)
        self.model.eval()
        self.transforms = self.build_transforms(first_dataset.hyp)
        self.device = device

    def update_memory(self, dataset):

        num_imgs_per_task = math.floor(self.capacity/(self.ntasks + 1))
        nimgs_per_task_to_remove = math.ceil(num_imgs_per_task/self.ntasks) if self.ntasks > 0 else 0

        # remove images old tasks
        if self.ntasks > 0:
            self.remove_imgs(nimgs_per_task_to_remove)


        self.nsamples_per_task  = [len(task) for task in self.img_paths_per_task]
        self.nimages = sum(self.nsamples_per_task)

        
        if ((self.nimages+num_imgs_per_task) < self.capacity) and self.ntasks>0:
            self.add_images(dataset, (self.capacity-self.nimages))
        else:       # add images
            self.add_images(dataset, num_imgs_per_task)

        self.ntasks+=1

        self.nsamples_per_task  = [len(task) for task in self.img_paths_per_task]
        self.nimages = sum(self.nsamples_per_task)
        print(self.nimages)

        
    def add_images(self, yolodataset, num_imgs):

        enc_outs = []

        for path in yolodataset.im_files:
            img = PIL.Image.open(path)
            inp =  self.preprocess(img).unsqueeze(0)
            with torch.no_grad():
                enc_outs.append(self.model(inp).detach())

        enc_out_tensor = torch.cat(enc_outs, dim=0)
        kmeans = KMeans(n_clusters=num_imgs, random_state=42)
        kmeans.fit(enc_out_tensor.numpy())

        centroids = kmeans.cluster_centers_
        selected_indices = []

        for centroid in centroids:
            # Compute the distance between the centroid and all points
            distances = torch.norm(enc_out_tensor - torch.tensor(centroid), dim=1)
            # Find the index of the closest point
            closest_index = torch.argmin(distances).item()
            selected_indices.append(closest_index)

        self.img_paths_per_task.append([])
        self.labels_per_task.append([])

        for index in selected_indices:
            self.img_paths_per_task[-1].append(yolodataset.im_files[index])
            self.labels_per_task[-1].append(deepcopy(yolodataset.labels[index]))


    def remove_imgs(self, nimgs_per_task_to_remove, random=True):


        for i in range(self.ntasks):
            if random:
                indeces = np.arange(len(self.img_paths_per_task[i]))
                np.random.shuffle(indeces)
                temp = []
                temp_labels = []

                for index in indeces[:len(self.img_paths_per_task[i])-nimgs_per_task_to_remove]:
                    temp.append(self.img_paths_per_task[i][index]) 
                    temp_labels.append(deepcopy(self.labels_per_task[i][index]))

                self.img_paths_per_task[i] = deepcopy(temp)
                self.labels_per_task[i] = deepcopy(temp_labels)


    def __len__(self):
        return self.nimages
    
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_batch(self, batch_dim):

        indices = np.random.randint(0, self.capacity, batch_dim)
        list_samples = [self[i] for i in indices]

        return self.collate_fn(list_samples)
        
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment else 0.0
            hyp.mixup = hyp.mixup if self.augment else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms
        

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        cum = 0

        for i, nimgs in enumerate(self.nsamples_per_task):
            if index < (cum+nimgs):
                idx = index - cum
                task = i
                break
            cum+=nimgs   
   
        label = deepcopy(self.labels_per_task[task][idx])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(task, idx)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def load_image(self, task, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = None, self.img_paths_per_task[task][i], None
        if im is None:  # not cached in RAM
            if fn:  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                im = cv2.imread(f)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]  


    def update_labels_info(self, label):
        """Custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # We can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label
    

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


import os
import yaml
import random
from yaml import Loader, CDumper as Dumper
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from prepa_data_sep import  TaskGenerator, get_dataloaders_task
from data.replay_data import ReplayDataloader, ReplayMemory
import numpy as np

import micromind as mm
from trainer.TiROD_trainer import YOLOOurs
from micromind.utils import parse_configuration
from micromind.utils.yolo import load_config
import sys
import os
import sys
from TiROD_utils import CLODLoggerTiROD
from yaml import SafeDumper
import yaml
import pathlib
from copy import deepcopy 

data = {'deny': None, 'allow': None}

SafeDumper.add_representer(
    type(None),
    lambda dumper, value: dumper.represent_scalar(u'tag:yaml.org,2002:null', '')
  )

def modify_yaml(path, key, path_val):
    """Modify .yaml by changing val path.
    Return path to new .yaml"""

    with open(path) as f:
        doc = yaml.load(f, Loader=Loader)

    doc[key] = str(path_val)
    new_path = path.split(".")[0]+"v2.yaml"


    with open(new_path, 'w') as f:
        yaml.dump(doc, f, Dumper=Dumper)

    return new_path

def set_seed():
    """ set seed for reproducibility"""
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    set_seed()

    assert len(sys.argv) > 1, "Please pass the configuration file to the script."
    hparams = parse_configuration(sys.argv[1])
    if len(hparams.input_shape) != 3:
        hparams.input_shape = [
            int(x) for x in "".join(hparams.input_shape).split(",")
        ]  # temp solution
        print(f"Setting input shape to {hparams.input_shape}.")

    # get clod exp e.g. 15p1
    exp_type = hparams.exp
    # save statistics of classes for each task or not
    save_stats = hparams.save_stats

    m_cfg, data_cfg = load_config(hparams.data_cfg)

    m_cfg.imgsz = hparams.input_shape[-1]  # temp solution
    all_names = [
        'bag', 'bottle', 'cardboard box', 'chair', 'potted plant', 'traffic cone', 'trashcan', 'ball',
          'broom', 'garden hose', 'bucket', 'bycicle', 'gardening tool'
    ]
    tasks_names = [['bag', 'bottle', 'cardboard box', 'chair', 'potted plant', 'traffic cone', 'trashcan'],
                   ['bag', 'bottle', 'cardboard box', 'chair', 'potted plant', 'traffic cone', 'trashcan'],
                   [ 'bottle', 'chair', 'potted plant', 'ball', 'broom', 'garden hose'],
                   [ 'bottle', 'chair', 'potted plant', 'ball', 'broom', 'garden hose'],
                   ['chair', 'potted plant', 'ball', 'garden hose', 'bucket', 'bycicle'],
                   ['chair', 'potted plant', 'ball', 'garden hose', 'bucket', 'bycicle'],
                   ['cardboard box', 'chair', 'potted plant', 'trashcan', 'ball', 'broom', 'garden hose', 'bucket'],
                   ['cardboard box', 'chair', 'potted plant', 'trashcan', 'ball', 'broom', 'garden hose', 'bucket'],
                   ['ball', 'bucket', 'gardening tool'],
                   ['ball', 'bucket', 'gardening tool']
                   ]
    datasets = ["domain1high", "domain1low", "domain2high", "domain2low", "domain3high", "domain3low",
                 "domain4high", "domain4low", "domain5high", "domain5low"]
    datasets_seen = []



    for i,dataset in enumerate(datasets):

        datasets_seen.append(dataset)

        path = pathlib.Path(__file__).parent.resolve()
        data_cfg["path"] =  str(path) + "/datasets/" + dataset
        data_cfg["train"] = data_cfg["path"]+"/train.txt"
        data_cfg["val"] = data_cfg["path"]+"/test.txt"

        train_loader, val_loader, cfg, val_cfg = get_dataloaders_task(all_names, tasks_names[i], m_cfg, data_cfg, hparams, [])
        # define experiment folder for current task
        exp_folder = mm.utils.checkpointer.create_experiment_folder(
                        "tirodkmeans2", hparams.experiment_name+f"_task_{i}"
                        )   
        
        checkpointer = mm.utils.checkpointer.Checkpointer(
                        exp_folder, hparams=hparams, key="loss"
                        )

        # define logger for CLOD (one file per task with mAPs classes seen)
        logger = CLODLoggerTiROD("./tirodkmeans2", len(all_names), i, use_tensorboard=hparams.use_tensorboard)
        
        # modify cfg for validator (temp fix)  
        data_cfg_new_path = modify_yaml(hparams.data_cfg, "path", data_cfg["path"])

        if i>= 1:
            yolo_mind = YOLOOurs(m_cfg, hparams=hparams, data_cfg_path_val = data_cfg_new_path,
                              exp_folder = exp_folder, logger=logger, oldlabels=True)
            # load model previous task. TODO: select best instead of last
            yolo_mind.load_model_prev_task("./tirodkmeans2/", i-1)
            # if current task is not the first task, use replay memory
            task_tr_loader = ReplayDataloader(train_loader, memory)
        else:
            yolo_mind = YOLOOurs(m_cfg, hparams=hparams, data_cfg_path_val = data_cfg_new_path,
                              exp_folder = exp_folder, logger=logger)
            task_tr_loader = train_loader

        yolo_mind.train(
            epochs=hparams.epochs,  # number epochs based on current task
            datasets={"train": task_tr_loader, "val": val_loader},
            metrics=[],
            checkpointer=checkpointer,
            debug=hparams.debug,
            warmup=True
        )

        for j, task in enumerate(datasets_seen):
            val_data_cfg = deepcopy(data_cfg)
            val_data_cfg["path"] = str(path) + "/datasets/" + task
            data_cfg_new_path = modify_yaml(hparams.data_cfg, "path", val_data_cfg["path"])

            mAP, mAP50 = yolo_mind.evaluate(data_cfg_new_path)
            logger.log_TiROD(mAP, mAP50, f"at_task{i}-task_{j}")

        data_cfg_new_path = str(path) + "/cfg/data/TiROD_test.yaml"
        mAP, mAP50 = yolo_mind.evaluate(data_cfg_new_path)
        logger.log_end_task(mAP)


        

        # save model
        yolo_mind.save_last_model("./tirodkmeans2/", i)

        if i == 0:
            memory = KmeansMemory(hparams.replay_mem_size, train_loader.dataset)

        memory.update_memory(train_loader.dataset)
        print(len(memory))
