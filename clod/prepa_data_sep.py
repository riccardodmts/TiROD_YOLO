from typing import Dict, Union
import os

from torch.utils.data import DataLoader, ConcatDataset
from data.mybuild import build_yolo_dataset
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd


def get_class_ids(all_names : list[str], task_names : list[str]) -> list[str]:
    """
    Get list of ids (str format) for the given task names and all the possible class names.7

    :param all_names: list of all the class names e.g. all 80 COCO class names
    :param task_names: list of class names for current task

    :return: list of ids (str format) for the given task names
    """

    ids = [str(id) for  id, class_name in enumerate(all_names) if class_name in task_names]

    if len(ids) == 0:
        raise Exception("None of the task-class names appear in the original list!")
    
    return ids

def create_loaders(train_m_cfg: Dict, val_m_cfg : Dict, data_cfg: Dict, batch_size: int):
    """Creates DataLoaders for dataset specified in the configuration file.
    Refer to ... for how to select the proper configuration.

    Arguments
    ---------
    m_cfg : Dict
        Contains information about the training process (e.g., data augmentation).
    data_cfg : Dict
        Contains details about the data configurations (e.g., image size, etc.).
    batch_size : int
        Batch size for the training process.
    filters: list[str]
        List with two items: path to filter for training and path to filter for validation

    """

    mode = "train"

    train_set = build_yolo_dataset(
        train_m_cfg,
        data_cfg["train"],
        batch_size,
        data_cfg,
        mode=mode,
        rect=mode == "val"
    )

    print(f"Number of images for training: {len(train_set)}, {len(train_m_cfg.classes)} classes")

    train_loader = DataLoader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=getattr(train_set, "collate_fn", None),
    )

    mode = "val"

    val_set = build_yolo_dataset(
        val_m_cfg,
        data_cfg["val"],
        batch_size,
        data_cfg,
        mode=mode,
        rect=mode == "val"
    )
    
    n_val_classes = len(data_cfg["names"]) if val_m_cfg.classes is None else len(val_m_cfg.classes)
    print(f"Number of images for validation: {len(val_set)}, {n_val_classes} classes")

    val_loader = DataLoader(
        val_set,
        batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=getattr(val_set, "collate_fn", None),
    )

    return train_loader, val_loader

def path_to_string(path : Union[list[Path], Path])->list[str]:
    """Convert Path/list[Path] to list[str]"""
    if isinstance(path, list):
        if isinstance(path[0], str):
            return path

    if isinstance(path, Path):
        path = [path]
    else:
        path = [Path(path)]
    return [str(p) for p in path]

def get_dataloaders_task(all_class_names : list[str], task_class_names : list[str], m_cfg, data_cfg, hparams, old_class_names : list[str] = None, return_stats=False, is_cum=False):

    """Create loaders for the current task. Download data if needed."""

    data_cfg["train"] = path_to_string(data_cfg["train"])
    data_cfg["val"] = path_to_string(data_cfg["val"])

    if isinstance(data_cfg["train"], list) and is_cum:
        for i,path in enumerate(data_cfg["train"]):
            data_cfg["train"][i] = data_cfg["path"]+"/"+path



    m_cfg.imgsz = hparams.input_shape[-1]  # temp solution
    val_m_cfg = deepcopy(m_cfg)
    
    # filter labels during training
    m_cfg.classes = [int(class_id) for class_id in get_class_ids(all_class_names, task_class_names)]

    # for validation consider either all classes or the ones seen up to now
    if isinstance(old_class_names, list):

        new_classes = []
        for class_name in task_class_names:
            if class_name not in old_class_names:
                new_classes.append(class_name)

        classes_seen = old_class_names + new_classes
        val_m_cfg.classes = [int(class_id) for class_id in get_class_ids(all_class_names, classes_seen)]
    else:
        val_m_cfg.classes = None
        
    # create current task loaders
    train_loader, val_loader = create_loaders(m_cfg, val_m_cfg, data_cfg, hparams.batch_size)

    return train_loader, val_loader, m_cfg, val_m_cfg



class TaskGenerator:

    def __init__(self, m_cfg, data_cfg, hparams, datasets, classes_per_task):

        self.m_cfg = m_cfg
        self.data_cfg = data_cfg
        self.hparams = hparams

        self.datasets = datasets
        self.classes_per_task = classes_per_task

        self.parent_dir = str(data_cfg["path"]).split("/")[0]
        self.datasets_paths = []

        for dataset in self.datasets:
            self.datasets_paths.append(self.parent_dir + "/" + dataset)

        self.num_tasks = len(self.datasets)
        self.all_class_names = [self.data_cfg["names"][id] for id in sorted(list(self.data_cfg["names"].keys()))]


        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.counter <self.num_tasks:

            old_task_names = [] if self.counter == 0 else old_task_names
            epochs = self.hparams.epochs if self.counter == 0 else self.hparams.epochs_per_task
            print(self.data_cfg["val"])
            data_cfg = deepcopy(self.data_cfg)
            data_cfg["path"] = self.datasets_paths[self.counter]
            print(data_cfg["path"])

            train_loader, val_loader, m_cfg, val_m_cfg = get_dataloaders_task(self.all_class_names, 
                                                                            self.classes_per_task[self.counter],
                                                                            self.m_cfg, data_cfg, self.hparams,
                                                                            old_task_names)

            old_classes = [int(idx) for idx in get_class_ids(self.all_class_names, old_task_names)] if len(old_task_names)>0 else []
            classes = old_classes + [int(idx) for idx in get_class_ids(self.all_class_names, self.classes_per_task[self.counter])]
            print(self.data_cfg["val"])
            other_info = {"tr_cfg" : m_cfg, "val_cfg" : val_m_cfg,
                            "epochs" : epochs,
                            "old_classes" : old_classes,
                            "classes" : classes
                        }
            
            return train_loader, val_loader, other_info
        
        else:
            raise StopIteration



