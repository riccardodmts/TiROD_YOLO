
import os
import yaml
import random
from yaml import Loader, CDumper as Dumper
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

    # replay memory (empty)
    replay_mem = ReplayMemory(hparams.replay_mem_size)

    for i,dataset in enumerate(datasets):

        datasets_seen.append(dataset)

        path = pathlib.Path(__file__).parent.resolve()
        data_cfg["path"] =  str(path) + "/datasets/" + dataset
        data_cfg["train"] = data_cfg["path"]+"/train.txt"
        data_cfg["val"] = data_cfg["path"]+"/test.txt"

        train_loader, val_loader, cfg, val_cfg = get_dataloaders_task(all_names, tasks_names[i], m_cfg, data_cfg, hparams, [])
        # define experiment folder for current task
        exp_folder = mm.utils.checkpointer.create_experiment_folder(
                        "tirod3", hparams.experiment_name+f"_task_{i}"
                        )   
        
        checkpointer = mm.utils.checkpointer.Checkpointer(
                        exp_folder, hparams=hparams, key="loss"
                        )
        # define logger for CLOD (one file per task with mAPs classes seen)
        logger = CLODLoggerTiROD("./tirod3", len(all_names), i, use_tensorboard=hparams.use_tensorboard)


        
        # modify cfg for validator (temp fix)  
        data_cfg_new_path = modify_yaml(hparams.data_cfg, "path", data_cfg["path"])

        if i>= 1:
            yolo_mind = YOLOOurs(m_cfg, hparams=hparams, data_cfg_path_val = data_cfg_new_path,
                              exp_folder = exp_folder, logger=logger, oldlabels=True)
            # load model previous task. TODO: select best instead of last
            yolo_mind.load_model_prev_task("./tirod2/", i-1)
            # if current task is not the first task, use replay memory
            task_tr_loader = ReplayDataloader(train_loader, replay_mem)
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
        yolo_mind.save_last_model("./tirod3/", i)

        replay_mem.update(train_loader.dataset)