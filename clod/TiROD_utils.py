from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np


def init_logger(path="./results"):

    logger = SummaryWriter(log_dir=path)

    return logger


class CLODLoggerTiROD():

    def __init__(self, results_dir, nc, task_id, losses = None, use_tensorboard=False):
        """
        Class for logging mAPs in a clod experiment.

        :param results_dir: path (str) to dir with results
        :param nc: number of calsses involved in the entire experiment (20 for VOC)
        :task_id: int for current task
        :use_tensorboard: use also tensorboard to log data
        """

        self.nc = nc
        self.results_dir = results_dir
        self.task_id = task_id
        self.use_tensorboard = use_tensorboard
        self.losses = losses

        self.header = ["epoch"] + ["mAP"] + [f"class_{id}" for id in range(nc)]

        if losses is not None:
            self.header += losses

        if use_tensorboard:
            self.tb_logger = SummaryWriter(log_dir=results_dir)
        if task_id ==0:
            pd.DataFrame(columns = self.header[2:]).to_csv(self.results_dir+f"/mAPs.csv", sep="\t", index=False)

    def on_task_start(self):
        """
        Init .csv file for current task
        """
        pd.DataFrame(columns = self.header).to_csv(self.results_dir+f"/mAPs_task_{self.task_id}.csv", sep="\t", index=False)
        pd.DataFrame(columns = self.header).to_csv(self.results_dir+f"/mAPs50_task_{self.task_id}.csv", sep="\t", index=False)

    def log(self, epoch, mAPs, mAP, aps50, mAP50, losses=None):
        """
        Log mAP50:90 and mAP50 for each class and mAP. In particular append array with epoch, mAP, mAPs all classes to csv for current task"""
        
        nlosses = 0 if losses is None else len(self.losses)
        to_save = np.zeros(self.nc + 2 + nlosses, dtype=np.float32)
        to_save[0] = epoch
        to_save[2:self.nc+2] = mAPs
        to_save[1] = mAP
        if losses is not None:
            to_save[self.nc + 2:] = losses


        pd.DataFrame(to_save.reshape(1,-1), columns=self.header).to_csv(self.results_dir+f"/mAPs_task_{self.task_id}.csv", sep="\t", header=None, index=False, mode="a")

        if self.use_tensorboard:
            self._log_on_tensorboard(epoch, mAPs, mAP, aps50, mAP50)

    def _log_on_tensorboard(self, epoch, mAPs, mAP, aps50, mAP50):
        """log on tensorboard"""

        for i, val in enumerate(mAPs):
            self.tb_logger.add_scalar(f"Task_{self.task_id}/mAP_class_{i}", val, epoch)

        for i, val in enumerate(aps50):
            self.tb_logger.add_scalar(f"Task_{self.task_id}/mAP50_class_{i}", val, epoch)
        

        self.tb_logger.add_scalar(f"Task_{self.task_id}/mAP50-95", mAP, epoch)
        self.tb_logger.add_scalar(f"Task_{self.task_id}/mAP50", mAP50, epoch)

    def log_TiROD(self, mAP, mAP50, file_name):

        header_mAP = [f"class_{id}" for id in range(len(mAP))]
        header_mAP50 = [f"class_{id}" for id in range(len(mAP50))]

        pd.DataFrame(columns = header_mAP).to_csv(self.results_dir+"/" + file_name + "mAP.csv", sep="\t", index=False)
        pd.DataFrame(columns = header_mAP50).to_csv(self.results_dir+"/" + file_name + "mAP50.csv", sep="\t", index=False)
        pd.DataFrame(mAP.reshape(1,-1), columns=header_mAP).to_csv(self.results_dir+"/" + file_name + "mAP.csv", sep="\t", header=None, index=False, mode="a")
        pd.DataFrame(mAP50.reshape(1,-1), columns=header_mAP50).to_csv(self.results_dir+"/" + file_name + "mAP50.csv", sep="\t", header=None, index=False, mode="a")

    def log_end_task(self, mAP):
        print("---------------------------------------------")
        pd.DataFrame(mAP.reshape(1,-1), columns=self.header[2:]).to_csv(self.results_dir+"/mAPs.csv", sep="\t", header=None, index=False, mode="a")

