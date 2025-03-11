# TiROD_YOLO

## Setup
In addition to PyTorch, install the extra requirements by following these steps:
1. Install ```micromind``` with ```pip install git+https://github.com/fpaissan/micromind```.
2. You find a ```extra_requirements.txt``` file. Please run  ```pip install -r extra_requirements.txt``` to install extra requirements e.g.  ```ultralytics```.

to execute the best approach: ```python kmeans_TiROD.py cfg/yolov8-TiROD.py```



## Results

Results for **YOLOv8 nano**
| Method               | Final mAP ↑  | RSD ↑ | RPD ↑ |
|----------------------|------|-------|-------|
| Fine-Tuning           | 15.9% | 0.19  | 1.00 |
| SID                  | 17.1% | 0.24  | 1.00 |
| Replay               | 40.7% | 0.70  | 0.95  |
| K-Means Replay       | **41.3**% | **0.75** | **0.99** |
| **Joint Training [mAP]** |  **59%**  |
