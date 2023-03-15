import torch
from face_detector import YoloDetector		# из yoloface

import numpy as np
from PIL import Image

from scipy.spatial.distance import euclidean


# todo: check avaliable device
DEVICE = 'cuda:0'
# DEVICE = 'cpu'
# 
model_face_detect = YoloDetector(target_size=720, min_face=30, device=DEVICE)         # по умолчанию device='cuda:0'
# min_face=30 -- минимальный размер лица в пикселях

from facenet_pytorch import MTCNN, InceptionResnetV1
# resnet = InceptionResnetV1(pretrained='vggface2').eval()                     # не получается скачать
model_face_recog = InceptionResnetV1(pretrained='casia-webface', device=DEVICE).eval()        # 111 Мб
mtcnn = MTCNN(device=DEVICE)             # предобработка картинки