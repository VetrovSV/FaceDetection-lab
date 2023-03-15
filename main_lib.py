"""
тут основные высокоуровневые функции для инициализации, детекции
"""
import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean
import torch
import sys	# для подключения файлов детектора yoloface
sys.path.append('yoloface/')
from face_detector import YoloDetector		# из папки yoloface

from facenet_pytorch import MTCNN, InceptionResnetV1



def recognise_faces(mtcnn, model_face_recog, face_boxes:list, img:np.array, device:str = 'cpu'):
    """распознаёт лица людей на RGB картинке img в прямоугольниках face_boxes"""
    face_embs = []
    for i,box in enumerate( face_boxes ):
        x1,y1,x2,y2 = box
        face_img = img[y1:y2, x1:x2, :]
        face_img = mtcnn(face_img)
        face_emb = model_face_recog( face_img.unsqueeze(0).to(device) )
        face_embs += [ face_emb ]
    return face_embs


def init_models(device:str = 'cpu'):
	"""инициализирует три модели: деткции, предобработки лиц, распознования; 
	скачивает веса (111 мб, если не скачены) нейроки для распознования"""
	model_face_detect = YoloDetector(target_size=720, min_face=30, device=device)         # по умолчанию device='cuda:0'
	# min_face=30 -- минимальный размер лица в пикселях
	# скачивание весов нейронки
	mtcnn = MTCNN(device=device)             # предобработка картинки
	model_face_recog = InceptionResnetV1(pretrained='casia-webface', device=device).eval()        # 111 Мб
	return model_face_detect, mtcnn, model_face_recog