"""
тут основные высокоуровневые функции для инициализации, детекции
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import euclidean
import torch
import sys	# для подключения файлов детектора yoloface
sys.path.append('yoloface/')
from face_detector import YoloDetector		# из папки yoloface

from facenet_pytorch import MTCNN, InceptionResnetV1


MIN_SIZE = 30		# минимальная ширина и высота прямоугольника с лицом


# FIX: тут нужны ещё проверки, т.к. не всякие лица корректно обрабатываются (см. тест с видео)
def recognise_faces(mtcnn, model_face_recog, face_boxes:list, img:np.array, device:str = 'cpu', debug = False):
    """распознаёт лица людей на RGB картинке img в прямоугольниках face_boxes;
    если debug = True, то будет сохрянять плохик картинки"""
    face_embs = []
    for i,box in enumerate( face_boxes ):		# итерируемся по лицам
        x1,y1,x2,y2 = box
        face_img = img[y1:y2, x1:x2, :]
        if (x2-x1 >= MIN_SIZE) and (y2 - y1 >= MIN_SIZE):

	        face_img1 = mtcnn(face_img)        

	        if face_img1 is not None:		# в прямоугольнике нет лица
	            face_emb = model_face_recog( face_img1.unsqueeze(0).to(device) )
	            face_embs += [ face_emb ]
	        else: 
	        	if debug:			# если пошло что-то не так, то сохраним картинку с плохими лицами
		        	im = Image.fromarray(img)
		        	draw = ImageDraw.Draw(im)
		        	draw.rectangle( box, outline = (255, 0, 255))
			        im.save(f"{x1}-{x2}, {y1}-{y2} face.jpeg")
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