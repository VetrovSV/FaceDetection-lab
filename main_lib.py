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


# минимальная ширина и высота прямоугольника с лицом; с этим парамаетром можно поэкперементировать
# MIN_FACE_SIZE = 30	# для детекции - ОК, но слишком маленький размер для распознования
MIN_FACE_SIZE = 45		


# (первое приближение) евклидово расстояние  между представлениями ниже этого порога будет означать, что сравниваемые лица одни и те же 
THRESHOLD_EUC = 0.5 # По видео понятно, что нужно больше; ещё влияет размер лиц
# 13 лиц вместо 7 при 0.99
# 35 лиц вместо 7 при 0.5, 45px
# если больше 0.55, то путает азиатов на картинке с 11 людьми


def recognise_faces(mtcnn, model_face_recog, face_boxes:list, img:np.array, device:str = 'cpu', debug = False):
    """распознаёт лица людей на RGB картинке img в прямоугольниках face_boxes;
    если debug = True, то будет сохрянять плохик картинки
    Возвращает представления лиц"""
    face_embs = []
    for i,box in enumerate( face_boxes ):		# итерируемся по лицам
        x1,y1,x2,y2 = box
        face_img = img[y1:y2, x1:x2, :]
        if (x2-x1 >= MIN_FACE_SIZE) and (y2 - y1 >= MIN_FACE_SIZE):

	        face_img1 = mtcnn(face_img)        

	        if face_img1 is not None:		# в прямоугольнике нет лица
	            face_emb = model_face_recog( face_img1.to(device).unsqueeze(0) )
	            face_embs += [ face_emb ]
	        else: 
	        	if debug:			# если пошло что-то не так, то сохраним картинку с плохими лицами
		        	im = Image.fromarray(img)
		        	draw = ImageDraw.Draw(im)
		        	draw.rectangle( box, outline = (255, 0, 255))
			        im.save(f"{x1}-{x2}, {y1}-{y2} face.jpeg")
    return face_embs
    # можно попробовать ускорить функцию обрабатывая лица в параллельных процессах, см. https://pytorch.org/docs/stable/notes/multiprocessing.html


def recognise_faces_img(mtcnn, model_face_recog, face_boxes:list, img:np.array, device:str = 'cpu', debug = False):
    """распознаёт лица людей на RGB картинке img в прямоугольниках face_boxes; 
    если debug = True, то будет сохрянять плохик картинки
    Возвращает представления и изображения лиц"""
    face_embs = []	# список представлений лиц
    face_imgs = []  # список изображений лиц
    for i,box in enumerate( face_boxes ):		# итерируемся по лицам
        x1,y1,x2,y2 = box
        face_img = img[y1:y2, x1:x2, :]
        if (x2-x1 >= MIN_FACE_SIZE) and (y2 - y1 >= MIN_FACE_SIZE):

	        face_img1 = mtcnn(face_img)        

	        if face_img1 is not None:		# в прямоугольнике нет лица
	            face_emb = model_face_recog( face_img1.to(device).unsqueeze(0) )
	            face_embs += [ face_emb ]
	            face_imgs += [face_img1 ]
	        else: 
	        	if debug:			# если пошло что-то не так, то сохраним картинку с плохими лицами
		        	im = Image.fromarray(img)
		        	draw = ImageDraw.Draw(im)
		        	draw.rectangle( box, outline = (255, 0, 255))
			        im.save(f"{x1}-{x2}, {y1}-{y2} face.jpeg")
    return face_embs, face_imgs


def init_models(device:str = 'cpu'):
	"""инициализирует три модели: деткции, предобработки лиц, распознования; 
	скачивает веса (111 мб, если не скачены) нейроки для распознования"""
	model_face_detect = YoloDetector(target_size=720, min_face=30, device=device)         # по умолчанию device='cuda:0'
	# min_face=30 -- минимальный размер лица в пикселях
	# скачивание весов нейронки
	mtcnn = MTCNN(device=device)             # предобработка картинки
	model_face_recog = InceptionResnetV1(pretrained='casia-webface', device=device).eval()        # 111 Мб
	return model_face_detect, mtcnn, model_face_recog



# первое приближение для порога различий получено их тестовых картинок
def filter_new_faces(faces:list, known_faces:list, threshold = THRESHOLD_EUC):
	"""выдаёт те представления лиц из faces, что не похожи на known_faces"""
	if not faces: return []

	start = 0
	if len(known_faces) == 0:
		known_faces += [faces[0]]
		start = 1

	for face in faces[start:]:
		dists = calc_distances(face, known_faces)
		if torch.min( dists ) >= threshold:			# лицо не похоже на остальные
			known_faces += [ face ]
	return known_faces


# служебная функция
def get_dubs_faces(model_face_detect, mtcnn, model_face_recog, picture, threshold = THRESHOLD_EUC, device = 'cpu'):
	"""
img -- картинка с несколькими лицами
выдаёт список представлений дубликатов и картинки"""
	bboxes,points = model_face_detect.predict(picture)
	embs, fimgs = recognise_faces_img(mtcnn, model_face_recog, bboxes[0], picture, device)

	dubs_embs = []		# представления
	dubs_fimgs = []		# изображения


	dmatr = calc_distances_matrix(embs)

	# todo: переписать с использованием функций торча
	for i in range(len(dmatr)):		# перебор номеров представлений лиц
		if (dmatr[:, i] < threshold).any():	# если любой элемент в i-м столбце меньше  threshold
			dubs_embs += [ embs[i] ]
			dubs_fimgs += [ fimgs[i] ]
	
	return dubs_embs, dubs_fimgs



def calc_distances(face:torch.Tensor, faces:list[torch.Tensor]):
	"""возвращает расстояние от представления лица face до всех остальных """
	distances = torch.zeros( len(faces) )
	for i,f in enumerate(faces):
		distances[i] = ((f - face)**2).sum()

	return distances


def calc_distances_matrix(faces:list[torch.tensor]):
	"""Строит верхнюю треуголььную матрицу расстояний всех лицевых представлений до всех лицевых представлений"""
	n = len(faces)
	dmat = torch.zeros( (n,n) ).fill_(torch.nan)		# nan чтобы потом удобнее было отбирать значения
	for i in range(n):
		for j in range(i+1,n):
			dmat[i,i+1:] = calc_distances(faces[i], faces[i+1:])
	return dmat



