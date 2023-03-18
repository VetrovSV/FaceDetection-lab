"""
служебные функции
"""
import main_lib
import torchvision.transforms as T
import torch

def save_faces(imgs:list[torch.tensor], prefix:str = ''):
	"""
	сохраняет лица из тензоров (3,h,w) в отдельные картинки
	"""
	transform = T.ToPILImage()
	for i, img in enumerate(imgs):
		transform(img).save(f'{prefix}face{i:02d}.png')