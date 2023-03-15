"""
Простые тесты
"""

import time
import numpy as np
import glob    # для перебора файлов в папке
from PIL import Image
import main_lib			# тут пара функций для удобной инициализации и детекции

import torch

def TEST_IMAGE(filename):
    """проверка на картинке с одним лицом"""
    global model_face_detect, mtcnn, model_face_recog, DEVICE
    orgimg = np.array( Image.open( filename ) )
    t0 = time.time()
    bboxes,points = model_face_detect.predict(orgimg)
    embs = main_lib.recognise_faces(mtcnn, model_face_recog, bboxes[0], orgimg, DEVICE)
    t1 = time.time()
    print(filename)
    print(f"frames: {1}; size {orgimg.shape[0]:4d}x{orgimg.shape[1]:4d}; face embeddings: {len(embs)}; time {t1-t0:6.3f} sec")



# todo: check avaliable device
DEVICE = 'cuda:0'
# DEVICE = 'cpu'
print(f"Device: {DEVICE}")


if DEVICE != 'cpu':
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        quit()
    else:
       print( "DEVICE: " + torch.cuda.get_device_name( torch.cuda.current_device() ))


model_face_detect, mtcnn, model_face_recog = main_lib.init_models(DEVICE)

jpgFilenamesList = glob.glob('test-images/*.jpg')
for file in jpgFilenamesList:
	TEST_IMAGE(file)
