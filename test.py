"""
Простые тесты
"""

import time
import numpy as np
import glob    # для перебора файлов в папке
from PIL import Image
import main_lib			# тут пара функций для удобной инициализации и детекции

import torch

import imageio          # читает видео


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


def TEST_IMAGES():
    # тест на картинках
    jpgFilenamesList = glob.glob('test-images/*.jpg')
    for file in jpgFilenamesList:
        TEST_IMAGE(file)
    print("[OK] TEST_IMAGES\n")


# FIX: тут где-то косяк в коде
def TEST_VIDEO():
    # wget https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4
    vid = imageio.get_reader('test-images/face-demographics-walking.mp4',  'ffmpeg')
    print(vid.get_meta_data(1))
    faces = []

    t0 = time.time()
    i = 0
    for frame in vid:
        bboxes,points = model_face_detect.predict(frame)
        
        # если найдены лица
        if bboxes[0]:
            # FIX: не на всех кадрых нормально лица обрабатываются
            try:
                faces = main_lib.recognise_faces(mtcnn, model_face_recog, bboxes[0], frame)
            except AttributeError as e:
                print(f"Error. frame{i}", end = "; ")
                print(bboxes[0])
            faces += faces
        i = i + 1
        # print(f"frame: {i:4d}; faces detected: {len(bboxes[0])}")

    t1 = time.time()

    print(f"frames: {i}; time {t1-t0:.3f} sec; face embeddings: {len(faces)}")

    print("[OK] TEST_VIDEO\n")



# DEVICE = 'cuda:0'
DEVICE = 'cpu'
print(f"Device: {DEVICE}")

if DEVICE != 'cpu':
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        quit()
    else:
       print( "DEVICE: " + torch.cuda.get_device_name( torch.cuda.current_device() ))



# инициализация моделей
model_face_detect, mtcnn, model_face_recog = main_lib.init_models(DEVICE)

TEST_IMAGES()

TEST_VIDEO()    # fix: тут где-то косяк в коде