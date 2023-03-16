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


def TEST_IMAGE(filename:str):
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
def TEST_VIDEO(filename:str):
    # wget https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4
    vid = imageio.get_reader(filename,  'ffmpeg')
    print(f"file: {filename};  size: {vid.get_meta_data(1)['size']}; duration {vid.get_meta_data(1)['duration']}; fps {vid.get_meta_data(1)['fps']}")
    Faces = []

    t0 = time.time()
    fps = vid.get_meta_data(1)['fps']
    duration = vid.get_meta_data(1)['duration']
    vid_len = int(fps * duration)

    print(f"vid len {vid_len} frames")
    for i,frame in enumerate(vid):
        bboxes,points = model_face_detect.predict(frame)
        
        # если найдены лица
        if bboxes[0]:
            # FIX: не на всех кадрах нормально лица обрабатываются
            faces = main_lib.recognise_faces(mtcnn, model_face_recog, bboxes[0], frame, DEVICE)
            Faces += faces
        # print(i)
        if ( i % (vid_len//10) == 0): print("|",end="", flush=True)
        # print(f"frame: {i:4d}; faces detected: {len(bboxes[0])}")

    t1 = time.time()

    print()
    print(f"frames: {i}; time {t1-t0:.3f} sec; fps {i/(t1-t0):4.2f} face embeddings: {len(Faces)}")

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

# wget https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4
# wget https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking-and-pause.mp4
TEST_VIDEO('test-images/face-demographics-walking-and-pause.mp4')
TEST_VIDEO('test-images/face-demographics-walking.mp4')