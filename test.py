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


def TEST_IMAGE(filename:str, faces_n:int = None):
    """проверка на картинке filename с одним лицом; 
    проверяет утверждение сравнивая faces_n (ожидаемое число лиц на картинке) с фактическим"""
    global model_face_detect, mtcnn, model_face_recog, DEVICE
    orgimg = np.array( Image.open( filename ) )
    t0 = time.time()
    bboxes,points = model_face_detect.predict(orgimg)
    embs = main_lib.recognise_faces(mtcnn, model_face_recog, bboxes[0], orgimg, DEVICE)
    t1 = time.time()
    print(f"{filename:40s}; size {orgimg.shape[0]:4d}x{orgimg.shape[1]:4d}; face embeddings: {len(embs):3d}; time {t1-t0:6.3f} sec")
    if faces_n is not None:
        assert len(embs) == faces_n, f"{filename} detected faces: {len(embs)}, should be {faces_n}"


# def TEST_IMAGES():
#     """тест распознования на картинках; не учитывает уникальность 
#     перед расширением в имене картинки должно быть указано число человек. Например: peoples-5.jpg"""
#     jpgFilenamesList = glob.glob('test-images/*.jpg')
#     for file in jpgFilenamesList:
#         persons = int(file[-5:-4])
#         TEST_IMAGE(file, persons)
#     print("[OK] TEST_IMAGES\n")


def TEST_IMAGES():
    """тест распознования на картинках; не учитывает уникальность 
    перед расширением в имене картинки должно быть указано число человек. Например: peoples-5.jpg"""
    # значения для мин ширины картинки 45px
    TEST_IMAGE('test-images/peoples-side-5.jpg', 2)
    TEST_IMAGE('test-images/peoples-1.jpg', 1)
    TEST_IMAGE('test-images/peoples-front-23.jpg', 23)
    TEST_IMAGE('test-images/peoples-11.jpg', 11)
    print("[OK] TEST_IMAGES\n")


def TEST_VIDEO(filename:str):
    """прогоняет видео, распознаёт лица, не проверяет утверждения"""
    vid = imageio.get_reader(filename,  'ffmpeg')
    print(f"file: {filename};  size: {vid.get_meta_data(1)['size']}; duration {vid.get_meta_data(1)['duration']}; fps {vid.get_meta_data(1)['fps']}")
    Faces = []

    t0 = time.time()
    fps = vid.get_meta_data(1)['fps']
    duration = vid.get_meta_data(1)['duration']
    vid_len = int(fps * duration)

    print(f"vid len {vid_len} frames")
    for i,frame in enumerate(vid):
        # TODO: перенести кадр сразу на видеокарту
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


def TEST_VIDEO_UNIQUE_FACES(filename:str, uniq_faces:int = None):
    """прогоняет видео, распознаёт лица, не проверяет утверждения"""
    vid = imageio.get_reader(filename,  'ffmpeg')
    print(f"file: {filename};  size: {vid.get_meta_data(1)['size']}; duration {vid.get_meta_data(1)['duration']}; fps {vid.get_meta_data(1)['fps']}")

    t0 = time.time()
    fps = vid.get_meta_data(1)['fps']
    duration = vid.get_meta_data(1)['duration']
    vid_len = int(fps * duration)
    known_faces = []

    for i,frame in enumerate(vid):
        bboxes,points = model_face_detect.predict(frame)
        
        # если найдены лица
        if bboxes[0]:
            # FIX: не на всех кадрах нормально лица обрабатываются
            faces = main_lib.recognise_faces(mtcnn, model_face_recog, bboxes[0], frame, DEVICE)
            if faces:       # если лица нашлись
                known_faces = main_lib.filter_new_faces(faces, known_faces)
        # print(i)
        if ( i % (vid_len//10) == 0): print("|",end="", flush=True)
        # print(f"frame: {i:4d}; faces detected: {len(bboxes[0])}")
    t1 = time.time()


    print(f"frames: {i}; time {t1-t0:.3f} sec; fps {i/(t1-t0):4.2f} uniq face embeddings: {len(known_faces)}")

    assert len(known_faces) == uniq_faces, f"{filename} detected faces: {len(known_faces)}, should be {uniq_faces}"

    print("[OK] TEST_VIDEO_UNIQUE_FACES\n")



def TEST_IMAGE_UNIQUE_FACES_COUNT(filename:str, uniq_faces:int = None):
    """Проверяет количество различных"""
    global model_face_detect, mtcnn, model_face_recog, DEVICE
    orgimg = np.array( Image.open( filename ) )
    t0 = time.time()
    bboxes,points = model_face_detect.predict(orgimg)
    embs = main_lib.recognise_faces(mtcnn, model_face_recog, bboxes[0], orgimg, DEVICE)
    t1 = time.time()
    print(f"{filename:40s}; size {orgimg.shape[0]:4d}x{orgimg.shape[1]:4d}; face embeddings: {len(embs):3d}; time {t1-t0:6.3f} sec")

    # поиск уникальных лиц
    known_faces = []
    known_faces = main_lib.filter_new_faces(embs, known_faces)
    if uniq_faces is not None:
        assert len(known_faces) == uniq_faces


def TEST_IMAGE_UNIQUE_FACES_COUNT_ALL():
    """тест распознования _уникальных лиц_ на картинках; 
    перед расширением в имене картинки должно быть указано число человек. Например: peoples-5.jpg"""
    TEST_IMAGE_UNIQUE_FACES_COUNT('test-images/peoples-side-5.jpg', 2)
    TEST_IMAGE_UNIQUE_FACES_COUNT('test-images/peoples-1.jpg', 1)
    TEST_IMAGE_UNIQUE_FACES_COUNT('test-images/peoples-front-23.jpg', 23)
    TEST_IMAGE_UNIQUE_FACES_COUNT('test-images/peoples-11.jpg', 11)
    print("[OK] TEST_IMAGE_UNIQUE_FACES_COUNT_ALL\n")




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


print(f"Parameters. MIN_FACE_SIZE {main_lib.MIN_FACE_SIZE:3d}; THRESHOLD_EUC: {main_lib.THRESHOLD_EUC:.4f}\n")
TEST_IMAGES()
TEST_IMAGE_UNIQUE_FACES_COUNT_ALL()

# wget https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4
# wget https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking-and-pause.mp4
# TEST_VIDEO('test-images/face-demographics-walking-and-pause.mp4')
# TEST_VIDEO('test-images/face-demographics-walking.mp4')


# FIX: на видео 7 ращзных людей, но программа находит 20. 
# Отрегулировать порог? 
# Фильтровать плохие детекции? 
# Слегка изменять представления знакомых лиц при их повторной стрече?
TEST_VIDEO_UNIQUE_FACES('test-images/face-demographics-walking-and-pause.mp4', 7)           # FAIL