"""
Простые тесты
"""

import time
import numpy as np
import glob    # для перебора файлов в папке
from PIL import Image

import torch

import imageio          # читает видео
# для сохранения тензоров в картинки

import main_lib			# тут пара функций для удобной инициализации и детекции
import util             # служебные функции для работы с лицами и т.



def TEST_IMAGE(filename:str, faces_n:int = None):
    """Проверка на картинке filename с одним лицом;
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
#     """тест распознавания на картинках; не учитывает уникальность
#     перед расширением в имени картинки должно быть указано число человек. Например: peoples-5.jpg"""
#     jpgFilenamesList = glob.glob('test-images/*.jpg')
#     for file in jpgFilenamesList:
#         persons = int(file[-5:-4])
#         TEST_IMAGE(file, persons)
#     print("[OK] TEST_IMAGES\n")


def TEST_IMAGES():
    """Тест распознования на картинках; не учитывает уникальность
    перед расширением в имени картинки должно быть указано число человек. Например: peoples-5.jpg"""
    # значения для мин ширины картинки 45px
    TEST_IMAGE('test-images/peoples-side-5.jpg', 2)
    TEST_IMAGE('test-images/peoples-1.jpg', 1)
    TEST_IMAGE('test-images/peoples-front-23_2.jpg', 23+2)      # + 2 повтора другой фотографии того же человека
    TEST_IMAGE('test-images/peoples-11.jpg', 11)
    print("[OK] TEST_IMAGES\n")



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
        assert len(known_faces) == uniq_faces, f"{filename} detected faces: {len(known_faces)}, should be {uniq_faces}"


def TEST_IMAGE_UNIQUE_FACES_COUNT_ALL():
    """Тест распознования _уникальных лиц_ на картинках;
    перед расширением в имене картинки должно быть указано число человек. Например: peoples-5.jpg"""
    TEST_IMAGE_UNIQUE_FACES_COUNT('test-images/peoples-side-5.jpg', 2)
    TEST_IMAGE_UNIQUE_FACES_COUNT('test-images/peoples-1.jpg', 1)
    TEST_IMAGE_UNIQUE_FACES_COUNT('test-images/peoples-front-23_2.jpg', 23)       # тут есть три повтора, они не в счёт
    TEST_IMAGE_UNIQUE_FACES_COUNT('test-images/peoples-11.jpg', 11)
    print("[OK] TEST_IMAGE_UNIQUE_FACES_COUNT_ALL\n")


def TEST_DIST_MITRIX(filename: str, check = lambda my_min: my_min >= main_lib.THRESHOLD_EUC ):
    """"Проверяет матрицу расстояний между векторными представлениями людей"""
    global model_face_detect, mtcnn, model_face_recog, DEVICE
    orgimg = np.array( Image.open( filename ) )

    bboxes,points = model_face_detect.predict(orgimg)
    embs = main_lib.recognise_faces(mtcnn, model_face_recog, bboxes[0], orgimg, DEVICE)
    matrix = main_lib.calc_distances_matrix(embs)      # матрица расстояний представлений всех лиц от каждого до каждого

    # print(matrix) # для отладки

    # выбор всех кто не NaN, на всех картинках лица разные, поэтому каждая разность долюна быть больше пороговой
    min = matrix[~matrix.isnan()].min()     # минимальные различия между лицами

    assert check(min) , f"Faces {filename:40s} not different"




def TEST_SIMPLE_VIDEO(filename:str):
    """Прогоняет видео, распознаёт лица, не проверяет утверждения"""
    vid = imageio.get_reader(filename,  'ffmpeg')
    print(f"file: {filename};  size: {vid.get_meta_data(1)['size']}; duration {vid.get_meta_data(1)['duration']}; fps {vid.get_meta_data(1)['fps']}")
    Faces = []

    t0 = time.time()
    fps = vid.get_meta_data(1)['fps']
    duration = vid.get_meta_data(1)['duration']
    vid_len = int(fps * duration)

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

    print("[OK] TEST_SIMPLE_VIDEO\n")


def TEST_VIDEO_UNIQUE_FACES(filename:str, uniq_faces:int = None):
    """Прогоняет видео, считает уникальные лица, не проверяет утверждения"""
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
        if i % (vid_len//10) == 0: print("|", end="", flush=True)
        # print(f"frame: {i:4d}; faces detected: {len(bboxes[0])}")
    t1 = time.time()

    # для отладки
    print(main_lib.calc_distances_matrix(known_faces))

    print(f"frames: {i}; time {t1-t0:.3f} sec; fps {i/(t1-t0):4.2f} uniq face embeddings: {len(known_faces)}")

    assert len(known_faces) == uniq_faces, f"{filename} detected faces: {len(known_faces)}, should be {uniq_faces}"

    print("[OK] TEST_VIDEO_UNIQUE_FACES\n")



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

# простая проверка распознавания людей на картинках, без учёта их уникальности
TEST_IMAGES()
# простая проверка распознавания людей на картинках, с учётом уникальности
TEST_IMAGE_UNIQUE_FACES_COUNT_ALL()

# проверка построения матриц расстояний. Расстояния между лицами разных людей должны быть не ниже THRESHOLD_EUC
TEST_DIST_MITRIX('test-images/peoples-side-5.jpg')
TEST_DIST_MITRIX('test-images/peoples-front-23_2.jpg', check = lambda min: min < 0.5)      # тут есть одни и те же люди

# для отладки
# orgimg = np.array( Image.open( 'test-images/peoples-11.jpg' ) )
# bboxes,points = model_face_detect.predict(orgimg)
# embs, imgs = main_lib.recognise_faces_img(mtcnn, model_face_recog, bboxes[0], orgimg, DEVICE)
# matrix = main_lib.calc_distances_matrix(embs)       # матрица расстояний представлений всех лиц от каждого до каждого
# print(matrix) # для отладки
# util.save_faces(imgs)


TEST_DIST_MITRIX('test-images/peoples-11.jpg')
print("[OK] TEST_DIST_MATRIX")


# простая проверка на вычленение похожих
dubs_embs, dubs_fimgs = main_lib.get_dubs_faces(model_face_detect, mtcnn, model_face_recog, 
                        picture = np.array( Image.open( 'test-images/peoples-front-23_2.jpg' ) ))
print( len(dubs_embs))
assert len(dubs_embs) == 2
print("[OK] TEST_SIMPLE_GET_DUBS_FACES")

# wget https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4
# wget https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking-and-pause.mp4
TEST_SIMPLE_VIDEO('test-images/face-demographics-walking-and-pause.mp4')
TEST_SIMPLE_VIDEO('test-images/face-demographics-walking.mp4')


# FIX: на видео 7 разных людей, но программа находит 20.
# Отрегулировать порог? 
# Фильтровать плохие детекции? 
# Слегка изменять представления знакомых лиц при их повторной встрече?
TEST_VIDEO_UNIQUE_FACES('test-images/face-demographics-walking-and-pause.mp4', 7)           # FAIL


# TODO: set no grad for tensors