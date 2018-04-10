# https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html

import os
import logging
import argparse
import cv2 as cv
from scipy import misc
import numpy as np


OPENCV_CASCADES_PATH = 'utils/face_detection/opencv_cascades'


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def read_img(img_path):
    log.info('Reading image: %s', img_path)
    if not os.path.exists(img_path):
        raise FileNotFoundError('There is no file: {}'.format(img_path))

    img = misc.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray


def crop_faces(img, gray, crop_size):
    if img.shape[1:] == crop_size:
        return img, gray

    for x, y, w, h in found_faces(gray):
        yield img[y:y + h, x:x + w]


def found_faces(gray):
    face_cascade = cv.CascadeClassifier('{}/haarcascade_frontalface_default.xml'.format(OPENCV_CASCADES_PATH))
    eye_cascade = cv.CascadeClassifier('{}/haarcascade_eye.xml'.format(OPENCV_CASCADES_PATH))

    faces = face_cascade.detectMultiScale(gray, 1.3, 2)

    for (x, y, w, h) in faces:
        log.info('Found face with coordinates x={}, y={}, width={}, height={}'.format(x, y, w, h))
        roi_gray = gray[y:y + h, x:x + w]

        if len(eye_cascade.detectMultiScale(roi_gray)) > 0:
            log.info('Eyes found on face, yielding it.')
            yield x, y, w, h

        else:
            log.info('There are no eyes on face, skipping it.')


def resize_photo(img, img_size=None):
    if img_size is None:
        return img
    log.info('Resizing photo from %s to %s', img.shape, img_size)
    return cv.resize(img, img_size)


def save_img(img, path):
    log.info('Saving photo to %s', path)
    misc.imsave(path, img)


def main(img_path, img_size=None):
    img, gray = read_img(img_path)
    name = os.path.basename(img_path)
    filename, ext = name.rsplit('.')

    cropped_faces = crop_faces(img, gray, img_size)
    photos_path = []
    photos = []

    cropped_path = 'cropped_imgs'
    if not os.path.exists(cropped_path):
        os.mkdir(cropped_path)

    for idx, face in enumerate(cropped_faces):
        img_name = '{}/{}.{}.{}'.format(cropped_path, filename, idx, ext)
        resized_face = resize_photo(face, img_size)
        save_img(resized_face, img_name)
        photos_path.append(img_name)
        photos.append(resized_face)

    if len(photos) == 0:
        img_name = '{}/{}.{}.{}'.format(cropped_path, filename, 'orig', ext)
        resized_face = resize_photo(img, img_size)
        save_img(resized_face, img_name)
        photos_path.append(img_name)
        photos.append(np.array(resized_face, dtype=float))

    return photos, photos_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', dest='img_path', help='Image path', required=True)
    parser.add_argument('--width', dest='width', help='Result image width', default=200, type=int)
    parser.add_argument('--height', dest='height', help='Result image height', default=200, type=int)
    args = parser.parse_args()

    print(main(args.img_path, (args.width, args.height)))

