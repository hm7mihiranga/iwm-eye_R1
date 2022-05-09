#!/usr/bin/env python3

import cv2 as cv

from skimage.util.shape import view_as_windows
from skimage.util.shape import view_as_blocks
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import os

import numpy as np
import imblearn
import sklearn

from typing import List, Tuple

OUTPUT_DIR = 'output/'
TRAIN_DIR = 'train/'
TEST_DIR = 'test/'
WINDOW_SIZE = (7, 7)
STEP = 4
PREDICTED_STEP = 1


def preprocess(img: np.ndarray) -> np.ndarray:
    print('Image preprocessing...')
    # CLAHE
    ycrcb_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv.equalizeHist(ycrcb_img[:, :, 0])
    equalized = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)

    # Denoising
    #denoised = cv.fastNlMeansDenoisingColored(equalized, None, 25, 25, 25)

    cv.imwrite(OUTPUT_DIR + 'preprocessed' +
               str(preprocess.counter) + '.jpg', equalized)
    preprocess.counter += 1

    return equalized


preprocess.counter = 1


def split_image(img: np.ndarray, size: Tuple[int, int], step: int) -> np.ndarray:
    if size[0] % 2 == 0 or size[1] % 2 == 0:
        raise Exception('Must be odd!')
    # RGB img
    if len(img.shape) == 3:
        return view_as_windows(img, (size[0], size[1], 3), step=step)
    # Grayscale img
    else:
        return view_as_windows(img, (size[0], size[1]), step=step)


def extract_features(chunk: np.ndarray) -> np.ndarray:

    values = hog(chunk, channel_axis=-1,
                 pixels_per_cell=(3, 3), cells_per_block=(1, 1))
    mean = np.mean(chunk, axis=(0, 1))
    std = np.std(chunk, axis=(0, 1))

    chunk = cv.cvtColor(chunk, cv.COLOR_BGR2GRAY)
    moments = cv.moments(chunk)
    hu = cv.HuMoments(moments).flatten()

    arr = np.concatenate((values, mean, std, hu))
    return arr


def get_data(img: np.ndarray, step: int) -> List:
    print("Getting data...")
    splitted = split_image(img, WINDOW_SIZE, step)
    print(f'Splitted shape: {splitted.shape}')

    train_data = [extract_features(splitted[i, j, 0])
                  for i, j in np.ndindex(splitted.shape[:2])]

    print(f'Train data size: {len(train_data) * len(train_data[0])}')
    return train_data


def get_labels(img: np.ndarray, step: int) -> List:
    print("Getting labels...")
    splitted = split_image(img, WINDOW_SIZE, step)

    labels = []

    for i, j in np.ndindex(splitted.shape[:2]):
        center = (splitted[i, j].shape[0] // 2, splitted[i, j].shape[1] // 2)
        if splitted[i, j, center[0], center[1]] == 0:
            labels.append(0)
        else:
            labels.append(255)

    print(f'Labels size: {len(labels)}')
    return labels


def make_model(x, y):
    print(f'Making model... ({len(x)})')
    assert len(x) == len(y)

    model = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=800, min_samples_split=5,
                                   min_samples_leaf=2, max_features='auto', max_depth=None, bootstrap=True)

    # n_estimators = [int(x) for x in np.linspace(start=800, stop=2000, num=10)]
    # max_features = ['auto', 'sqrt']
    # max_depth = [int(x) for x in np.linspace(50, 150, num=11)]
    # max_depth.append(None)
    # min_samples_split = [5, 10, 15, 20, 25]
    # min_samples_leaf = [1, 2, 3, 4]
    # bootstrap = [True, False]

    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}

    # model_random = RandomizedSearchCV(
    #     estimator=model, param_distributions=random_grid, n_iter=120, cv=5, random_state=0, n_jobs=-1, verbose=1)

    model.fit(x, y)
    print('-----')
    print(model.score(x, y))
    print('-----')
    return model


def recover(labels, shape, step):
    img = np.zeros(shape, dtype=np.uint8)
    chunk_shape = split_image(img, WINDOW_SIZE, step).shape
    counter = 0

    for x in range(chunk_shape[0]):
        for y in range(chunk_shape[1]):
            x1, y1 = x * PREDICTED_STEP, y * PREDICTED_STEP
            x2, y2 = x1 + WINDOW_SIZE[0], y1 + WINDOW_SIZE[1]
            img[(x1 + x2) // 2, (y1 + y2) //
                2] = 255 if labels[counter] == 255 else 0
            counter += 1

    cv.imwrite(OUTPUT_DIR + 'recovered.jpg', img)


def scale(img):
    shape = (700, 462)
    # shape = (img[1].shape, img[0].shape)
    print(f'Resizing to {shape}...')
    return cv.resize(img, shape, cv.INTER_AREA)
    # return img


def read_images():
    images = sorted(os.listdir(TRAIN_DIR), key=lambda x: x.lower())
    img = images[::2]
    truth = images[1::2]

    assert len(img) == len(truth)

    img_arr = [scale(cv.imread(TRAIN_DIR + str(i))) for i in img]
    truth_arr = [scale(cv.imread(TRAIN_DIR + str(i), cv.IMREAD_GRAYSCALE))
                 for i in truth]

    return np.concatenate(img_arr), np.concatenate(truth_arr)


if __name__ == "__main__":
    train_img, train_truth = read_images()

    train_processed = preprocess(train_img)
    train_data = get_data(train_processed, step=STEP)
    train_labels = get_labels(train_truth, step=STEP)

    undersampler = RandomUnderSampler(random_state=0, sampling_strategy=0.5)
    x_train, y_train = undersampler.fit_resample(train_data, train_labels)

    model = make_model(x_train, y_train)

    test_img = scale(cv.imread(TEST_DIR + '02_h.jpg'))
    test_processed = preprocess(test_img)
    test_data = get_data(test_processed, step=PREDICTED_STEP)
    test_truth = scale(cv.imread(TEST_DIR + '02_h.tif', cv.IMREAD_GRAYSCALE))
    cv.imwrite(OUTPUT_DIR + 'mask.jpg', test_truth)
    test_labels = get_labels(test_truth, step=PREDICTED_STEP)

    predicted = model.predict(test_data)
    recovered = recover(predicted, test_truth.shape, step=PREDICTED_STEP)
    print(classification_report(test_labels, predicted))
