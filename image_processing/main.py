#!/usr/bin/env python3
import cv2 as cv

from skimage.filters import frangi

import numpy as np

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.metrics import sensitivity_specificity_support

import os

####
IMAGE_PATH = 'examples/02_dr.JPG'
GROUND_TRUTH_PATH = 'examples/02_dr.tif'
OUTPUT_DIR = 'output/'
####


def remove_small_elements(image: np.ndarray, min_size: int) -> np.ndarray:
    components, output, stats, _ = cv.connectedComponentsWithStats(
        image, connectivity=8)

    sizes = stats[1:, -1]
    components -= 1

    result = np.zeros((output.shape))

    for i in range(0, components):
        if sizes[i] >= min_size:
            result[output == i + 1] = 255

    return result


def normalize(image: np.ndarray, factor: float) -> np.ndarray:
    image = image * (factor / image.max())
    return image


def remove_border(color_img: np.ndarray, image: np.ndarray) -> np.ndarray:
    hsv = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, 80])
    upper = np.array([255, 255, 255])
    mask = cv.inRange(hsv, lower, upper)

    return cv.bitwise_and(image, mask)


def process_input(path: str) -> np.ndarray:
    color_image = cv.imread(path)
    if color_image is None:
        raise FileNotFoundError(f'"{path}" not found')

    gray_image = color_image[:, :, 1]

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray_image)

    denoised = cv.fastNlMeansDenoising(equalized, None, 15, 25, 25)
    vessels = normalize(frangi(denoised), 255).astype(np.uint8)
    _, thresh = cv.threshold(vessels, 0, 255, cv.THRESH_BINARY)
    small_removed = remove_small_elements(thresh, 800).astype(np.uint8)
    end_result = remove_border(color_image, small_removed)

    return end_result


def is_binary(image: np.ndarray) -> bool:
    return ((image == 0) | (image == 255)).all()


def compare(image: np.ndarray, truth: np.ndarray) -> dict:
    if not is_binary(image):
        raise Exception('Image is not binary (0 or 255)')

    if not is_binary(truth):
        raise Exception('Ground truth is not binary (0 or 255')

    image = image.flatten()
    truth = truth.flatten()

    report = classification_report(truth, image, output_dict=True)

    accuracy = round(accuracy_score(truth, image), 2)
    sensivity = round(report['255']['recall'], 2)
    specifity = round(report['0']['recall'], 2)

    weighted_result = sensitivity_specificity_support(
        truth, image, average='weighted')

    weight_sensivity = round(weighted_result[0], 2)
    weight_specifity = round(weighted_result[1], 2)
    matrix = confusion_matrix(truth, image)

    return {
        'accuracy': accuracy,
        'sensivity': sensivity,
        'specifity': specifity,
        'weight_sensivity': weight_sensivity,
        'weight_specifity': weight_specifity,
        'matrix': matrix.flatten()
    }


def visualise(image: np.ndarray, truth: np.ndarray) -> np.ndarray:
    predicted_vessels = (image == 255)
    true_vessels = (truth == 255)

    predicted_correct = (predicted_vessels & true_vessels)

    green_predicted = np.zeros((*truth.shape, 3), dtype=np.uint8)
    green_predicted[true_vessels, :] = 255

    green_predicted[predicted_correct] = [0, 255, 0]

    return green_predicted


if __name__ == "__main__":
    img = process_input(IMAGE_PATH)
    truth = cv.imread(GROUND_TRUTH_PATH, cv.IMREAD_GRAYSCALE)

    comparison = compare(img, truth)
    visualisation = visualise(img, truth)

    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        print(f'"{OUTPUT_DIR}" directory has been created.')

    out_path = OUTPUT_DIR + 'output.jpg'
    cv.imwrite(out_path, img)
    print(f'"{out_path}" has been saved.')

    vis_path = OUTPUT_DIR + 'visualisation.jpg'
    cv.imwrite(vis_path, visualisation)
    print(f'"{vis_path}" has been saved.')

    print(comparison)
