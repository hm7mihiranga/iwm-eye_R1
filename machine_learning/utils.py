#!/usr/bin/env python3

import cv2 as cv
import numpy as np


def normalize(image: np.ndarray, factor: float) -> np.ndarray:
    image = image * (factor / image.max())
    return image


def preprocess(img: np.ndarray) -> np.ndarray:
    # CLAHE
    ycrcb_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv.equalizeHist(ycrcb_img[:, :, 0])
    equalized = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)

    # Denoising
    denoised = cv.fastNlMeansDenoisingColored(equalized, None)

    # Normalizing 0-255
    normalized = normalize(denoised, 255).astype(np.uint8)

    return normalized


if __name__ == "__main__":
    img = cv.imread('output/01_dr.JPG')
    assert img is not None
    pre = preprocess(img)
    cv.imwrite('test/01_dr.JPG', pre, [int(cv.IMWRITE_JPEG_QUALITY), 100])
