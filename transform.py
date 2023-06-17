import numpy as np
import matplotlib.pyplot as plt
import random
import cv2 as cv
from funcs import matplotlib_imshow
import math


def scale(img):
    img_w, img_h = img.shape[:2]
    image_center = (img_w / 2), (img_h / 2)

    scale_factor = round(random.uniform(0.3, 1), 2)

    transformation_matrix = cv.getRotationMatrix2D(image_center, 0, scale_factor)

    scaled_img = cv.warpAffine(img, transformation_matrix, (img_w, img_h))
    scaled_img_title = f"Scale factor: {scale_factor}"

    return scaled_img, scaled_img_title


def rotate(img):
    img_w, img_h = img.shape[:2]
    image_center = (img_w / 2), (img_h / 2)
    diagonal = math.sqrt(img_h**2 + img_w**2)

    rotation_angle = np.random.randint(10, 320)

    transformation_matrix = cv.getRotationMatrix2D(image_center, rotation_angle, 1)

    rotated_img = cv.warpAffine(img, transformation_matrix, (diagonal, diagonal))
    rotated_img_title = f"Rotate (Angle (CCW): {rotation_angle})"

    return rotated_img, rotated_img_title


def translate(img):
    img_width, img_height = img.shape[:2]
    Tx = int(np.random.randint(10, 50) / 100 * img_width)
    Ty = int(np.random.randint(10, 50) / 100 * img_height)

    T = np.float32([[1, 0, Tx], [0, 1, Ty]])

    dsize_width, dsize_height = Tx + img_width + 10, Ty + img_height + 10

    translated_img = cv.warpAffine(img, T, (dsize_width, dsize_height))
    translated_img_title = f"Translation (X shift: {Tx}, Y shift: {Ty})"

    return translated_img, translated_img_title


def transform(
    img=None,
    translate_img=True,
    scale_img=False,
    rotate_img=False,
    plot_result=True,
):
    if img is None:
        print("Please provide valid image (Numpy Array).")
        exit()

    if scale_img:
        scaled_img, scaled_img_title = scale(img)

        if plot_result:
            matplotlib_imshow(scaled_img_title, scaled_img)

        return scaled_img, scaled_img_title

    if rotate_img:
        rotated_img, rotated_img_title = rotate(img)
        if plot_result:
            matplotlib_imshow(rotated_img_title, rotated_img)

        return rotated_img, rotated_img_title

    if translate_img:
        translated_img, translated_img_title = translate(img)
        if plot_result:
            matplotlib_imshow(translated_img_title, translated_img)

        return translated_img, translated_img_title
