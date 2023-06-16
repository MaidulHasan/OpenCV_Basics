import numpy as np
import matplotlib.pyplot as plt
import random
import cv2 as cv


def scale(img):
    img_w, img_h = img.shape[:2]
    image_center = (img_w / 2), (img_h / 2)

    scale_factor = round(random.uniform(0.3, 1.5), 2)

    transformation_matrix = cv.getRotationMatrix2D(image_center, 0, scale_factor)

    scaled_img = cv.warpAffine(img, transformation_matrix, (img_w, img_h))
    scaled_img_title = f"Scale factor: {scale_factor}"

    return scaled_img, scaled_img_title


def rotate(img):
    img_w, img_h = img.shape[:2]
    image_center = (img_w / 2), (img_h / 2)

    rotation_angle = np.random.randint(10, 320)

    transformation_matrix = cv.getRotationMatrix2D(image_center, rotation_angle, 1)

    rotated_img = cv.warpAffine(img, transformation_matrix, (img_w, img_h))
    rotated_img_title = f"Rotate (Angle (CCW): {rotation_angle})"

    return rotated_img, rotated_img_title


def translate(img):
    img_width, img_height = img.shape[:2]
    Tx = np.random.randint(10, 50) / 100 * img_width
    Ty = np.random.randint(10, 50) / 100 * img_height

    T = np.float32([[1, 0, Tx], [0, 1, Ty]])

    translated_img = cv.warpAffine(img, T, (int(2 * img_width), int(1.5 * img_height)))
    translated_img_title = f"Translation (X shift: {Tx}, Y shift: {Ty})"

    return translated_img, translated_img_title


def plots(
    scaled_img,
    scaled_img_title,
    rotated_img,
    rotated_img_title,
    translated_img,
    translated_img_title,
):
    plt.subplot(1, 3, 1).imshow(scaled_img), plt.title(scaled_img_title)
    plt.subplot(1, 3, 2).imshow(rotated_img), plt.title(rotated_img_title)
    plt.subplot(1, 3, 3).imshow(translated_img), plt.title(translated_img_title)

    plt.show()


def transform(
    img_path=None, translate_img=True, scale_img=True, rotate_img=True, plot_result=True
):
    if img_path is None:
        print("Please provide valid image path.")
        exit()

    img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)

    if scale_img:
        scaled_img, scaled_img_title = scale(img)
    if rotate_img:
        rotated_img, rotated_img_title = rotate(scaled_img)
    if translate_img:
        translated_img, translated_img_title = translate(rotated_img)

    if plot_result:
        plots(
            scaled_img,
            scaled_img_title,
            rotated_img,
            rotated_img_title,
            translated_img,
            translated_img_title,
        )
