import numpy as np
import matplotlib.pyplot as plt
import random
import cv2 as cv
from imutils import rotate_bound


def find_contours(img):
    thresh_val, thresh_img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
    cnt, hier = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    return cnt


def translate(img):
    img_width, img_height = img.shape[:2]
    Tx = int(np.random.randint(10, 50) / 100 * img_width)
    Ty = int(np.random.randint(10, 50) / 100 * img_height)

    T = np.float32([[1, 0, Tx], [0, 1, Ty]])

    dsize_width, dsize_height = Tx + img_width + 10, Ty + img_height + 10

    translated_img = cv.warpAffine(img, T, (dsize_width, dsize_height))
    translated_img_title = f"Translation (X shift: {Tx}, Y shift: {Ty})"

    return translated_img, translated_img_title


def scale(img):
    img_w, img_h = img.shape[:2]
    image_center = (img_w / 2), (img_h / 2)

    scale_factor = round(random.uniform(0.6, 1), 2)

    transformation_matrix = cv.getRotationMatrix2D(image_center, 0, scale_factor)

    scaled_img = cv.warpAffine(img, transformation_matrix, (img_w, 2 * img_h))
    scaled_img_title = f"Scale factor: {scale_factor}"

    return scaled_img, scaled_img_title


def rotate(img):
    rotation_angle = np.random.randint(10, 320)

    # under the hood, imutils.rotate_bound uses opencv to do this
    rotated_img = rotate_bound(img, -rotation_angle)
    rotated_img_title = f"Rotation Angle (CCW): {rotation_angle}"

    return rotated_img, rotated_img_title


def transform(
    img=None,
    translate_img=False,
    scale_img=False,
    rotate_img=False,
    translate_and_scale_img=False,
    translate_and_rotate_img=False,
    translate_scale_and_rotate_img=False,
):
    if img is None:
        print("Please provide valid image (Numpy Array).")
        exit()

    if translate_img:
        translated_img, translated_img_title = translate(img)
        translated_img_cnt = find_contours(translated_img)[0]

        return translated_img_cnt, translated_img_title

    if scale_img:
        scaled_img, scaled_img_title = scale(img)
        scaled_img_cnt = find_contours(scaled_img)[0]

        return scaled_img_cnt, scaled_img_title

    if rotate_img:
        rotated_img, rotated_img_title = rotate(img)
        rotated_img_cnt = find_contours(rotated_img)[0]

        return rotated_img_cnt, rotated_img_title

    if translate_and_scale_img:
        translated_img, translated_img_title = translate(img)
        translated_and_scaled_img, scaled_img_title = scale(translated_img)

        translated_and_scaled_img_cnt = find_contours(translated_and_scaled_img)[0]

        translated_and_scaled_img_title = f"{translated_img_title}\n{scaled_img_title}"

        return translated_and_scaled_img_cnt, translated_and_scaled_img_title

    if translate_and_rotate_img:
        translated_img, translated_img_title = translate(img)
        translated_and_rotated_img, rotated_img_title = rotate(translated_img)

        translated_and_rotated_img_cnt = find_contours(translated_and_rotated_img)[0]

        # # bringing the contour back to the center
        # translated_and_rotated_img_cnt = translated_and_rotated_img_cnt - np.array(
        #     [img.shape[0] / 2, img.shape[1] / 2], dtype=np.int32
        # )

        translated_and_rotated_img_title = (
            f"{translated_img_title}\n{rotated_img_title}"
        )

        return translated_and_rotated_img_cnt, translated_and_rotated_img_title

    if translate_scale_and_rotate_img:
        translated_img, translated_img_title = translate(img)
        translated_and_scaled_img, scaled_img_title = scale(translated_img)
        translated_scaled_and_rotated_img, rotated_img_title = rotate(
            translated_and_scaled_img
        )

        translated_scaled_and_rotated_img_cnt = find_contours(
            translated_scaled_and_rotated_img
        )[0]

        # # bringing the contour back to the center
        # translated_scaled_and_rotated_img_cnt = (
        #     translated_scaled_and_rotated_img_cnt
        #     - np.array([img.shape[0] / 2, img.shape[1] / 2], dtype=np.int32)
        # )

        translated_scaled_and_rotated_img_title = (
            f"{translated_img_title}\n{scaled_img_title}\n{rotated_img_title}"
        )

        return (
            translated_scaled_and_rotated_img_cnt,
            translated_scaled_and_rotated_img_title,
        )
