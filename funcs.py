import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# ------------------------------------------------------------
### matplotlib_imshow
# ------------------------------------------------------------


# converting color mode to RGB and displaying the image as matplotlib figure
def matplotlib_imshow(
    title="", img=None, fig_h=7, cv_colorspace_conversion_flag=cv.COLOR_BGR2RGB
):
    """
    title: plot title (to be shown)
    img: image to plot
    fig_h = 7: figure height is set to 7
    cv_colorspace_conversion_flag = cv.COLOR_BGR2RGB: colorspace conversion
    """

    # tinkering with size
    try:
        img_width, img_height = img.shape[0], img.shape[1]
        aspect_ratio = img_width / img_height
        plt.figure(figsize=(fig_h * aspect_ratio, fig_h))
    except AttributeError:
        print(
            "None Type image. Correct_syntax is, matplotlib_imshow(img_title, img, fig_h, cv_colorspace_conversion_flag)."
        )

    # actual code for displaying the image
    plt.imshow(cv.cvtColor(img, cv_colorspace_conversion_flag))
    plt.title(title)
    plt.show()


# ------------------------------------------------------------
### Get grayscale canvas
# ------------------------------------------------------------


def get_canvas(shape, color_code=(80, 80, 80)):
    return np.full(shape, fill_value=color_code, dtype=np.uint8)


# ------------------------------------------------------------
### Auto Canny Edge detector
# ------------------------------------------------------------


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    # return the edged image
    return edged


# ------------------------------------------------------------
### Find Contours
# ------------------------------------------------------------

# defining a function to find contours from images
# with some default preprocessing applied (Grayscale, GaussianBlur, Adaptive Gaussian Thresholding)
# objects represented as black and background as white needs to be inverted first before contour detection


def find_contours(
    img_path=None,
    black_obj_on_white_bg=True,
    mode=cv.RETR_CCOMP,
    method=cv.CHAIN_APPROX_SIMPLE,
):
    try:
        img_read = cv.imread(img_path)
    except:
        print("Provide a valid image path along with the image extension.")

    img = cv.cvtColor(img_read, cv.COLOR_BGR2GRAY)
    img_blurred = cv.GaussianBlur(img, (5, 5), sigmaX=0)

    # Thresholding/ applying Canny Edge is a must to have step in finding contours.
    # Without this step we may not find contours at all
    img_thresholded = cv.adaptiveThreshold(
        img_blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 2
    )
    # img_canny = auto_canny(img_blurred)
    if black_obj_on_white_bg:
        img_final = cv.bitwise_not(img_thresholded)
    else:
        img_final = img_thresholded

    contours, hierarchy = cv.findContours(img_final, mode, method)
    return img_read, img_final, contours, hierarchy


# ------------------------------------------------------------
### Draw all the contours one by one
# ------------------------------------------------------------


# a function to draw all the detected contours one by one with 0.5 second pause after each one is drawn
def draw_all_contours_one_by_one(
    canvas_shape,
    contours,
    idx=-1,
    color=(200, 0, 0),
    thickness=1,
    time_to_pause_in_between=0.5,
):
    # to draw every contour on the same canvas
    # canvas = get_canvas(canvas_shape)
    for i, cnt in enumerate(contours):
        # to draw each one in a completely blank canvas
        canvas = get_canvas(canvas_shape)
        cv.drawContours(canvas, [cnt], idx, color, thickness)
        plt.imshow(canvas)
        plt.title(f"Contour = {i}")
        plt.pause(time_to_pause_in_between)
        plt.show()
