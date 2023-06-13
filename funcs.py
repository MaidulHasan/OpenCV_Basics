import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

### matplotlib_imshow


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


### Auto Canny Edge detector


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    # return the edged image
    return edged
