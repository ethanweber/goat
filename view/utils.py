"""Helper code to visualize images quickly and efficiently in the web.

Examples:
    https://gist.github.com/ethanweber/79dd0a0a1341c243e26bae3772cd4505
"""

from IPython.core.display import display, HTML
import cv2
import base64

from goat.edit import utils as edit_utils

def get_html_from_image_urls(image_urls):
    html_str = """"""
    for image_url in image_urls:
        html_str += """<img src="{}" height="100px" width="100px">""".format(image_url)
    return html_str


def get_html_from_image(image,
                        height=None,
                        width=None,
                        label=None):
    """

    :param image: np.array
    :param height: int
    :param width: int
    :return:
    """
    if height is not None and width is not None:
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    elif height is not None:
        h, w = image.shape[:2]
        width = int(height * (w / h))
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    elif width is not None:
        h, w = image.shape[:2]
        height = int(width * (h / w))
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        resized_image = image
    if len(resized_image.shape) == 2:
        resized_image = edit_utils.get_three_channel_image(resized_image)
    base64_string = base64.b64encode(cv2.imencode('.jpg', resized_image[:, :, ::-1])[1]).decode()
    html_img_str = """<img src="data:image/png;base64, {}" />""".format(base64_string)
    html_label_str = "" if label is None else """<div style="text-align: center">{}</div>""".format(label)
    html_div = """
        <div style="display: inline-block">
            <div style="text-align: center">{}</div>
            {}
        </div>
        """.format(html_img_str, html_label_str)
    return html_div


def imshow(image,
           height=None,
           width=None,
           label=None):
    html_div = get_html_from_image(image, height=height, width=width, label=label)
    display(HTML(html_div))


def show_images(images,
                labels=None,
                height=None,
                width=None):
    html_div = ""
    if labels is None:
        labels = [None] * len(images)
    for image, label in zip(images, labels):
        html_div += get_html_from_image(image, height=height, width=width, label=label)
    display(HTML(html_div))
