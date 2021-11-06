"""Helper code to visualize images quickly and efficiently in the web.

Examples:
    https://gist.github.com/ethanweber/79dd0a0a1341c243e26bae3772cd4505
"""

from IPython.display import IFrame
from IPython.core.display import display, HTML
import cv2
import base64
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import urllib
import imageio

from . import edit_utils


def save_gif_from_images(gif_filename, images_in, backwards=True, fps=5):
    """
    filename - gif filename
    """
    images = images_in
    if backwards:
        images = images_in + images_in[::-1]
    imageio.mimsave(gif_filename, images, fps=fps)


def save_gif_from_image_filenames(gif_filename, image_filenames):
    """
    filename - gif filename
    """
    images = []
    for image_filename in tqdm(image_filenames):
        images.append(cv2.imread(image_filename)[:, :, ::-1])
    save_gif_from_images(gif_filename, images)


def show_gif_from_filenames(gif_filenames, labels=None, width=None):
    html_str = ""
    width_str = "" if width is None else """ width="{}" """.format(width)
    for idx, gif_filename in enumerate(gif_filenames):
        if labels:
            html_str += """{}<br>""".format(labels[idx])
        html_str += """<img src="{}" "{}" /><br>""".format(gif_filename, width_str)
    display(HTML(html_str))
    return html_str


def get_tile_from_image(image, size, tile):
    """Using 0 indexing.
    size - tuple (num cols, num rols)
    tile - tuple (desired col, desired row)
    """
    outer_h, outer_w, _ = image.shape
    inner_h, inner_w = outer_h // size[0], outer_w // size[1]
    c, r = tile
    return image[c * inner_h:c * inner_h + inner_h, r * inner_w:r * inner_w + inner_w, :]


def get_image_grid(
    images,
    rows=None,
    cols=None
):
    """Returns a grid of images.
    Assumes images are same height and same width.
    """
    def get_image(images, idx):
        # returns white if out of bounds
        if idx < len(images):
            return images[idx]
        else:
            return np.ones_like(images[0]) * 255

    im_rows = []
    idx = 0
    for i in range(rows):
        im_row = []
        for j in range(cols):
            im_row.append(get_image(images, idx))
            idx += 1
        im_rows.append(np.hstack(im_row))
    im = np.vstack(im_rows)
    return im


def get_html_from_image_urls(image_urls):
    html_str = """"""
    for image_url in image_urls:
        html_str += """<img src="{}" height="100px" width="100px">""".format(
            image_url)
    return html_str


def get_html_from_image(image,
                        height=None,
                        width=None,
                        label=None,
                        fontsize=None):
    """

    :param image: np.array
    :param height: int
    :param width: int
    :return:
    """
    if height is not None and width is not None:
        resized_image = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_NEAREST)
    elif height is not None:
        h, w = image.shape[:2]
        width = int(height * (w / h))
        resized_image = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_NEAREST)
    elif width is not None:
        h, w = image.shape[:2]
        height = int(width * (h / w))
        resized_image = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        resized_image = image
    if len(resized_image.shape) == 2:
        resized_image = edit_utils.get_three_channel_image(resized_image)
    base64_string = base64.b64encode(cv2.imencode(
        '.jpg', resized_image[:, :, ::-1])[1]).decode()
    html_img_str = """<img src="data:image/png;base64, {}" />""".format(
        base64_string)

    html_fontsize_str = "" if fontsize is None else "; font-size: {}px".format(fontsize)
    html_label_str = "" if label is None else """<div style="text-align: center {}">{}</div>""".format(
        html_fontsize_str, label)
    
    html_div = """
        <div style="display: inline-block; border-style: solid; margin: 1px">
            {} 
            <div style="text-align: center">{}</div>
        </div>
        """.format(html_label_str, html_img_str)
    return html_div


def imshow(image,
           height=None,
           width=None,
           label=None):
    html_div = get_html_from_image(
        image, height=height, width=width, label=label)
    display(HTML(html_div))


def show_images(images,
                labels=None,
                height=None,
                width=None,
                return_html=False,
                fontsize=None):
    html_div = ""
    if labels is None:
        labels = [None] * len(images)
    for image, label in zip(images, labels):
        html_div += get_html_from_image(image,
                                        height=height, width=width, label=label, fontsize=fontsize)
    if return_html:
        return html_div
    else:
        display(HTML(html_div))


def get_animation_from_images(images, interval=1000):
    fig = plt.figure()
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)

    pltims = []
    for i in range(len(images)):
        im = images[i].copy()
        val = plt.imshow(im)
        plt.axis('off')
        pltims.append([val])

    ani = animation.ArtistAnimation(fig,
                                    pltims,
                                    interval=interval,
                                    blit=True,
                                    repeat_delay=1000)
    plt.close()
    return ani

    # save the video
    # ani.save('video_of_images.mp4')

    # show the animation
    # display(HTML(ani.to_html5_video()))

# METHOD #1: OpenCV, NumPy, and urllib
# https://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, ::-1].copy()
    return image

# def show_pcd_in_notebook(pcd, width=800, height=400):
#     # save the point cloud
#     o3d.io.write_point_cloud(
#         "/data/vision/torralba/humanitarian/ethan/recon/vis/static/pcd_files/temp.pcd", pcd)
#     return IFrame(src='https://recon.ethanweber.me/viewer/temp', width=width, height=height)


# def show_depth_image_in_notebook(depth_image,
#                                  scalar=1.0,
#                                  color_image=None,
#                                  display_width=800,
#                                  display_height=400):
#     """If color_image is not None, then display those colors for the points.
#     """
#     max_depth = depth_image.max()
#     height, width = depth_image.shape
#     pcd_text = """"""
#     pcd_text += "# .PCD v0.7 - Point Cloud Data file format\n"
#     pcd_text += "VERSION 0.7\n"
#     pcd_text += "FIELDS x y z rgb\n"
#     pcd_text += "SIZE 4 4 4 4\n"
#     pcd_text += "TYPE F F F F\n"
#     pcd_text += "COUNT 1 1 1 1\n"
#     pcd_text += "WIDTH 1000\n"
#     pcd_text += "HEIGHT 1\n"
#     pcd_text += "VIEWPOINT 0 0 0 1 0 0 0\n"
#     pcd_text += "POINTS {}\n".format(height * width)
#     pcd_text += "DATA ascii\n"
#     for y in range(height):
#         for x in range(width):
#             z = max_depth - depth_image[y, x]
#             rgb = 0.0
#             if color_image is not None:
#                 r, g, b = color_image[y, x, :]
#                 rgb = str((int(r) << 16) + (int(g) << 8) + (int(b)))
#             line = "{} {} {} {} \n".format(
#                 x * scalar, y * scalar, z * scalar, rgb)
#             pcd_text += line
#     temp = open(
#         "/data/vision/torralba/humanitarian/ethan/recon/vis/static/pcd_files/temp.pcd", "w")
#     temp.write(pcd_text)
#     temp.close()
#     return IFrame(src='https://recon.ethanweber.me/viewer/temp',
#                   width=display_width,
#                   height=display_height)
