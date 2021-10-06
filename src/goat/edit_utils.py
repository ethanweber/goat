"""Helper utility functions to edit images.
"""

import numpy as np
import torch
import cv2
from scipy.spatial.distance import euclidean as Euc
from scipy.spatial.transform import Rotation as R

from torchgeometry.core import (
    convert_points_from_homogeneous,
    convert_points_to_homogeneous
)


# TODO(ethan): clean this up properly, since it's currently copy pasted without much thought

def get_polygon_from_mask(mask, thresh=10, largest_only=True):
    """Get the polygon from a binary mask.
    Inputs:
        mask: (h, w) 0/1
        threshold: threshold is in pixel euclidean distance
    Return:
        polygon (np array of x,y points)
    """
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
    original_contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # RETR_EXTERNAL if only care about outer contours

    h, w = mask.shape
    lengths = np.zeros(64)  # assume no more than 64 polygons
    polygon = []
    i = 0

    largest_polygon = None
    largest_polygon_area = 0
    largest_polygon_length = None

    for original_contour in original_contours:

        original_polygon = np.squeeze(original_contour)
        temp_polygon = [original_polygon[0]]
        for point in original_polygon[1:]:
            dist = np.linalg.norm(point - temp_polygon[-1])
            if dist >= thresh:
                temp_polygon.append(point)
        temp_polygon = np.array(temp_polygon)
        if len(temp_polygon) < 3:  # can't have a polygon less than 3 points
            continue

        contour_area = cv2.contourArea(original_contour)
        if contour_area > largest_polygon_area:
            largest_polygon_area = contour_area
            largest_polygon = temp_polygon
            largest_polygon_length = len(largest_polygon)

        lengths[i] = len(temp_polygon)
        i += 1
        polygon.append(temp_polygon)
    if len(polygon) == 0:
        raise ValueError("Must have polygon array of nonzero length.")
    polygon = np.vstack(polygon)

    if largest_only:
        new_lengths = np.zeros(64)
        new_lengths[0] = largest_polygon_length
        return largest_polygon, new_lengths

    return polygon, lengths


def get_polygons_from_mask(mask, thresh=1):
    try:
        points, lengths = get_polygon_from_mask(
            mask, thresh=thresh, largest_only=False)

        points = points.tolist()
        lengths = lengths.tolist()

        polygons = []
        curr_length = 0
        for length in lengths:
            if int(length) == 0:
                break
            polygons.append(points[curr_length:curr_length + int(length)])
            curr_length += int(length)
    except:
        polygons = []
    return polygons


def get_padded_points(mask, padding=512, thresh=10, largest_only=True):
    """
        if gt, then don't perform any filtering
    """
    padded_points = torch.zeros(padding, 2)
    points, polygons = get_polygon_from_mask(
        mask, thresh=thresh, largest_only=largest_only)
    points = torch.from_numpy(points)
    points = points.float()
    points[:, 0] = points[:, 0] / mask.shape[1]  # range (0, 1)
    points[:, 1] = points[:, 1] / mask.shape[0]  # range (0, 1)
    num_points = points.shape[0]
    if num_points < padding:
        padded_points[:num_points] = points
    else:
        padded_points = points[:padding]
    return padded_points, torch.from_numpy(polygons.astype("int32"))


def get_expanded_bounding_box(bbox, img, perc_exp=0.1):
    """Returns the expanded bounding box.
    Args:
        perc_exp: percent expansion
    """
    h, w = img.shape[:2]
    bbox = [int(round(b)) for b in bbox]  # [x, y, width, height]
    bbox[0] = max(int(round(bbox[0] - bbox[2] * perc_exp)), 0)
    bbox[1] = max(int(round(bbox[1] - bbox[3] * perc_exp)), 0)
    bbox[2] = min(int(round(bbox[2] + bbox[2] * perc_exp * 2)), w - bbox[0])
    bbox[3] = min(int(round(bbox[3] + bbox[3] * perc_exp * 2)), h - bbox[1])
    return bbox


def get_image_with_mask_overlayed(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    Args:
        color (tuple): (r, g, b) with range (0, 1)
    """
    im = image.copy()
    for c in range(3):
        im[:, :, c] = np.where(
            mask == 1, im[:, :, c] * (1 - alpha) + alpha * color[c] * 255, im[:, :, c])
    return im


def get_image_with_bbox(image, bbox, color=(255, 0, 0)):
    """Apply bounding box to the image.
    Args:
        color (tuple): (r, g, b) with range (0, 255)
    """
    im = image.copy()
    x, y, width, height = bbox
    start_point = (x, y)
    end_point = (x + width, y + height)
    return cv2.rectangle(im, start_point, end_point, color, thickness=1)


def get_three_channel_image(image):
    """
    If single channel (2 dim), convert to a three dimensional image and return.
    """
    im = image.copy()
    if len(im.shape) == 2:
        im = np.stack((im, im, im), axis=2)
    return im


def draw_polygon_on_image(image,
                          polygon,
                          mask_color=None,
                          radius=4,
                          point_color=(255, 0, 0)):
    """Draws polygon points on image.
    """

    im = get_three_channel_image(image)

    if mask_color:
        im[:, :, 0][im[:, :, 0] == 255] = mask_color[0]
        im[:, :, 1][im[:, :, 1] == 255] = mask_color[1]
        im[:, :, 2][im[:, :, 2] == 255] = mask_color[2]

    for point in polygon:
        x, y = point  # TODO(ethan): make sure this is an integer
        try:
            im = cv2.circle(im, (x, y), radius, tuple(
                reversed(point_color)), -1)
        except:
            pass
    return im


def get_camera_points(depth,
                      intrinsics,
                      image=None):
    """Return points, colors after backprojecting into camera coordinates."""
    points = []
    colors = []
    index_to_x_y = []

    height, width = depth.shape[:2]
    # TODO: use meshgrid(?) or faster method
    for j in range(height):
        for i in range(width):
            d = depth[j, i]
            if d == 0:
                # TODO: don't project if the depth is 0
                # d = 10000
                continue
            points.append(
                [i * d, j * d, d]
            )
            color = image[j, i] if image is not None else [0, 0, 0]
            colors.append(color)
            index_to_x_y.append(
                (i, j)
            )

    points = np.array(points)
    colors = np.array(colors)
    points = np.transpose(np.linalg.inv(intrinsics) @ np.transpose(points))
    return points, colors, index_to_x_y

def get_camera_points_from_raydepth(depth, intrinsics, image=None):
    """Depth is defined along the ray, not Z.
    https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
    https://gist.githubusercontent.com/cbaus/6e04f7fe5355f67a90e99d7be0563e88/raw/883db511cf67e9898f3e8959ef2c55ad1848e8b0/convert.py
    """
    points = []
    colors = []
    index_to_x_y = []

    height, width = depth.shape[:2]
    depths = []
    # TODO: use meshgrid(?) or faster method
    for j in range(height):
        for i in range(width):
            d = depth[j, i]
            if d == 0:
                # TODO: don't project if the depth is 0
                # d = 10000
                continue
            depths.append(d)
            points.append(
                [i, j, 1]
            )
            color = image[j, i] if image is not None else [0, 0, 0]
            colors.append(color)
            index_to_x_y.append(
                (i, j)
            )

    points = np.array(points)
    colors = np.array(colors)
    depths = np.array(depths)
    points = np.transpose(np.linalg.inv(intrinsics) @ np.transpose(points))
    denominator = np.sum(points**2.0, axis=1) ** 0.5
    Z = depths / denominator
    XYZ = points * Z[:, None]
    return XYZ, colors, index_to_x_y

def points_to_world(points, pose):
    """points.shape == (X, 3)"""
    x = np.transpose(points)
    ones = np.ones((1, x.shape[1]))
    x = np.concatenate((x, ones), axis=0)  # homogeneous
    x = pose @ x
    x = x[:3, :] / x[3, :]
    return np.transpose(x)


def get_angle_magnitude(pose0, pose1):
    """Returns angle of quaternion rotation in degrees."""
    rel = pose0 @ np.linalg.inv(pose1)
    quat = R.from_matrix(rel[:3, :3]).as_quat()
    w = quat[3]
    angle = 2 * np.arccos(w)
    angle *= (180 / np.pi)
    return angle


def get_distance(pose0, pose1):
    """Returns the euclidean distance between two poses."""
    distance = Euc(pose0[:3, 3], pose1[:3, 3])
    return distance


def get_normalized(x):
    """Normalize numpy array to [0, 1].
    """
    x = (x - x.min())
    x /= x.max()
    return x


def pose_to_homogeneous(pose):
    """Doesn't work for batch shape.
    """
    assert pose.shape == (3, 4)
    row = torch.zeros_like(pose[0:1])
    row[0, 3] = 1.0
    x = torch.cat([pose, row], dim=0)
    return x
