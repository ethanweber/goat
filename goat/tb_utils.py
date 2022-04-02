"""Tensorboard utils.
"""

import os
import glob
from collections import defaultdict
import scipy.misc
# try:
#     import tensorflow.compat.v1 as tf
#     tf.disable_v2_behavior()
# except:
#     pass
from tqdm import tqdm


def get_tensorboard_filename_from_folder(folder):
    # first get the event filename in the folder
    assert(os.path.isdir(folder))
    event_filenames = glob.glob(os.path.join(folder, "events.out.tfevents.*"))
    assert len(event_filenames) > 0
    event_filename = event_filenames[-1]
    return event_filename


def get_images_from_tensorboard_folder(folder, tag):
    """Return the images that `tag` using the most recent tensorboard
    file in the `folder`.
    """
    event_filename = get_tensorboard_filename_from_folder(folder)

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    images = []
    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tqdm(tf.train.summary_iterator(event_filename)):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    images.append(im)
    sess.close()
    return images

def get_values_from_tensorboard_folder(folder, tags):
    """Return the images that `tag` using the most recent tensorboard
    file in the `folder`.
    - https://stackoverflow.com/questions/37304461/tensorflow-importing-data-from-a-tensorboard-tfevent-file
    """
    event_filename = get_tensorboard_filename_from_folder(folder)

    values = defaultdict(list)
    for e in tf.train.summary_iterator(event_filename):
        for v in e.summary.value:
            if v.tag in tags:
                val = v.simple_value
                values[v.tag].append(val)
    values = dict(values)
    return values