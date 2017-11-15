from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

import dataset_utils

_NUM_SHARDS = 5

class ImageReader(object):
    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
#
    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]
#
    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                        feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'car_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, filenames, dataset_dir):
    assert split_name in ['train', 'validation', 'test']
    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i+1, len(filenames), shard_id))
                        sys.stdout.flush()
                        # Read the filename:
                        try:
                            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                            height, width = image_reader.read_image_dims(sess, image_data)
                            class_id_1 = filenames[i].split("/")[-1]
                            class_id_2 = class_id_1.split("_")[0]
                            class_id = int(class_id_2)
                            example = dataset_utils.image_to_tfexample(
                                image_data, b'jpg', height, width, class_id)
                            tfrecord_writer.write(example.SerializeToString())
                        except:
                            print("BAD image found!")
                            continue
    sys.stdout.write('\n')
    sys.stdout.flush()


import glob
# grab all the training images
all_files = glob.glob("/../training_images/*.jpg")
training_filenames = all_files
_convert_dataset('train', training_filenames, "/../data_preparation/example_data/")


category_list = ["Front 3/4 View Drivers", "Front 3/4 View Passenger", "Side View Passenger", 
                 "Rear 3/4 View Passenger", "Side View Drivers", "Rear View", "Rear 3/4 View Drivers",
                 "Front", "Roof/Sunroof", "Drivers Dashboard/Centre Console", "Center Console",
                 "Trunk Compartment", "Door Controls", "Drivers Front Seat", "Drivers Side Interior",
                 "Passenger Front Seat", "Rear Seat", "Navigation System with CD", "Instrument Panel",
                 "Mileage - Odometer", "Keys and Manuals", "Engine Compartment", "Other"]

labels_to_class_names = dict(zip(range(len(category_list)), category_list))
dataset_utils.write_label_file(labels_to_class_names, "/../data_preparation/example_data/")


