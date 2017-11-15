"""Export model code from tensorflow serving repo.
   The model is exported as SavedModel with proper signatures that can be loaded by
   standard tensorflow_model_server.
Link:
    - https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_saved_model.py
"""


import os.path

# This is a placeholder for a Google-internal import.

import tensorflow as tf
import resnet_model
import vgg_preprocessing


tf.app.flags.DEFINE_string('checkpoint_dir', '/../model_serving/model_checkpoints/',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', '/../model_serving/saved_model/',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
tf.app.flags.DEFINE_integer('image_size', 32,
                            """Image size.""")
FLAGS = tf.app.flags.FLAGS

NUM_CLASSES = 23
NUM_TOP_CLASSES = 1
resnet_size = 32

category_list = ["Front 3/4 View Drivers", "Front 3/4 View Passenger", "Side View Passenger", 
                 "Rear 3/4 View Passenger", "Side View Drivers", "Rear View", "Rear 3/4 View Drivers",
                 "Front", "Roof/Sunroof", "Drivers Dashboard/Centre Console", "Center Console",
                 "Trunk Compartment", "Door Controls", "Drivers Front Seat", "Drivers Side Interior",
                 "Passenger Front Seat", "Rear Seat", "Navigation System with CD", "Instrument Panel",
                 "Mileage - Odometer", "Keys and Manuals", "Engine Compartment", "Other"]

def export():
  with tf.Graph().as_default():
    # Build inference model.
    # Input transformation.
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {
        'image/encoded': tf.FixedLenFeature(shape=[],
            dtype=tf.string),
    }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    jpegs = tf_example['image/encoded']
    images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

    # Run inference.
    network = resnet_model.cifar10_resnet_v2_generator(
      resnet_size, NUM_CLASSES, 'channels_first')
    logits = network(
      inputs=images, is_training=False)

    # Transform output to topK result.
    values, indices = tf.nn.top_k(tf.nn.softmax(logits), NUM_TOP_CLASSES)
    #pred = tf.argmax(logits, axis=1)

    # Create a constant string Tensor where the i'th element is
    # the human readable class description for the i'th index.
    class_descriptions = category_list
    class_tensor = tf.constant(class_descriptions)

    table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
    classes = table.lookup(tf.to_int64(indices))

    # Restore variables from training checkpoint.
    saver = tf.train.Saver()
    with tf.Session() as sess:
      # Restore variables from training checkpoints.
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        #saver.restore(sess, ckpt.model_checkpoint_path)
        saver.restore(sess, FLAGS.checkpoint_dir + "/" + ckpt.model_checkpoint_path.split('/')[-1])
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print 'Successfully loaded model from %s at step=%s.' % (
            ckpt.model_checkpoint_path, global_step)
      else:
        print 'No checkpoint file found at %s' % FLAGS.checkpoint_dir
        return

      # Export inference model.
      output_path = os.path.join(
          tf.compat.as_bytes(FLAGS.output_dir),
          tf.compat.as_bytes(str(FLAGS.model_version)))
      print 'Exporting trained model to', output_path
      builder = tf.saved_model.builder.SavedModelBuilder(output_path)

      # Build the signature_def_map.
      classify_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(
          serialized_tf_example)
      classes_output_tensor_info = tf.saved_model.utils.build_tensor_info(
          classes)
          #pred)
      scores_output_tensor_info = tf.saved_model.utils.build_tensor_info(values)

      classification_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={
                  tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                      classify_inputs_tensor_info
              },
              outputs={
                  tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                      classes_output_tensor_info,
                  tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                      scores_output_tensor_info
              },
              method_name=tf.saved_model.signature_constants.
              CLASSIFY_METHOD_NAME))

      predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(jpegs)
      prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'images': predict_inputs_tensor_info},
              outputs={
                  'classes': classes_output_tensor_info,
                  'scores': scores_output_tensor_info
              },
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
          ))

      legacy_init_op = tf.group(
          tf.tables_initializer(), name='legacy_init_op')
      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              'predict_images':
                  prediction_signature,
              tf.saved_model.signature_constants.
              DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  classification_signature,
          },
          legacy_init_op=legacy_init_op)

      builder.save()
      print 'Successfully exported model to %s' % FLAGS.output_dir


def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""
  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  image = tf.image.decode_jpeg(image_buffer, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = vgg_preprocessing.preprocess_image(
    image=image,
    output_height=FLAGS.image_size,
    output_width=FLAGS.image_size,
    is_training=False)
  return image

def main(unused_argv=None):
  export()


if __name__ == '__main__':
  tf.app.run()
