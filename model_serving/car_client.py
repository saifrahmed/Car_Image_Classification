""" Client example code from tensorflow serving repo
Link:
    - https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_client.py
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import glob
import os
import sys
import threading
import time
import csv
import numpy as np 

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image_path', '/../model_serving/testing_data', 
                           'path to images')
tf.app.flags.DEFINE_string('output_csv', '/../output.csv', 
                           'csv file with predictions')
tf.app.flags.DEFINE_integer('concurrency', 5,
                            'maximum number of concurrent inference requests')
FLAGS = tf.app.flags.FLAGS

class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._prediction = []
    self._name = []
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def inc_prediction(self, prediction, name):
    with self._condition:
      self._prediction.append(prediction)
      self._name.append(name)

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_prediction_list(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._prediction

  def get_name_list(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._name

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1

def _create_rpc_callback(name, result_counter):
  """Creates RPC callback function.

  Args:
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      print(exception)
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      pred_result = result_future.result().outputs['classes'].string_val
      result_counter.inc_prediction(prediction=pred_result, name=name)
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback

def do_inference(hostport, image_dir, concurrency):
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  image_list = glob.glob(os.path.join(image_dir, "*.jpg"))
  result_counter = _ResultCounter(len(image_list), concurrency)
  for i in image_list:
    name = i.split("/")[-1]
    data = open(i,"rb").read()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'car_component_classification'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data, shape=[1]))
    result_counter.throttle()
    result_future = stub.Predict.future(request, 10.0)
    result_future.add_done_callback(
      _create_rpc_callback(name, result_counter))
  return result_counter.get_prediction_list(), result_counter.get_name_list()

def main(_):
  if not FLAGS.server:
    print('please specify server host:port')
    return
  t1 = time.time()
  prediction_results, filename_list = do_inference(FLAGS.server, FLAGS.image_path,
                            FLAGS.concurrency)
  #print('\nImages are {}'.format(filename_list))
  #print('\nPredictions are {}'.format(prediction_results))
  t_diff = time.time() - t1
  print("Total prediction time is {} seconds".format(t_diff))
  predictions_human_readable = np.column_stack((np.array(filename_list), prediction_results))
  print("Saving prediction to ...")
  with open(FLAGS.output_csv, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
  print("Done")


if __name__ == '__main__':
  tf.app.run()
