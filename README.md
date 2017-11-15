# Car Component Classification Model

This repo shows you how to train a deep learning model in TensorFlow and serve the model in production using TensorFlow Serving.


## Requirements
- TensorFlow 1.3
- GPU support (CUDA 8.0 and cuDNN 6.0)
- Numpy


## Model Training 

If you just want to know how to serve the model, go to Model Serving section. The pre-trained model is provided.

### Data Preparation

First convert the original images to TFRecord format.
Run the following:
```
cd data_preparation
python convert_images_to_tfrecords.py
```

Once your dataset is ready, you can begin training the model as follows:
```
cd model_training
python main.py
```

Important flags:
- data_dir: The directory where the input data is stored.
- model_dir: The directory where the model will be stored.
There are more flag options as described in main.py.


## Model Serving

To get started with TensorFlow Serving:
- Read the [overview](https://www.tensorflow.org/serving/architecture_overview)
- [Set up](https://www.tensorflow.org/serving/setup) your environment.
**NOTE**: Enable CUDA support when configuring TensorFlow. To build the entire tree or individual targets, add cuda configuration. Example code:
```
bazel build -c opt --config=cuda tensorflow_serving/...
```
- Do the [basic tutorial](https://www.tensorflow.org/serving/serving_basic)

### Export TensorFlow Model

Firstly clone TensorFlow Serving repo (you already done in [here](https://www.tensorflow.org/serving/setup))

Secondly clone model_serving folder from this repo into TensorFlow Serving directory (path /serving/tensorflow_serving/)

Lastly load the trained checkpoint add the signiture, and export the model for serving
```
bazel-bin/tensorflow_serving/model_serving/car_saved_model
```

Important flags:
- model_version: Model version to be saved
- checkpoint_dir: Directory where to read training checkpoints.
- output_dir: Directory where to export inference model.

Let's take a look at the export directory.
```
ls /../serving/tensorflow_serving/model_serving/saved_model
```
As you can see, a sub-directory will be created for exporting each verion of the model. FLAGS.model_version has the default value of 1, therefore the corresponding sub-directory 1 is created.


### Start the server

Run the [gRPC](https://grpc.io/) tensorflow_model_server with Standard TensorFlow ModelServer.
```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=car_component_classification --model_base_path=/../serving/tensorflow_serving/model_serving/saved_model
```

### Query the server

Query the server with car_client.py. The client sends images to the server over gRPC for classification and save the predictinos into a csv file finally.
```
bazel-bin/tensorflow_serving/model_serving/car_client --server=localhost:9000 --image_path=/../serving/tensorflow_serving/model_serving/testing_data --output_csv=/../output.csv
```


Congratulations! You have successfully deployed the car component classifcation model into production.