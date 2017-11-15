# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

py_library(
    name = "vgg_preprocessing",
    srcs_version = "PY2AND3",
    srcs = ["vgg_preprocessing.py"],
)

py_library(
    name = "resnet_model",
    srcs_version = "PY2AND3",
    srcs = ["resnet_model.py"],
)

py_binary(
    name = "car_saved_model",
    srcs = [
        "car_saved_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "car_client",
    srcs = [
        "car_client.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
