
# /Users/sauravbakshi/Documents/projects/mmwml_course_repo/mmwml_course_repo/Week1
# organize imports
import numpy as np
import os
import tensorflow as tf
from keras.models import Model
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
import tensorflowjs as tfjs

MODEL_PATH = 'models/'
TEST_IMG_PATH = "/Users/sauravbakshi/Documents/projects/mmwml_course_repo/mmwml_course_repo/Week1/images/adorable-animal-breed-356378.jpg"

# process an image to be mobilenet friendly


def load_model():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.disable_eager_execution()
    model = ResNet50(weights='imagenet')
    return model


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    return x


def model_predict(input_image, model):

    preds = model.predict(input_image)
    return preds


# main function
if __name__ == '__main__':

    # load the model
    resnet = load_model()
    print("[Step 1] -->  ResNet50 model loaded....")

    # process the test image
    input_image = preprocess_image(TEST_IMG_PATH)
    print("[Step 2] --> Image loaded....", TEST_IMG_PATH)

    # generate predictions
    preds = model_predict(input_image, resnet)
    print("[Step 3] --> prediction completed")
    # obtain the top-5 predictions
    results = imagenet_utils.decode_predictions(preds)
    print("[Step 4] ---> ResNet50 model predict result...", results)
    # convert the mobilenet model into tf.js model
    tfjs.converters.save_keras_model(resnet, MODEL_PATH)
    print("[Step 5] --> [INFO] saved tf.js resnet model to disk..")
