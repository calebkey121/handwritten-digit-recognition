"""
Reading from Learning Deep Learning by Magnus Ekman

As I follow along, I might decide to use some of his implementations
"""
from neural_network import NeuralNetwork
from grid_editor.grid_editor import GridEditor
import tensorflow
import keras
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
from pytorch_model_defs import my_first_pytorch

def list_files(directory, ext):
    # List to store filenames
    files = []
    
    # Walk through the directory
    for filename in os.listdir(directory):
        # Check if the file ends with ext
        if filename.endswith(ext):
            files.append(filename)
    
    return files

TRAIN_IMAGE_FILENAME = "./data/mnist/train-images.idx3-ubyte"
TRAIN_LABEL_FILENAME = "./data/mnist/train-labels.idx1-ubyte"
TEST_IMAGE_FILENAME = "./data/mnist/t10k-images.idx3-ubyte"
TEST_LABEL_FILENAME = "./data/mnist/t10k-labels.idx1-ubyte"

EPOCHS = 3

mean = None
stddev = None
preload = True
my_model_filename = "models/my_model.pkl"
my_model = None
keras_model_filenames = list_files(directory="models/", ext=".keras")
keras_models = [ keras.saving.load_model(f"models/{filename}") for filename in keras_model_filenames ]
pytorch_model_filenames = list_files(directory="models/", ext=".pth")
pytorch_models = [ torch.load(f"models/{filename}") for filename in pytorch_model_filenames ]

import numpy as np

def get_predictions(layer):
    if torch.is_tensor(layer):
        layer = layer.cpu().detach().numpy()[0]
    percentages = softmax(layer)
    prediction = layer.argmax()
    output = {}
    for digit, percent in enumerate(percentages):
        output[digit] = percent
    output_descending = {k: f"{round(v * 100, 2)}%" for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}
    return output_descending

def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)


def read_mnist_data():
    f"""
    Reads and formats MNIST data set for our use
    Returns training and testing input and labels
    """
    # need to adjust manual input the same
    global mean
    global stddev

    train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
    train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
    test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
    test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)

    # show_images(test_images)

    #start_grid_editor(train_images[0])
    # Reformat and standardize
    # we know there are 60,000 28x28 images, for our purpose, reshape the 3d array into 2d
    x_train = train_images.reshape(60000, 784) # instead of 28x28 vector we get 1x784
    # Standardize x_train and x_test
    mean = np.mean(x_train)
    stddev = np.std(x_train)
    x_train = (x_train - mean) / stddev
    x_test = test_images.reshape(10000, 784)
    x_test = (x_test - mean) / stddev

    # One-hot encoded output
    y_train = np.zeros((60000, 10))
    y_test = np.zeros((10000, 10))
    # set the correct digit to one (high)
    for i, y in enumerate(train_labels):
        y_train[i][y] = 1
    for i, y in enumerate(test_labels):
        y_test[i][y] = 1
    
    return x_train, y_train, x_test, y_test


def show_images(images):
    for image in images:
        draw_number(image)


def prediction(pred_input):
    global mean, stddev, keras_models, my_model, keras_model_filenames

    manual_input = np.array(pred_input)
    x = np.array(manual_input)
    x = x.reshape(1, 784)
    x = (x - mean) / stddev
    my_model.forward_pass(x[0])
    my_model_pred = get_predictions(my_model.output_layer)

    manual_input = np.expand_dims(manual_input, axis=0)
    keras_predictions = {}
    for i, model in enumerate(keras_models):
        input_tensor = tensorflow.convert_to_tensor(manual_input)
        output = model.predict(input_tensor)[0]
        keras_model_pred = get_predictions(output)
        keras_predictions[keras_model_filenames[i]] = keras_model_pred
    
    pytorch_predictions = {}
    manual_input = torch.tensor(np.array(manual_input), dtype=torch.float32)
    for i, model in enumerate(pytorch_models):
        model.eval()
        with torch.no_grad():
            output = model(manual_input)
        pytorch_model_pred = get_predictions(output)
        pytorch_predictions[pytorch_model_filenames[i]] = pytorch_model_pred

    models = {
        "My Model": my_model_pred
    }
    for name, prediction in keras_predictions.items():
        models[name[:-6]] = prediction
    for name, prediction in pytorch_predictions.items():
        models[name[:-4]] = prediction
    return models


def main():
    global my_model

    layout = [25, 10]

    x_train, y_train, x_test, y_test = read_mnist_data()
    index_list = list(range(len(x_train))) # used for random order
    if not preload:
        # create and train the model
        my_model = NeuralNetwork(layout, x_train, y_train, x_test, y_test, learning_rate=0.01)
        my_model.training_loop(EPOCHS)
        # save the model
        with open(my_model_filename, "wb") as f:
            pickle.dump(my_model, f)
    else:
        # unload mnist_model.pkl
        with open(my_model_filename, "rb") as f:
            my_model = pickle.load(f)
    models = { "My Model": {} }
    for name in keras_model_filenames:
        models[name[:-6]] = {}
    for name in pytorch_model_filenames:
        models[name[:-4]] = {}
    GridEditor(models=models, prediction_callback=prediction)

if __name__ == "__main__":
    main()
