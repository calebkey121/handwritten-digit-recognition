Caleb Key's Digit Recognition Machine Learning Algorithm

Welcome to Caleb Key's digit recognition machine learning algorithm! This is a simple machine learning algorithm designed to recognize digits from images.
Overview

The algorithm is based on the classic MNIST dataset, which consists of 70,000 images of handwritten digits (0-9). The algorithm is trained on a subset of the dataset (60,000 images) and tested on the remaining 10,000 images.

The algorithm uses a supervised learning approach, specifically a neural network with a single hidden layer. The input to the network is a flattened version of the 28x28 pixel images (784 pixels in total). The output of the network is a 10-dimensional vector, where each element corresponds to a digit (0-9).

The algorithm uses the cross-entropy loss function and the Adam optimizer to train the network. The hyperparameters are chosen using a grid search approach.
Requirements

To run the algorithm, you will need at least python3.10:
    Python 3.x

Installation:
Install python (if not already)
Check if already installed
    python --version
    # or
    python3 --version
Create a virtual environment
    python3 -m venv hdr
Activate the virtual environment
    source hdr/bin/activate # (on macOS/Linux)
    .\hdr\Scripts\activate # (on Windows)
Install Packages
    pip install -r requirements.txt
(When you've finished) Deactive Virtual Environment
    deactivate


Usage:
To use the algorithm, simply run the main.py script:
python main.py

This will train the neural network on the MNIST dataset and test it on the test set. The results will be displayed in the console, along with a confusion matrix and some example images with their predicted labels.

You can also modify the hyperparameters in the config.py file to experiment with different settings.
Conclusion

This algorithm provides a simple and effective way to recognize digits from images using machine learning. It is designed for beginners who want to learn about machine learning and neural networks.

If you have any questions or feedback, feel free to contact Caleb Key at calebkey121@gmail.com. Happy learning!
