import csv
import os
import random
import sys
import time
import warnings  # Control warning messages that pop up

"""
https://github.com/mohamedameen93/German-Traffic-Sign-Classification-Using-TensorFlow
"""

warnings.filterwarnings("ignore")  # Ignore all warnings

import matplotlib.pyplot as plt  # Plotting library
import numpy as np  # Scientific computing library
import pickle  # Converts an object into a character stream (i.e. serialization)
from sklearn.utils import shuffle  # Machine learning library
import tensorflow as tf  # Machine learning library
from tensorflow import keras  # Deep learning library
from tensorflow.keras.models import load_model  # Loads a trained neural network
import skimage.morphology as morp
from skimage.filters import rank

import cv2


def example():
    try:
        msg = tf.constant('Hello, TensorFlow!')
        tf.print(msg)
        print("TensorFlow version:", tf.__version__)

        print(tf.test.gpu_device_name())

        # Show current TensorFlow version
        tf.__version__

        # Load a dataset
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # Build a machine learning model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])

        # For each example, the model returns a vector of logits or log-odds scores, one for each class.
        predictions = model(x_train[:1]).numpy()
        predictions
        tf.nn.softmax(predictions).numpy()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_fn(y_train[:1], predictions).numpy()

        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)

        model.evaluate(x_test, y_test, verbose=2)

        probability_model = tf.keras.Sequential([
            model,
            tf.keras.layers.Softmax()
        ])

        probability_model(x_test[:5])

    except Exception as e:
        return


signs = []


def list_images(dataset, dataset_y, ylabel="", cmap=None):
    """
    Display a list of images in a single figure with matplotlib.
        Parameters:
            images: An np.array compatible with plt.imshow.
            lanel (Default = No label): A string to be used as a label for each image.
            cmap (Default = None): Used to display gray images.
    """
    plt.figure(figsize=(15, 16))
    for i in range(6):
        plt.subplot(1, 6, i + 1)
        indx = random.randint(0, len(dataset))
        # Use gray scale color map if there is only one channel
        cmap = 'gray' if len(dataset[indx].shape) == 2 else cmap
        plt.imshow(dataset[indx], cmap=cmap)
        plt.xlabel(signs[dataset_y[indx]])
        plt.ylabel(ylabel)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.gcf().canvas.set_window_title(ylabel)
    plt.show()


def histogram_plot(dataset, label, n_classes):
    """
    Plots a histogram of the input data.
        Parameters:
            dataset: Input data to be plotted as a histogram.
            label: A string to be used as a label for the histogram.
    """
    hist, bins = np.histogram(dataset, bins=n_classes)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel(label)
    plt.ylabel("Image count")
    plt.gcf().canvas.set_window_title(label)
    plt.show()


def build_cnn_model():
    try:
        """        
        Step1 = we preprocess individual images in each folder and apply a label, so that a vector/binary
        representation of features is created along with the class. This array of vectors is nothing but the input to the model
        for training/test purposes. We may apply basic pre-processing on images liking graying/scaling down
        to reduce the computational requirements.
        This can be done on GPU and very process intensive, one time activity.
        Step2 = The binary can be split in the train/test/validate files and is used to build the model
        Loads pickled training and test data.
        """

        """
        The first thing we need to do is to load the image data from the pickle files.
        Open the training, validation, and test data sets
        """
        training_file = "./road-sign-data/train.p"
        validation_file = "./road-sign-data/valid.p"
        testing_file = "./road-sign-data/test.p"

        with open(training_file, mode='rb') as training_data:
            train = pickle.load(training_data)
        with open(validation_file, mode='rb') as validation_data:
            valid = pickle.load(validation_data)
        with open(testing_file, mode='rb') as testing_data:
            test = pickle.load(testing_data)

        # Mapping ClassID to traffic sign names
        label_file = "./signnames.csv"
        with open(label_file, 'r') as csvfile:
            signnames = csv.reader(csvfile, delimiter=',')
            next(signnames, None)
            for row in signnames:
                signs.append(row[1])
            csvfile.close()

        """
        We then split the data set into a training set, testing set and validation set.
        Store the features and the labels
        Here train/test/valid split = 65%, 25%, 10%
        """
        X_train, y_train = train['features'], train['labels']
        X_valid, y_valid = valid['features'], valid['labels']
        X_test, y_test = test['features'], test['labels']

        # Output the dimensions of the training data set
        # Feel free to uncomment these lines below
        # Number of training examples
        n_train = X_train.shape[0]

        # Number of testing examples
        n_test = X_test.shape[0]

        # Number of validation examples.
        n_validation = X_valid.shape[0]

        # What's the shape of an traffic sign image?
        image_shape = X_train[0].shape

        # How many unique classes/labels there are in the dataset.
        n_classes = len(np.unique(y_train))

        print("Number of training examples: ", n_train)
        print("Number of testing examples: ", n_test)
        print("Number of validation examples: ", n_validation)
        print("Image data shape =", image_shape)
        print("Number of classes =", n_classes)
        print(X_train.shape)
        print(y_train.shape)

        """
        Display an image from all the data sets
        """
        list_images(X_train, y_train, "Training example")
        list_images(X_test, y_test, "Testing example")
        list_images(X_valid, y_valid, "Validation example")

        # display any random image at i location
        i = 500
        # plt.imshow(X_train[i])
        # plt.show() # Uncomment this line to display the image
        # print(y_train[i])

        # Plotting histograms of the count of each sign
        histogram_plot(y_train, "Training examples", n_classes)
        histogram_plot(y_test, "Testing examples", n_classes)
        histogram_plot(y_valid, "Validation examples", n_classes)

        """
        Shuffle the data set to make sure that we don’t have unwanted biases and patterns.
        Shuffle the image data set
        Shuffling: In general, we shuffle the training data to increase randomness and variety in training dataset, 
        in order for the model to be more stable. We will use sklearn to shuffle our data
        """
        X_train, y_train = shuffle(X_train, y_train)

        """
        Convert Data Sets from RGB Color Format to Grayscale
        Our images are in RGB format. We convert the images to grayscale so that the neural network can process them more easily.
        Convert the RGB image data set into grayscale
        
        using grayscale images instead of color improves the ConvNet's accuracy. 
        We will use OpenCV to convert the training images into grey scale.
        """
        X_train_grscale = np.sum(X_train / 3, axis=3, keepdims=True)
        X_test_grscale = np.sum(X_test / 3, axis=3, keepdims=True)
        X_valid_grscale = np.sum(X_valid / 3, axis=3, keepdims=True)

        """
        # Normalize the data set
        # Note that grayscale has a range from 0 to 255 with 0 being black and
        # 255 being white
        
        We normalize the images to speed up training and improve the neural network’s performance.
        """
        X_train_grscale_norm = (X_train_grscale - 128) / 128
        X_test_grscale_norm = (X_test_grscale - 128) / 128
        X_valid_grscale_norm = (X_valid_grscale - 128) / 128

        # Display the shape of the grayscale training data
        # print(X_train_grscale.shape)

        # Display a sample image from the grayscale data set
        i = 500
        # squeeze function removes axes of length 1
        # (e.g. arrays like [[[1,2,3]]] become [1,2,3])
        # plt.imshow(X_train_grscale[i].squeeze(), cmap='gray')
        # plt.figure()
        # plt.imshow(X_train[i])
        # plt.show()

        # Get the shape of the image
        # IMG_SIZE, IMG_SIZE, IMG_CHANNELS
        img_shape = X_train_grscale[i].shape
        # print(img_shape)

        """
        # Build the convolutional neural network's (i.e. CNN model) architecture
        filters: define the number of filters
        kernel_size: 
        strides:
        input_shape: 
        activation: this is the function to activate neurons
        pool_size: scale down all the feature maps to generalize more and to reduce over-fitting of the curve
        """
        cnn_model = tf.keras.Sequential()  # Plain stack of layers
        cnn_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                             strides=(3, 3), input_shape=img_shape, activation='relu'))
        cnn_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                             activation='relu'))
        cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        cnn_model.add(tf.keras.layers.Dropout(0.25))
        cnn_model.add(tf.keras.layers.Flatten())
        cnn_model.add(tf.keras.layers.Dense(128, activation='relu'))
        cnn_model.add(tf.keras.layers.Dropout(0.5))
        cnn_model.add(tf.keras.layers.Dense(43, activation='sigmoid'))  # 43 classes

        """
        The compilation process sets the model’s architecture and configures its parameters.
        """
        # Compile the model
        cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=(
            keras.optimizers.Adam(
                0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)), metrics=[
            'accuracy'])

        """
        # Train the model
        # We now train the neural network on the training data set.
        """
        history = cnn_model.fit(x=X_train_grscale_norm,
                                y=y_train,
                                batch_size=64,
                                epochs=30,
                                verbose=1,
                                validation_data=(X_valid_grscale_norm, y_valid))
        """
        # Show the loss value and metrics for the model on the test data set
        """
        score = cnn_model.evaluate(X_test_grscale_norm, y_test, verbose=0)
        print('Test Accuracy : {:.4f}'.format(score[1]))

        """
        # Plot the accuracy statistics of the model on the training and valiation data
        """
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        epochs = range(len(accuracy))

        ## Uncomment these lines below to show accuracy statistics
        line_1 = plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
        line_2 = plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
        plt.title('Accuracy on Training and Validation Data Sets vs Epochs')
        plt.setp(line_1, linewidth=2.0, marker='+', markersize=10.0)
        plt.setp(line_2, linewidth=2.0, marker='4', markersize=10.0)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.show()  # Uncomment this line to display the plot
        plt.gcf().canvas.set_window_title('Accuracy on Training and Validation Data Sets vs Epochs')

        # Save the model
        cnn_model.save("./road_sign.h5")

        """
        # predict on test data
        # Reload the model
        """
        model = load_model('./road_sign.h5')

        # Get the predictions for the test data set
        predicted_classes = np.argmax(model.predict(X_test_grscale_norm), axis=-1)

        # Retrieve the indices that we will plot
        y_true = y_test

        # Plot some of the predictions on the test data set
        for i in range(15):
            plt.subplot(5, 3, i + 1)
            plt.imshow(X_test_grscale_norm[i].squeeze(),
                       cmap='gray', interpolation='none')
            plt.title("Predict {}, Actual {}".format(predicted_classes[i],
                                                     y_true[i]), fontsize=10)
        plt.tight_layout()
        plt.savefig('road_sign_output.png')
        plt.gcf().canvas.set_window_title('Predictions on test data')
        plt.show()

    except Exception as e:
        exception_msg = str(e)
        exception_type, exception_object, exception_traceback = sys.exc_info()
        file_name = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        procedure_name = ""

        print(f'[ERROR] [fn] Exception type: {exception_type}, Exception: {exception_msg}, Line number: {line_number}')
        return


def image_normalize(image):
    """
    Normalize images to [0, 1] scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    image = np.divide(image, 255)
    return image

def gray_scale(image):
    """
    Convert images to gray scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def local_histo_equalize(image):
    """
    Apply local histogram equalization to grayscale images.
        Parameters:
            image: A grayscale image.
    """
    kernel = morp.disk(30)
    img_local = rank.equalize(image, selem=kernel)
    return img_local


def preprocess(data):
    """
    Applying the preprocessing steps to the input data.
        Parameters:
            data: An np.array compatible with plt.imshow.
    """
    gray_images = list(map(gray_scale, data))
    equalized_images = list(map(local_histo_equalize, gray_images))
    n_training = data.shape
    normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
    for i, img in enumerate(equalized_images):
        normalized_images[i] = image_normalize(img)
    normalized_images = normalized_images[..., None]
    return normalized_images

def predict_unknown_images():
    try:
        # Loading and resizing new test images
        new_test_images = []
        path = './unknown_images/'
        for image in os.listdir(path):
            img = cv2.imread(path + image)
            img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_test_images.append(img)


        #####
        expected_IDs = [13, 3, 14, 27, 17, 2]
        print("Number of new testing examples: ", len(new_test_images))

        # load labels data, in case model is already built
        if not signs:
            # Mapping ClassID to traffic sign names
            label_file = "./signnames.csv"
            with open(label_file, 'r') as csvfile:
                signnames = csv.reader(csvfile, delimiter=',')
                next(signnames, None)
                for row in signnames:
                    signs.append(row[1])
                csvfile.close()

        #print(signs)

        # Displaying the new testing examples, with their respective ground-truth labels:
        plt.figure(figsize=(15, 16))
        for i in range(len(new_test_images)):
            #print(str(i))
            plt.subplot(2, 5, i + 1)
            plt.imshow(new_test_images[i])
            plt.xlabel(signs[expected_IDs[i]])
            plt.ylabel("New testing image")
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.show()

        # New test data needs preprocessing
        """
        gray the images
        normalize the images
        """
        arr1 = np.sum(np.asarray(new_test_images) / 3, axis=3, keepdims=True)
        new_test_images_preprocessed = (arr1 - 128) / 128



        model = load_model('./road_sign.h5')

        # Get the predictions for the test data set
        predicted_classes = np.argmax(model.predict(new_test_images_preprocessed), axis=-1)

        # Retrieve the indices that we will plot
        y_true = expected_IDs

        # Plot some of the predictions on the test data set
        for i in range(len(new_test_images)):
            plt.subplot(5, 3, i + 1)
            plt.imshow(new_test_images_preprocessed[i].squeeze(),
                       cmap='gray', interpolation='none')
            plt.title("Predict {}, Actual {}".format(predicted_classes[i],
                                                     y_true[i]), fontsize=10)
        plt.tight_layout()
        plt.savefig('road_sign_output_predicted.png')
        plt.gcf().canvas.set_window_title('Accuracy of unknown images')
        plt.show()

    except Exception as e:
        exception_msg = str(e)
        exception_type, exception_object, exception_traceback = sys.exc_info()
        file_name = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        procedure_name = ""

        print(f'[ERROR] [fn] Exception type: {exception_type}, Exception: {exception_msg}, Line number: {line_number}')
        return


def capture_image_predict():
    try:
        path = './captured_images/'

        videoCaptureObject = cv2.VideoCapture(0)
        result = True
        while result:
            ret, frame = videoCaptureObject.read()
            cv2.imwrite(path + 'NewPicture.jpg', frame)
            result = False
        videoCaptureObject.release()
        cv2.destroyAllWindows()


        # Loading and resizing new test images
        new_test_images = []
        for image in os.listdir(path):
            img = cv2.imread(path + image)
            img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_test_images.append(img)

        #####
        # this is actual
        expected_IDs = [2]
        print("Number of new testing examples: ", len(new_test_images))

        # load labels data, in case model is already built
        if not signs:
            # Mapping ClassID to traffic sign names
            label_file = "./signnames.csv"
            with open(label_file, 'r') as csvfile:
                signnames = csv.reader(csvfile, delimiter=',')
                next(signnames, None)
                for row in signnames:
                    signs.append(row[1])
                csvfile.close()

        #print(signs)

        # Displaying the new testing examples, with their respective ground-truth labels:
        plt.figure(figsize=(15, 16))
        for i in range(len(new_test_images)):
            #print(str(i))
            plt.subplot(2, 5, i + 1)
            plt.imshow(new_test_images[i])
            plt.xlabel(signs[expected_IDs[i]])
            plt.ylabel("New testing image")
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.show()

        # New test data needs preprocessing
        """
        gray the images
        normalize the images
        """
        arr1 = np.sum(np.asarray(new_test_images) / 3, axis=3, keepdims=True)
        new_test_images_preprocessed = (arr1 - 128) / 128

        model = load_model('./road_sign.h5')

        # Get the predictions for the test data set
        predicted_classes = np.argmax(model.predict(new_test_images_preprocessed), axis=-1)

        # Retrieve the indices that we will plot
        y_true = expected_IDs

        # Plot some of the predictions on the test data set
        for i in range(len(new_test_images)):
            plt.subplot(5, 3, i + 1)
            plt.imshow(new_test_images_preprocessed[i].squeeze(),
                       cmap='gray', interpolation='none')
            plt.title("Predict {}, Actual {}".format(predicted_classes[i],
                                                     y_true[i]), fontsize=10)
        plt.tight_layout()
        plt.savefig('road_sign_output1.png')
        plt.gcf().canvas.set_window_title('Accuracy of unknown images')
        plt.show()

    except Exception as e:
        exception_msg = str(e)
        exception_type, exception_object, exception_traceback = sys.exc_info()
        file_name = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        procedure_name = ""

        print(f'[ERROR] [fn] Exception type: {exception_type}, Exception: {exception_msg}, Line number: {line_number}')
        return

"""
Areas of improvement
There are multiple potential areas of improvement in this project:

Image data augmentation
Hyper-parameter tuning
Try different base networks
Expand to more traffic sign classes
"""


def process_ml():
    try:
        # comment this line once the model is built with good accuracy, no need to run it everytime,
        # as it is one-time activity
        ######
        #build_cnn_model()
        predict_unknown_images()
         #capture_image_predict()
    except Exception as e:
        exception_msg = str(e)
        exception_type, exception_object, exception_traceback = sys.exc_info()
        file_name = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        procedure_name = ""

        print(f'[ERROR] [fn] Exception type: {exception_type}, Exception: {exception_msg}, Line number: {line_number}')
        return


if __name__ == "__main__":
    try:
        print('#---START---#')
        t1 = time.perf_counter()

        process_ml()

        t2 = time.perf_counter()
        print(f'Time taken(s)= {t2 - t1:0.2f}')
        print('#---END---#')
    except Exception as e:
        sys.exit(0)
