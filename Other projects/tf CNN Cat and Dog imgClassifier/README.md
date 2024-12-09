# Problem:

Complete the code to build a convolutional neural network (CNN) using TensorFlow 2.0 and Keras to classify images of cats and dogs with at least 63% accuracy (70% for extra credit). You'll implement image data generators, apply data augmentation, build and train a CNN model, and test its performance on a dataset. The project includes setting up image data pipelines, creating a model, and evaluating it on test images.

---------
# CNN architecture sketch

![reg](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/CNN%20Cat%20and%20Dog%20imgClassifier/img/representacion.png)

# Diagram

![axcv](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/CNN%20Cat%20and%20Dog%20imgClassifier/img/class%20diagram.png)

### Code Description

This script details the workflow for building, training, and evaluating an image classification model using a convolutional neural network (CNN) with TensorFlow.

1. **Setup and Library Imports**:
   - TensorFlow 2.x is configured if running on Google Colab. Necessary libraries such as `tensorflow`, `keras`, `numpy`, `matplotlib`, and `os` for file handling are imported.

2. **Data Download and Extraction**:
   - A ZIP file containing the cat and dog image dataset is downloaded and extracted. Paths to the training, validation, and test directories are defined.

3. **Dataset Exploration**:
   - The number of files in the training, validation, and test directories is counted to determine the dataset size. Variables such as `batch_size`, `epochs`, `IMG_HEIGHT`, and `IMG_WIDTH` are set for preprocessing and training.

4. **Data Generation**:
   - An image generator is set up using `ImageDataGenerator` to normalize image pixels to values between 0 and 1. Generators for training, validation, and test sets are configured, applying appropriate preprocessing for each.

5. **Image Visualization**:
   - A function is defined to visualize images and their classification probabilities. A set of training images with their probabilistic labels is displayed for inspection.

6. **Data Augmentation**:
   - The training image generator is redefined to include data augmentation techniques such as horizontal and vertical flipping, shifts, zoom, and shear to improve the model's robustness.

7. **Model Construction**:
   - A convolutional neural network (CNN) architecture (`Sequential`) is defined with `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers. The model is compiled with the `Adam` optimizer and the `SparseCategoricalCrossentropy` loss function, suitable for multiclass classification.

8. **Model Training**:
   - The model is trained using the training and validation data generators. The number of steps per epoch is calculated based on the dataset size and batch size.

  ![curve](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/CNN%20Cat%20and%20Dog%20imgClassifier/img/curva%20de%20aprendisaje.png)

9. **Model Evaluation**:
   - Accuracy and loss curves for both the training and validation sets are visualized over epochs. This provides insight into the model's performance during training.

![ryret](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/CNN%20Cat%20and%20Dog%20imgClassifier/img/Captura%20de%20pantalla%202024-07-19%20205554.png)

10. **Prediction and Evaluation**:
    - Model predictions on the test set are obtained. Test images along with their classification probabilities are displayed to evaluate the model's accuracy on unseen data.

11. **Performance Evaluation**:
    - Model predictions are compared with predefined correct answers to calculate the percentage of correct identification. It is determined if the model has passed the challenge based on a 63% accuracy threshold.

## Flowchart

![ouio](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/CNN%20Cat%20and%20Dog%20imgClassifier/img/diagram%20languaje.png)


## Fundamental Concepts

**Convolution operation** is like applying a filter to an image to detect patterns. Imagine sliding a small window (the filter) over the image and multiplying the values inside the window by the filter values. You then sum those products to get a new value for each window position. This process helps to highlight important features such as edges or textures.

A **convolutional neural network (CNN)** uses this convolution operation to analyze images. Instead of processing the entire image at once, a CNN uses filters that slide over the image to detect patterns such as edges, shapes, or textures. It then combines this information to recognize objects or complex features in the image.

CNNs have the advantage over dense layers because they detect patterns locally rather than globally. This means they can identify specific features in small areas of the image, such as edges or textures. This ability to focus on local details makes them particularly useful for image classification, as they can recognize and combine these features to identify objects in an image.

---
