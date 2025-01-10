# Problem:

Predict healthcare costs using a DL algorithm with tf. Utilize a dataset with healthcare cost information to train a model, converting categorical data to numbers and splitting the data into training and testing sets. The goal is to achieve a Mean Absolute Error under $3500. The challenge involves creating and training a regression model, evaluating its performance, and visualizing predictions against test data.

---------

# Diagram

![lyl](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/Dense%20Layers%20Health%20Cost%20Predictor/img/class%20diagram.png)

---

### Code Description

The script presents a complete workflow for data preprocessing and creating a classification model using TensorFlow.

1. **Library Imports**:
   - TensorFlow documentation is installed, and essential libraries such as `matplotlib`, `numpy`, `pandas`, and `tensorflow` are imported. Tools for TensorFlow documentation are also set up.

2. **Data Loading**:
   - A CSV file containing medical insurance cost data is downloaded and loaded into a `pandas` DataFrame. The last rows of the dataset are visualized for preliminary inspection.

3. **Data Exploration**:
   - General information about the dataset is evaluated, confirming that there are no null values, and descriptive statistics are presented. A histogram of age is plotted to analyze the distribution of the data.

4. **Data Splitting**:
   - The DataFrame is partitioned into training (80%) and test (20%) sets using `train_test_split`. The size of both sets is printed to verify the correct division.

5. **Data Preparation**:
   - Expense labels are extracted and removed from the training and test DataFrames. Categorical and numerical columns are defined, and unique values in categorical columns are visualized for analysis.

6. **Feature Preprocessing**:
   - Each categorical column is converted into numerical feature columns using `tf.feature_column`. Numerical columns are defined as `numeric_column` for use in the model.

7. **Input Function Creation**:
   - A `make_input_fn` function is defined to transform DataFrames into `tf.data.Dataset` objects, which allow for model training and evaluation. Functions for training and evaluation are configured, specifying epochs, batch size, and whether shuffling should be performed.

8. **Model Training and Evaluation**:
   - A linear classifier is built using `tf.estimator.LinearClassifier` with the preprocessed features. The model is trained with the training dataset and evaluated with the test dataset. The model's accuracy is printed to assess its performance.

## Flowchart:

![asd](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/Dense%20Layers%20Health%20Cost%20Predictor/img/flow%20chart%202.png)


### Fundamental Concepts

![acx](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/Dense%20Layers%20Health%20Cost%20Predictor/img/perceptron%20simple.jpg)

A simple perceptron is a basic unit of a neural network. It works as follows:

1. **Input:** Receives a series of values (such as features of an image or text data).
2. **Weights:** Each input is multiplied by a weight. Weights determine the importance of each input.
3. **Weighted Sum:** All the weighted values are summed.
4. **Activation Function:** The sum is passed through a function (such as a sigmoid or ReLU function) to decide if the unit should be activated or not.
5. **Output:** Produces a result that can be used as input for other units or to provide the final prediction.

In TensorFlow, a "dense" layer is simply a group of connected perceptrons. Each neuron in a layer receives inputs from all the neurons in the previous layer, applies weights, and then uses an activation function to produce its output. Dense layers allow the network to learn and capture complex patterns in the data.

---
