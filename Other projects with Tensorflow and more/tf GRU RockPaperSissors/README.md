# Problem:
### Rock-Paper-Scissors Challenge

In this project, you will create a program to play Rock-Paper-Scissors against four different bots. To pass, your program must win at least 60% of the games in each match, while a random choice typically wins 50% of the time.

**Details:**
- Implement your logic in `RPS.py`, specifically in the `player` function, which takes the opponent's last move ("R", "P", or "S") and returns the next move.
- The function receives an empty string for the first game in a match, as there's no previous play.
- Use multiple strategies to defeat all four bots based on their moves.

**Development Notes:**
- Do not modify `RPS_game.py`. Write all code in `RPS.py`.
- Use `main.py` to test your code with the `play` function, which takes two player functions, the number of games, and an optional verbosity argument.

**Testing:**
- Unit tests are in `test_module.py`, and you can run them by uncommenting the last line in `main.py`.

---------

## Hyperparameter Tuning and Model Evaluation with TensorFlow and Keras

In this repository, we explore the process of hyperparameter tuning and model evaluation for neural networks using TensorFlow and Keras, applied to a Rock-Paper-Scissors (RPS) game using advanced techniques such as Bayesian optimization to enhance model accuracy.

## I use GRU architecture: 

In the Rock-Paper-Scissors (RPS) project, a Gated Recurrent Unit (GRU) architecture was used instead of Long Short-Term Memory (LSTM). The GRU was chosen for its ability to 
capture sequential patterns with lower computational complexity compared to LSTM. While both architectures are designed to handle long-term dependencies in sequential data, GRUs 
tend to be more efficient and require fewer parameters than LSTMs, which is beneficial for projects that need a balance between performance and efficiency.

![ewrwe](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/GRU%20RockPaperSissors/img/GRU.png)

### Project Description

This project focuses on the development and fine-tuning of a neural network model to predict moves in the RPS game. The project structure includes:

1. **Dependency Installation**:
    - We install `keras-tuner` for hyperparameter optimization.
    - We install `pydot` and `graphviz` for graph visualization.
    - We install `scikit-optimizer` for additional optimization techniques.

2. **Environment Setup**:
    - We mount Google Drive and add the project directory to `sys.path`.

3. **Data Preparation**:
    - Sequences of data and labels are generated using custom functions.
    - Data is split into training and testing sets.

4. **Model Building**:
    - We create a base model using GRU (Gated Recurrent Unit) layers, Dropout, BatchNormalization, and GlobalAveragePooling1D.
    - The basic model is trained and its performance evaluated.

5. **Hyperparameter Optimization**:
    - We define a model-building function that accepts hyperparameters.
    - We use the Bayesian optimizer from `keras-tuner` to find the best hyperparameter configuration.
    - We evaluate the best model found and save the results.

6. **Model Evaluation and Usage**:
    - We evaluate the final model on the test set.
    - We save and load the model for use in the RPS game.
    - We implement an AI player that uses the model to predict moves and play against predefined opponents.


### Code

- **`Mi_ML.py`**: Contains functions and classes related to data preparation and model definition.
- **`RPS_game.py`**: Includes functions for playing the RPS game with different types of players.
- **`RPS1.py`**: Defines the base player for the RPS game.
- **`main_notebook.ipynb`**: The main notebook containing the entire workflow, from package installation to model evaluation.

### Results

The tuned model achieves an accuracy of 60.94% on the test set. Graphs show the progress of loss and accuracy during training.

### How to Run the Code

1. **Prepare the Environment**: Install the necessary dependencies.
2. **Run the Notebook**: Load the main notebook and follow the instructions to execute the code.

-------

# Learning curve
![rtyrt](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/GRU%20RockPaperSissors/img/sec_14_GRU%20curve.png)

We observe a learning curve that illustrates how the model's performance improves as it is trained with more data and epochs. Initially, the model shows rapid learning, but eventually, the improvement stabilizes and tends to reach an optimal performance level

---------

### Notebook Overview

**Objective**: This notebook is focused on evaluating the generalization performance of our models through hyperparameter tuning. We will use Keras Tuner for this purpose. Documentation for Keras Tuner can be found [here](https://keras.io/keras_tuner/).

**Installation**:
```python
!pip install keras-tuner
!pip install pydot graphviz
!pip install scikit-optimizer
```

**Setup**:
1. **Keras Tuner**: A tool for hyperparameter tuning in TensorFlow/Keras models.
2. **Pydot and Graphviz**: Required for visualizing model architectures.
3. **Scikit-Optimizer**: For optimization tasks.

**Initialization**:
```python
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append('/content/drive/My Drive/rps2')

from Mi_ML import *
```
Mount Google Drive and include necessary paths for accessing custom modules.

**Data Preparation**:
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
%matplotlib inline

comb = inicio_RPSgame()
X, y = Secuencial_OneHot(comb, 5)
```
The dataset is loaded and preprocessed into sequences of 5 for training.

**Model Definition and Training**:
```python
def create_model(optimizer='adam', neurons=500, dropout_rate=0.2, l1_reg=0.01, l2_reg=0.01):
    model = Sequential()
    model.add(GRU(neurons, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(GRU(neurons, return_sequences=True))
    model.add(Dropout(dropout_rate // 1.5))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(neurons // 3, activation='relu'))
    model.add(Dropout(dropout_rate // 2))
    model.add(Dense(X_train.shape[2], activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```
A GRU-based model with dropout and batch normalization is defined.

**Training**:
```python
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=35, batch_size=30, validation_split=0.2, callbacks=[early_stopping])
```
The model is trained with early stopping to prevent overfitting.

**Evaluation and Visualization**:
```python
loss, accuracy = model.evaluate(X_train, y_train)
print(f'Loss: {loss}, Accuracy: {accuracy}')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Loss : {NAME}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'Accuracy : {NAME}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
Model performance is evaluated and visualized for both training and validation metrics.

**Hyperparameter Tuning**:
```python
from keras_tuner.tuners import BayesianOptimization

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.GRU(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(keras.layers.GRU(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        return_sequences=True))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(
        units=hp.Int('dense_units', min_value=32, max_value=512, step=32),
        activation='relu'))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout3', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(keras.layers.Dense(X_train.shape[2], activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='my_dir',
    project_name='hyperparam_tuning_bayesian')

tuner.search(X_train, y_train, epochs=15, validation_split=0.2)
```
Hyperparameter tuning is performed using Bayesian Optimization to find the best model configuration.

**Final Model Training and Evaluation**:
```python
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=30, validation_split=0.2)

model = best_model
model.evaluate(X_test, y_test)
model.save("sec_14_GRU.h5")
```
The best model from the hyperparameter tuning is evaluated and saved for future use.

**AI Player Integration**:
```python
label_map = {0: 'P', 1: 'R', 2: 'S'}

def rps_one_hot_single(play):
    if play == 'P':
        return [1, 0, 0]
    elif play == 'R':
        return [0, 1, 0]
    elif play == 'S':
        return [0, 0, 1]

def AI_player5(prev_play, opponent_history=[]):
    if prev_play == '':
        prev_play = 'R'

    opponent_history.append(prev_play)

    sequence_length = 5
    if len(opponent_history) < sequence_length:
        return 'R'

    input_sequence = opponent_history[-sequence_length:]
    one_hot_sequence = [rps_one_hot_single(play) for play in input_sequence]

    input_sequence = np.array(one_hot_sequence, dtype=np.float32)
    input_sequence = np.expand_dims(input_sequence, axis=0)

    prediction = model.predict(input_sequence, verbose=0)
    prediction_label = np.argmax(prediction, axis=-1)[0]

    label_map = {0: 'P', 1: 'R', 2: 'S'}
    predicted_move = label_map[prediction_label]

    if predicted_move == 'R':
        return 'P'
    elif predicted_move == 'P':
        return 'S'
    elif predicted_move == 'S':
        return 'R'
```
The trained model is used in an AI player for the Rock-Paper-Scissors game, where it predicts the opponent's move and responds optimally.

**Play and Test**:
```python
from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS1 import player
from unittest import main

with tf.device('/device:GPU:0'):
    play(AI_player5, quincy, 1000)
    play(AI_player5, abbey, 1000)
    play(AI_player5, kris, 1000)
    play(AI_player5, mrugesh, 1000)
```
The AI player is tested against various opponents.

-----

# Layer Architecture Diagram

![iopoi](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/GRU%20RockPaperSissors/img/GRU%20neural%20net%20architecture.png)
