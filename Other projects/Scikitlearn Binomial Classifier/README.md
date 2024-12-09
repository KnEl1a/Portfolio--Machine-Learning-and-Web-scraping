# [Problem](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/neural-network-sms-text-classifier)

In this challenge, you need to create a machine learning model that will classify SMS messages as either "ham" or "spam". A "ham" message is a normal message sent by a friend. A "spam" message is an advertisement or a message sent by a company.

You should create a function called predict_message that takes a message string as an argument and returns a list. The first element in the list should be a number between zero and one that indicates the likeliness of "ham" (0) or "spam" (1). The second element in the list should be the word "ham" or "spam", depending on which is most likely.

For this challenge, you will use the SMS Spam Collection dataset. The dataset has already been grouped into train data and test data.

The first two cells import the libraries and data. The final cell tests your model and function. Add your code in between these cells.



---

# Spam vs. Ham Classifier with scikit-learn

This project demonstrates a simple binary text classification task to distinguish between "spam" and "ham" messages using scikit-learn. The classifier is built using a Naive Bayes model with the `CountVectorizer` for text feature extraction. Here's a step-by-step breakdown of the process:

# Diagram

![rty](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/Scikitlearn%20Binomial%20Classifier/diagram%20c%20N.png?raw=true)

## Setup

First, we install the necessary libraries. TensorFlow is installed via `pip`, and `tensorflow-datasets` is also included:

```python
try:
    # %tensorflow_version only exists in Colab.
    !pip install tf-nightly
except Exception:
    pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
```

We then download the dataset files for training and validation:

```python
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"
```

## Data Preparation

The dataset is loaded into pandas DataFrames. The TSV (Tab-Separated Values) files are read with the `pd.read_csv()` function, specifying the tab character as the separator:

```python
column_names = ["label", "text"]
df_train = pd.read_csv(train_file_path, sep="\t", names=column_names)
df_test = pd.read_csv(test_file_path, sep="\t", names=column_names)
```

We check the shape of the test dataset to ensure it loaded correctly:

```python
print(df_test.head().shape)
df_test
```

## Feature Extraction and Model Training

We combine the training and test datasets and convert the labels to binary values (`0` for "ham" and `1` for "spam"). We then split the data into training and testing sets:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.concat([df_train, df_test], ignore_index=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()  # Tokenization
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_counts, y_train)
```

We evaluate the model using accuracy and a detailed classification report:

```python
y_pred = clf.predict(X_test_counts)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
```

## Prediction Function

The `predict_message` function classifies a given text as "spam" or "ham" using the trained model and vectorizer:

```python
def predict_message(text, vectorizer=vectorizer, model=clf):
    """
    Predicts if a text is 'spam' or 'ham' using a trained model.

    Args:
    - text (str): The text to classify.
    - vectorizer (CountVectorizer): The vectorizer used to transform the text.
    - model (MultinomialNB): The trained model.

    Returns:
    - str: 'spam' or 'ham'.
    """
    text_counts = vectorizer.transform([text])
    prediction = model.predict(text_counts)[0]
    return 'spam' if prediction == 1 else 'ham'
```

## Testing the Model

We have a testing function to validate the model with a set of predefined messages and expected results:

```python
def test_predictions():
    test_messages = ["how are you doing today",
                     "sale today! to stop texts call 98912460324",
                     "i dont want to go. can we try it a different day? available sat",
                     "our new mobile video service is live. just install on your phone to start watching.",
                     "you have won £1000 cash! call to claim your prize.",
                     "i'll bring it tomorrow. don't forget the milk.",
                     "wow, is your arm alright. that happened to me one time too"]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        if prediction != ans:
            print(f"Failed for message: '{msg}' | Expected: {ans} | Got: {prediction}")
            passed = False

    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")

test_predictions()
```

### Results

The model achieved an accuracy of approximately 97.76% on the test set, demonstrating its effectiveness in classifying messages as "spam" or "ham."

---
