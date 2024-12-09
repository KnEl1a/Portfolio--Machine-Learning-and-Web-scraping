# KNN

![fgc](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/KNN%20bk%20rec%20engine/img/f.png)

# Diagram

![sad](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/KNN%20bk%20rec%20engine/img/diagram.png)


# [Problem](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/book-recommendation-engine-using-knn)

In this challenge, you will develop a book recommendation algorithm using the K-Nearest Neighbors (KNN) algorithm. You will utilize the Book-Crossings dataset, which includes 1.1 million ratings (on a scale of 1-10) for 270,000 books provided by 90,000 users.

### Steps:
1. **Import and clean the data:** Filter out users with fewer than 200 ratings and books with fewer than 100 ratings to ensure statistical significance.
2. **Develop the model:** Use the `NearestNeighbors` class from `sklearn.neighbors` to create a KNN model. This model will measure the distance between books to determine their similarity.
3. **Create the recommendation function:** Define a function named `get_recommends` that takes a book title as an argument and returns a list of 5 similar books along with their distances from the provided book.

### Example:
Calling `get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")` should return a list where the first element is the input book title, and the second element is a list of five recommended books with their corresponding distances, such as:

```python
[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.793983519077301], 
    ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479], 
    ['Interview with the Vampire', 0.7345068454742432],
    ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
    ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
  ]
]
```

---

## Book Recommendation System (Algorithm using K-Nearest Neighbors)

This project implements a book recommendation system using collaborative filtering and the k-nearest neighbors (KNN) algorithm. The system is designed to suggest books based on user ratings and provides a simple and efficient method for users to find new books they might enjoy.

---

## Overview

## Steps and Methodology

### 1. Data Import and Cleaning

We started by importing the Book-Crossings dataset which consists of three files: 

- `BX-Books.csv` - Contains information about the books (ISBN, title, author)
- `BX-Book-Ratings.csv` - Contains user ratings for the books

The data was imported into pandas DataFrames and cleaned to ensure it was in the correct format for analysis.

```python
import pandas as pd

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# Import CSV data into DataFrames
df_books = pd.read_csv(
    books_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})
```

### 2. Data Filtering

To ensure statistical significance, we removed users with less than 200 ratings and books with less than 100 ratings.

```python
# Filter users with less than 200 ratings
user_rating_counts = df_ratings.groupby("user").rating.count()
significant_users = user_rating_counts[user_rating_counts >= 200].index
df_ratings_filtered = df_ratings[df_ratings["user"].isin(significant_users)]

# Filter books with less than 100 ratings
book_rating_counts = df_ratings.groupby("isbn").rating.count()
significant_books = book_rating_counts[book_rating_counts >= 100].index
df_books_filtered = df_books[df_books["isbn"].isin(significant_books)]
```

### 3. Merging Data

We merged the filtered books and ratings DataFrames on the ISBN column.

```python
df_merged = pd.merge(df_books_filtered, df_ratings_filtered, on='isbn')
```

### 4. Model Building

Using the merged data, we created a user-item matrix and applied the Nearest Neighbors algorithm to find similar books.

```python
from sklearn.neighbors import NearestNeighbors

# Create a pivot table
user_item_matrix = df_merged.pivot(index='title', columns='user', values='rating').fillna(0)

# Fit the NearestNeighbors model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_item_matrix.values)
```

### 5. Recommendation Function

We created the `get_recommends` function to return a list of 5 similar books for a given book title.

```python
def get_recommends(book):
    book_index = user_item_matrix.loc[book].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(book_index, n_neighbors=6)
    similar_books = [
        (user_item_matrix.index[i], distances[0][idx])
        for idx, i in enumerate(indices[0]) if i != book_index
    ]
    return [book, similar_books]
```

### 6. Testing

We tested the recommendation system with a sample book.

```python
print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
```

### Expected Output

The expected output of the `get_recommends` function for the book "The Queen of the Damned (Vampire Chronicles (Paperback))" is:

```
[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.793983519077301], 
    ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479], 
    ['Interview with the Vampire', 0.7345068454742432],
    ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
    ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
  ]
]
```

## Conclusion

This project demonstrates the application of K-Nearest Neighbors for building a book recommendation system. By filtering the dataset for significant users and books, we ensure that the recommendations are statistically meaningful and relevant.

Feel free to explore the code and experiment with different parts of the dataset to improve and extend the recommendation algorithm.

## Main Dependencies

- pandas
- numpy
- scikit-learn


# Flow Diagram

![ytut](https://github.com/KnEl1a/Deep_Learning_with_Tensorflow_and_sklearn_fccamp/blob/main/KNN%20bk%20rec%20engine/img/flow%20diagram.png)

