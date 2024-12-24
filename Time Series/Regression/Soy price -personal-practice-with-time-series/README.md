# Soy Seasonality - Time Series Practice

## Project Objective

The goal of this project is to analyze the seasonality of soybean production using time series analysis techniques. We will employ various methods learned in the time series course to decompose and model soybean production data, identifying seasonal patterns to accurately forecast future production trends. Subsequently, we will apply hybrid Machine Learning models for soybean value prediction.

## Common Methodology

1. **Data Preparation**: Organize the data and select the relevant year.
2. **Deterministic Features**: Create seasonal and cyclical features using Fourier transforms and a deterministic process.
3. **Training**: Train a hybrid model combining linear regression and XGBoostRegressor with the generated features.
4. **Prediction and Visualization**: Make predictions and visualize the results, evaluating model performance using the "RMSE" metric.

## Package Installation

To run the code, install the following libraries using `pip`:

```bash
pip install yfinance seaborn matplotlib numpy pandas scikit-learn statsmodels xgboost
```

## **Time Series Analysis and Hybrid Models**

### **Periodogram and Seasonality**

The periodogram reveals significant weekly seasonality, indicated by a peak at the weekly frequency (52). This suggests that closing prices exhibit a weekly repeating pattern, while other frequencies show smaller peaks, indicating less variability at those time scales.

### **Capturing Seasonal Patterns with CalendarFourier and DeterministicProcess**

To model and capture the observed seasonality, we use:

1. **CalendarFourier**:
   - **Description**: Utilizes Fourier transforms to model seasonal components through sinusoidal and cosine waves.
   - **Utility**: Ideal for identifying and modeling seasonal effects with various frequencies.

2. **DeterministicProcess**:
   - **Description**: A framework that uses deterministic functions to build time series models, including trend and seasonal terms.
   - **Utility**: Allows for the integration of seasonal and trend terms, ideal for fitting and predicting data with complex seasonalities.

**Summary**:
- CalendarFourier models seasonality using Fourier terms.
- DeterministicProcess combines these terms with other components for a robust model.

### **Issues and Solutions**

1. **Date Format**: A specific date format and separate columns for weekdays are required.
   - **Solution**: An auxiliary date (2020) was used, and missing values were filled with `fillna(mean)`.

### **Creating Fourier Terms**

```python
# Create DataFrame from Series
df = y_allY_mean.reset_index()
df.columns = ['day_of_year', 'Close']

# Create a date index to represent the days of the year
date_rng = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
df_date = pd.DataFrame(date_rng, columns=['date'])
df_date['day_of_year'] = df_date['date'].dt.dayofyear

# Merge data with average values
df = pd.merge(df_date, df, on='day_of_year', how='left')

# Fill missing values
df['Close'] = df['Close'].fillna(method='ffill')

# Set the date index
df.set_index('date', inplace=True)
mean = df['Close'].mean()
df = df.fillna(mean)  # Fill NaN values with the mean
df
```

### **Creating Fourier Terms**

```python
# Create Fourier terms
fourier = CalendarFourier(freq='W', order=4)

# Create the deterministic process
dp = DeterministicProcess(
    index=df.index,
    constant=True,
    order=1,
    seasonal=False,
    additional_terms=[fourier],
    drop=True
)

# Generate deterministic terms
X = dp.in_sample()
```

### **Model Fitting**

```python
# Assign values to y without using inplace=True
y = df['Close']

# Fit the model
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
```

**Prediction vs. Actual Values Plot**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='Real Close')
plt.plot(df.index, y_pred, label='Seasonal Prediction', color='red')
plt.legend()
plt.title('Predicted Mean Close Price per Day of Year (All Years)')

# Hide X-axis labels
plt.gca().xaxis.set_major_formatter(plt.NullFormatter())

plt.show()
```

### **Using Hybrid Models in Time Series Prediction**

**Models:**

1. **Linear Regression**:
   - **Description**: Estimates linear relationships between variables.
   - **Utility**: Captures basic trends and seasonal effects.

2. **XGBoostRegressor**:
   - **Description**: A tree-based model known for its accuracy and ability to handle non-linear patterns.
   - **Utility**: Captures complex and non-linear patterns.

**Hybrid Models**:
- **Description**: Combine linear and advanced models, leveraging the strengths of both.
- **Advantage**: Enhances accuracy by capturing both simple and complex patterns.

### **Defining a Class for Hybrid Models**

```python
# Fit method of the BoostedHybrid class
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2

    def fit(self, X_1, X_2, y):
        self.model_1.fit(X_1, y)
        y_fit = pd.Series(
            self.model_1.predict(X_1),
            index=X_1.index
        )
        y_resid = y - y_fit
        self.model_2.fit(X_2, y_resid)

    def predict(self, X_1, X_2):
        y_pred = pd.Series(
            self.model_1.predict(X_1),
            index=X_1.index
        )
        y_pred += self.model_2.predict(X_2)
        return y_pred
```

### **Training and Evaluating the Hybrid Model**

```python
from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, shuffle=False)

# Create matrices for X_1
dp_train = DeterministicProcess(index=y_train.index, order=1)
X_1_train = dp_train.in_sample()

dp_test = DeterministicProcess(index=y_test.index, order=1)
X_1_test = dp_test.in_sample()

# Create the hybrid model
model = BoostedHybrid(LinearRegression(), XGBRegressor())

# Fit the model
model.fit(X_1_train, X_train, y_train)

# Predictions
y_pred_train = model.predict(X_1_train, X_train)
y_pred_test = model.predict(X_1_test, X_test)

# Clip to avoid negative values
y_pred_train = y_pred_train.clip(0.0)
y_pred_test = y_pred_test.clip(0.0)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_train.index, y_train, label='Train Actual', color='blue')
plt.plot(y_train.index, y_pred_train, label='Train Prediction', color='orange', linestyle='dotted')
plt.plot(y_test.index, y_test, label='Test Actual', color='green')
plt.plot(y_test.index, y_pred_test, label='Test Prediction', color='red', linestyle='dotted')
plt.legend()
plt.title('Predictions vs. Actual Values (Train and Test)')
plt.show()
```

**Model Evaluation**

```python
from sklearn.metrics import mean_squared_error

train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

print(f"Train RMSE: {train_rmse:.2f}\nTest RMSE: {test_rmse:.2f}")
```

**Conclusion**:
- The importance of temporal order in time series requires an adapted approach.
- Hybrid models combine simple and advanced models to improve accuracy.
- Identifying and modeling seasonalities is crucial for accurate predictions
