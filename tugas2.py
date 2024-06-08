import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to read data from CSV file
def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data['Hours Studied'].values, data['Performance Index'].values.astype(float)

# Function to calculate linear regression coefficients
def linear_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - (slope * x_mean)
    return slope, intercept

# Function to calculate power regression coefficients
def power_regression(x, y):
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept = linear_regression(log_x, log_y)
    return slope, np.exp(intercept)

# Function to predict values using linear regression
def predict_linear(x, slope, intercept):
    return slope * x + intercept

# Function to predict values using power regression
def predict_power(x, slope, intercept):
    return intercept * (x ** slope)

# Function to calculate RMS error
def rms_error(predicted, actual):
    return np.sqrt(np.mean((predicted - actual) ** 2))

# Read the data
TB, NT = read_csv('student_performance.csv')

# Calculate linear regression coefficients
linear_slope, linear_intercept = linear_regression(TB, NT)

# Calculate power regression coefficients
power_slope, power_intercept = power_regression(TB, NT)

# Predict values
linear_predictions = predict_linear(TB, linear_slope, linear_intercept)
power_predictions = predict_power(TB, power_slope, power_intercept)

# Calculate RMS errors
linear_rms_error = rms_error(linear_predictions, NT)
power_rms_error = rms_error(power_predictions, NT)

print("Linear Regression RMS Error:", linear_rms_error)
print("Power Regression RMS Error:", power_rms_error)

# Plot data and regression lines
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(TB, NT, color='blue', label='Actual Data')
plt.plot(TB, linear_predictions, color='red', label='Linear Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Linear Regression')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(TB, NT, color='blue', label='Actual Data')
plt.plot(TB, power_predictions, color='green', label='Power Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Power Regression')
plt.legend()

plt.tight_layout()
plt.show()