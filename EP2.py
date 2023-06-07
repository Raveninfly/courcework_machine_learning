import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from scipy.stats import linregress

# Load dataset
dataset = pd.read_csv("data.csv")

# Prepare predictors and target
predictors = pd.get_dummies(dataset.drop("salary", axis=1))
target = dataset["salary"]

# Split into training and testing sets
predictors_train, predictors_test, target_train, target_test = train_test_split(
    predictors, target, test_size=0.3, random_state=100)

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectFromModel(LassoCV(cv=5, random_state=100))),
    ("regressor", LinearRegression())
])

# Train model
pipeline.fit(predictors_train, target_train)

# Make predictions and calculate MAPE
predictions = pipeline.predict(predictors_test)
mape = mean_absolute_percentage_error(target_test, predictions)

# Print model parameters and MAPE
b0, b1 = pipeline.named_steps["regressor"].intercept_, pipeline.named_steps["regressor"].coef_[0]
print(f"Intercept (b0): {b0:.5f}")
print(f"Coefficient (b1): {b1:.5f}")
print(f"MAPE: {mape:.5f}")

# Set background color to white
plt.rcParams['axes.facecolor'] = 'white'

numeric_predictors_test = predictors_test.select_dtypes(include=[np.number])
for predictor in numeric_predictors_test.columns:
    # Get regression parameters
    slope, intercept, _, _, _ = linregress(predictors_test[predictor], target_test)
    # Generate and plot regression line
    line_values = slope * predictors_test[predictor] + intercept
    plt.figure(figsize=(10, 6))
    plt.scatter(predictors_test[predictor], target_test, color="blue", label="Actual salary")
    plt.plot(predictors_test[predictor], line_values, color="red", label="Predicted salary")
    plt.title("Relationship between "+predictor+" and Salary", fontsize=16, fontweight='bold')
    plt.xlabel(predictor, fontsize=12)
    plt.ylabel("Salary", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.show()
