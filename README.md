# Car Price Prediction Using Linear Regression – ML.NET

This project predicts the price of used cars based on key features such as **Model**, **Year**, and **Mileage** using a regression model built with **ML.NET**.

##  Project Description

The model is trained on a synthetic dataset and supports the following car models:

- Dodge Challenger  
- Ford Mustang  
- BMW 3 Series  
- Mercedes C-Class  

It leverages the **FastTree Regression** algorithm for training and makes use of feature engineering steps such as normalization and one-hot encoding.

##  Machine Learning Pipeline

1. **Load CSV** training data (up to 1 million rows)
2. **One-hot encode** the `Model` column
3. **Normalize** the `Year` and `Mileage` columns
4. **Concatenate** all features into a single input vector
5. **Train** the model using the `FastTree` regression algorithm
6. **Evaluate** the model’s accuracy
7. **Save** the trained model to disk for reuse

##  Sample Dataset

```csv
Model,Year,Mileage,Price
Ford Mustang,2020,60000,14000.00
BMW 3 Series,2024,10000,37000.00
