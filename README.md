Car Price Prediction - ML.NET
This project is a simple ML.NET application that predicts the price of used cars based on:

Model
Year
Mileage
It uses a regression model trained on synthetic data for:

Dodge Challenger
Ford Mustang
BMW 3 Series
Mercedes C-Class
ML Pipeline
Load CSV training data (up to 1M rows)
One-hot encode the Model field
Normalize numeric fields: Year, Mileage
Combine features: ModelEncoded + Year + Mileage
Train using the FastTree regression algorithm
Evaluate the model and save it to disk
Dataset Sample
Model,Year,Mileage,Price
Ford Mustang,2020,60000,14000.00
BMW 3 Series,2024,10000,37000.00
This dataset is generated using ChatGPT and doesn't represent real world data.

Evaluation Example (on synthetic data)
R-squared: 1.00
RMSE: 37.04
Requirements
.NET 8.0 SDK
NuGet Packages:
Microsoft.ML
Microsoft.ML.FastTree
Spectre.Console
License
MIT – use it, do whatever you want.
