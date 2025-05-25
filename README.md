# Car Price Prediction - ML.NET Console App

This project is a simple ML.NET console application that predicts the price of used cars based on:

- **Model**
- **Year**
- **Mileage**

It uses a regression model trained on sample data for:
- **Dodge Challenger**
- **Ford Mustang**
- **BMW 3 Series**
- **Mercedes C-Class**

---

## ML Pipeline

1. Load CSV data
2. One-hot encode the car model
3. Combine features: `Model + Year + Mileage`
4. Train using **SDCA regression**
5. Evaluate and save the model

---

## Dataset Sample

```csv
Model,Year,Mileage,Price
Ford Mustang,2020,60000,14000.00
BMW 3 Series,2024,10000,37000.00
```

---

## Evaluation Example

```
R-squared: 0.92
RMSE: 2115.67
```

---

## Requirements

- [.NET 8.0 SDK](https://dotnet.microsoft.com/)
- ML.NET NuGet packages

---

## License

MIT.
