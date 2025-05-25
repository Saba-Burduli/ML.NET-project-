# Car Price Prediction - ML.NET

This project is a simple ML.NET application that predicts the price of used cars based on:

- **Model**
- **Year**
- **Mileage**

It uses a regression model trained on synthetic data for:
- **Dodge Challenger**
- **Ford Mustang**
- **BMW 3 Series**
- **Mercedes C-Class**

---

## ML Pipeline

1. Load CSV training data (up to 1M rows)
2. One-hot encode the `Model` field
3. Normalize numeric fields: `Year`, `Mileage`
4. Combine features: `ModelEncoded + Year + Mileage`
5. Train using the **FastTree regression** algorithm
6. Evaluate the model and save it to disk

---

## Dataset Sample

```csv
Model,Year,Mileage,Price
Ford Mustang,2020,60000,14000.00
BMW 3 Series,2024,10000,37000.00
```

---

## Evaluation Example (on synthetic data)

```
R-squared: 1.00
RMSE: 37.04
```

---

## Requirements

- [.NET 8.0 SDK](https://dotnet.microsoft.com/)
- NuGet Packages:
  - `Microsoft.ML`
  - `Microsoft.ML.FastTree`

---

## License

MIT – use it, do whatever you want.
