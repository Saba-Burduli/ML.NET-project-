using MachineLearningDemo.ConsoleApp;

var predictor = new PredictionService();

Console.Write("Enter car model (Dodge Challenger, Ford Mustang, BMW 3 Series, Mercedes C-Class): ");
var model = Console.ReadLine()!;

Console.Write("Enter year (2020-2024): ");
var year = int.Parse(Console.ReadLine()!);

Console.Write("Enter mileage (e.g., 50000): ");
var mileage = int.Parse(Console.ReadLine()!);

var price = predictor.Predict(model, year, mileage);

Console.WriteLine($"Predicted car price: ${price:0.00}");