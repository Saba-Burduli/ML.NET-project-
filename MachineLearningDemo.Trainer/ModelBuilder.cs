using Microsoft.ML;
using MachineLearningDemo.Model;

namespace MachineLearningDemo.Trainer;

public static class ModelBuilder
{
    private const string OutputModelFile = "data/model.zip";
    private const string InputDataFile = "data/input.csv";

    private static readonly MLContext Context = new();

    public static void CreateModel(string inputDataFileName = InputDataFile,
        string outputModelFileName = OutputModelFile)
    {
        try
        {
            if (!File.Exists(inputDataFileName))
            {
                Console.WriteLine($"Input file not found: {inputDataFileName}");
                return;
            }

            Console.WriteLine("Loading car data...");
            var dataView = Context.Data.LoadFromTextFile<ModelInput>(
                path: inputDataFileName,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                trimWhitespace: true
            );

            Console.WriteLine("Splitting data...");
            var split = Context.Data.TrainTestSplit(dataView, testFraction: 0.2);

            Console.WriteLine("Building training pipeline...");
            var pipeline = Context.Transforms.Categorical.OneHotEncoding("ModelEncoded", nameof(ModelInput.Model))
                .Append(Context.Transforms.NormalizeMinMax(nameof(ModelInput.Year)))
                .Append(Context.Transforms.NormalizeMinMax(nameof(ModelInput.Mileage)))
                .Append(Context.Transforms.Concatenate("Features", "ModelEncoded", nameof(ModelInput.Year), nameof(ModelInput.Mileage)))
                .AppendCacheCheckpoint(Context)
                .Append(Context.Regression.Trainers.FastTree(labelColumnName: nameof(ModelInput.Price),
                    numberOfLeaves: 20,
                    numberOfTrees: 100,
                    minimumExampleCountPerLeaf: 10));

            Console.WriteLine("Training model...");
            var model = pipeline.Fit(split.TrainSet);

            Console.WriteLine("Evaluating model...");
            var predictions = model.Transform(split.TestSet);
            var metrics = Context.Regression.Evaluate(predictions, labelColumnName: nameof(ModelInput.Price));

            // Range: 0 to 1 (sometimes can be negative if the model sucks)
            // Shows how well the model explains the variance in the data.
            // 1 means perfect prediction.
            // 0 means that the model predicts nothing.
            // 0.92 means the model explains 92% of the variation in prices
            Console.WriteLine($"R-squared: {metrics.RSquared:0.00}");
            // Range: 0 to infinity (lower is better)
            // It tells you the average prediction error in the same unit as your target variable.
            // RMSE = 2200 means that on average, the predicted price is $2200 off from the actual price
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:#.##}");

            Console.WriteLine("Saving model...");
            Context.Model.Save(model, split.TrainSet.Schema, outputModelFileName);
            Console.WriteLine($"Model saved to: {outputModelFileName}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}