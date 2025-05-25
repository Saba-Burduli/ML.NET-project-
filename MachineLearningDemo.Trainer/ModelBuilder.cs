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
                .Append(Context.Transforms.Concatenate("Features", "ModelEncoded", nameof(ModelInput.Year),
                    nameof(ModelInput.Mileage)))
                .Append(Context.Regression.Trainers.Sdca(labelColumnName: nameof(ModelInput.Price),
                    maximumNumberOfIterations: 100));

            Console.WriteLine("Training model...");
            var model = pipeline.Fit(split.TrainSet);

            Console.WriteLine("Evaluating model...");
            var predictions = model.Transform(split.TestSet);
            var metrics = Context.Regression.Evaluate(predictions, labelColumnName: nameof(ModelInput.Price));

            Console.WriteLine($"R-squared: {metrics.RSquared:0.00}");
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