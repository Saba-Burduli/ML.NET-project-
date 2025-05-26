using Microsoft.ML;
using MachineLearningDemo.Model;
using Spectre.Console;

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
            AnsiConsole.MarkupLine($"[bold green]Training Car Price Prediction Model[/]{Environment.NewLine}");

            if (!File.Exists(inputDataFileName))
            {
                AnsiConsole.MarkupLine($"[red]Input file not found: {inputDataFileName}[/]");
                return;
            }

            AnsiConsole.MarkupLine("[yellow]Loading car data...[/]");
            var dataView = Context.Data.LoadFromTextFile<ModelInput>(
                path: inputDataFileName,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                trimWhitespace: true
            );

            AnsiConsole.MarkupLine("[yellow]Splitting data...[/]");
            var split = Context.Data.TrainTestSplit(dataView, testFraction: 0.2);

            AnsiConsole.MarkupLine("[yellow]Building training pipeline...[/]");
            var pipeline = Context.Transforms.Categorical.OneHotEncoding("ModelEncoded", nameof(ModelInput.Model))
                .Append(Context.Transforms.NormalizeMinMax(nameof(ModelInput.Year)))
                .Append(Context.Transforms.NormalizeMinMax(nameof(ModelInput.Mileage)))
                .Append(Context.Transforms.Concatenate("Features", "ModelEncoded", nameof(ModelInput.Year),
                    nameof(ModelInput.Mileage)))
                .AppendCacheCheckpoint(Context)
                .Append(Context.Regression.Trainers.FastTree(labelColumnName: nameof(ModelInput.Price),
                    numberOfLeaves: 20,
                    numberOfTrees: 100,
                    minimumExampleCountPerLeaf: 10));

            AnsiConsole.MarkupLine("[yellow]Training model...[/]");
            var model = pipeline.Fit(split.TrainSet);

            AnsiConsole.MarkupLine("[yellow]Evaluating model...[/]");
            var predictions = model.Transform(split.TestSet);
            var metrics = Context.Regression.Evaluate(predictions, labelColumnName: nameof(ModelInput.Price));

            AnsiConsole.WriteLine();
            var table = new Table()
                .Border(TableBorder.Rounded)
                .AddColumn("Metric")
                .AddColumn("Value")
                .AddRow("R-squared", $"{metrics.RSquared:0.00}")
                .AddRow("RMSE", $"{metrics.RootMeanSquaredError:#.##}");

            AnsiConsole.Write(table);

            AnsiConsole.MarkupLine($"{Environment.NewLine}[yellow]Saving model...[/]");
            Context.Model.Save(model, split.TrainSet.Schema, outputModelFileName);
            AnsiConsole.MarkupLine($"[green]Model saved to: {outputModelFileName}[/]{Environment.NewLine}");
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
        }
    }
}