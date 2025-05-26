using Spectre.Console;
using MachineLearningDemo.ConsoleApp;

var predictor = new PredictionService();

var supportedCarModels = new[]
{
    "Dodge Challenger",
    "Ford Mustang",
    "BMW 3 Series",
    "Mercedes C-Class"
};

// 2020 to 2025
var supportedYears = Enumerable.Range(2020, 6).ToList();

AnsiConsole.MarkupLine($"[bold green]Car Price Predictor[/]{Environment.NewLine}");

var model = AnsiConsole.Prompt(
    new SelectionPrompt<string>()
        .Title("Select a [green]car model[/]:")
        .PageSize(10)
        .AddChoices(supportedCarModels)
);

var year = AnsiConsole.Prompt(
    new SelectionPrompt<int>()
        .Title("Select the [blue]year[/]:")
        .PageSize(6)
        .AddChoices(supportedYears)
);

var mileage = AnsiConsole.Prompt(
    new TextPrompt<int>("Enter the [yellow]mileage[/] (e.g., 50000):")
        .PromptStyle("yellow")
        .Validate(m => m >= 0
            ? ValidationResult.Success()
            : ValidationResult.Error("[red]Mileage must be non-negative integer[/]"))
);

var price = predictor.Predict(model, year, mileage);

AnsiConsole.WriteLine();

AnsiConsole.MarkupLine(price >= 0
    ? $"💰 [bold green]Predicted Price:[/] [bold yellow]{price:N0} $[/]"
    : "[red]Prediction failed.[/]");