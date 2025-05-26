using MachineLearningDemo.Model;
using Microsoft.ML;

namespace MachineLearningDemo.ConsoleApp;

public class PredictionService
{
    private const string ModelPath = "data/model.zip";
    private readonly PredictionEngine<ModelInput, ModelOutput> _predictionEngine;

    public PredictionService()
    {
        var context = new MLContext();
        var model = context.Model.Load(ModelPath, out _);
        _predictionEngine = context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
    }

    public float Predict(string model, int year, int mileage)
    {
        var data = new ModelInput
        {
            Model = model,
            Year = year,
            Mileage = mileage
        };

        var result = _predictionEngine.Predict(data);

        return result.Score;
    }
}