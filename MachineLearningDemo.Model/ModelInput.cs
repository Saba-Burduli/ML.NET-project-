using Microsoft.ML.Data;

namespace MachineLearningDemo.Model;

public class ModelInput
{
    [LoadColumn(0)] public string Model { get; set; }
    [LoadColumn(1)] public float Year { get; set; }
    [LoadColumn(2)] public float Mileage { get; set; }
    [LoadColumn(3)] public float Price { get; set; }
}