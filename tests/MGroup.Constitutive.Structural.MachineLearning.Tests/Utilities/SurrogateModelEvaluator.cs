namespace MGroup.Constitutive.Structural.MachineLearning.Tests.Utilities
{
	using System;
	using System.Collections.Generic;
	using System.Linq;
	using System.Text;
	using System.Threading.Tasks;

	using MGroup.MachineLearning.Interfaces;
	using MGroup.MachineLearning.Utilities;

	//TODO: Plot histogram of residuals
	public class SurrogateModelEvaluator
	{
		private readonly int _numExperiments;
		private readonly string _outputPathPrefix;

		public SurrogateModelEvaluator(int numExperiments, string outputPathPrefix)
		{
			_numExperiments = numExperiments;
			_outputPathPrefix = outputPathPrefix;
		}

		public (int sampleSize, double mean, double stdev) PerformStatisticAnalysisOnErrors(string path)
		{
			var errors = new List<double>();
			using (var reader = new StreamReader(path))
			{
				string line = reader.ReadLine();

				while (line != null)
				{
					bool isNumber = double.TryParse(line, out double number);
					if (isNumber)
					{
						errors.Add(number);
					}
					line = reader.ReadLine();
				}
			}

			double mean = CalcMean(errors);
			double stdev = CalcStandardDeviation(errors, mean);
			return (errors.Count, mean, stdev);
		}

		public void RunExperiments(ISurrogateModel2DTo2D surrogate, double[,] inputDataset, double[,] outputDataset)
		{
			foreach (string errorName in surrogate.ErrorNames)
			{
				string path = WriteOutputPath(errorName);
				using (var writer = new StreamWriter(path, true))
				{
					writer.AutoFlush = true;
					writer.WriteLine();
					writer.WriteLine("*** Date: " + DateTime.Now + " ***");
				}
			}

			for (int i = 0; i < _numExperiments; i++)
			{
				var splitter = new DatasetSplitter();
				splitter.MinTestSetPercentage = 0.2;
				splitter.MinValidationSetPercentage = 0.0;
				splitter.SetOrderToContiguous(DataSubsetType.Training, DataSubsetType.Test);

				Dictionary<string, double> errors = surrogate.TrainAndEvaluate(inputDataset, outputDataset, splitter);
				foreach (string errorName in errors.Keys)
				{
					string path = WriteOutputPath(errorName);
					using (var writer = new StreamWriter(path, true))
					{
						writer.AutoFlush = true;
						writer.WriteLine(errors[errorName]);
					}
				}
			}
		}

		private double CalcMean(List<double> numbers)
		{
			double sum = 0.0;
			foreach (double num in numbers)
			{
				sum += num;
			}
			return sum / numbers.Count;
		}

		private double CalcStandardDeviation(List<double> numbers, double mean)
		{
			double sum = 0.0;
			foreach (double num in numbers)
			{
				sum += (num - mean) * (num - mean);
			}
			return Math.Sqrt(sum / numbers.Count);
		}

		private string WriteOutputPath(string errorName)
		{
			// There may be more than one "."
			string[] words = _outputPathPrefix.Split('.');

			// Only the last one is before the extension
			string extension = words[words.Length - 1];

			// The rest should be readded
			var result = new StringBuilder();
			result.Append(words[0]);
			for (int i = 1; i < words.Length - 1; ++i)
			{
				result.Append('.');
				result.Append(words[i]);
			}

			// Add the name of the error to the filename
			result.Append('_');
			result.Append(errorName);

			// Add the extension
			result.Append('.');
			result.Append(extension);

			return result.ToString();
		}
	}
}
