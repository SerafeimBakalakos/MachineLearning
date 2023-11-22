namespace MGroup.Constitutive.Structural.MachineLearning.Tests
{
	using System;
	using System.Collections.Generic;
	using System.Diagnostics;
	using System.Linq;
	using System.Text;
	using System.Threading.Tasks;

	using MGroup.Constitutive.Structural.MachineLearning.Surrogates;
	using MGroup.Constitutive.Structural.MachineLearning.Tests.Utilities;
	using MGroup.MachineLearning.TensorFlow.KerasLayers;
	using MGroup.MachineLearning.Utilities;

	using NumSharp;

	using Tensorflow;

	using Xunit;

	[Collection("Run sequentially")]
	public static class CaeFffnnSurrogateTests
	{
		[Fact]
		public static void TestDecoder2DStatistics()
		{
			string outputPath = "C:\\Users\\Serafeim\\Desktop\\AISolve\\Surrogate testing\\decoder2D_errors_NET.txt";
			(double[,] solutions, double[,] parameters, double[,] latentSpace) = ReadData();
			var decoder = new Surrogates.Decoder2D(latentSpaceSize: 8, caeKernelSize: 5, caePadding: ConvolutionPaddingType.Same);

			var evaluator = new SurrogateModelEvaluator(100, outputPath);
			evaluator.RunExperiments(decoder, latentSpace, solutions);
			(int sampleSize, double mean, double stdev) = evaluator.PerformStatisticAnalysisOnErrors(outputPath);
			Debug.WriteLine($"Errors: sample size = {sampleSize}, mean = {mean}, stdev = {stdev}");
		}

		[Fact]
		public static void TestEncoder1DStatistics()
		{
			string outputPath = "C:\\Users\\Serafeim\\Desktop\\AISolve\\Surrogate testing\\encoder1D_errors_NET.txt";
			(double[,] solutions, double[,] parameters, double[,] latentSpace) = ReadData();
			var encoder = new Surrogates.Encoder1D(latentSpaceSize:8, caeKernelSize:5, caePadding: ConvolutionPaddingType.Same);

			var evaluator = new SurrogateModelEvaluator(100, outputPath);
			evaluator.RunExperiments(encoder, solutions, latentSpace);
			(int sampleSize, double mean, double stdev) = evaluator.PerformStatisticAnalysisOnErrors(outputPath);
			Debug.WriteLine($"Errors: sample size = {sampleSize}, mean = {mean}, stdev = {stdev}");
		}

		[Fact]
		public static void TestEncoder2DStatistics()
		{
			string errorType = "surrogate error";
			string outputPath = "C:\\Users\\Serafeim\\Desktop\\AISolve\\Surrogate testing\\encoder2D_errors_NET.txt";
			(double[,] solutions, double[,] parameters, double[,] latentSpace) = ReadData();
			var encoder = new Surrogates.Encoder2D(latentSpaceSize: 8, caeKernelSize: 5, caePadding: ConvolutionPaddingType.Same);

			var evaluator = new SurrogateModelEvaluator(100, outputPath);
			evaluator.RunExperiments(encoder, solutions, latentSpace);
			(int sampleSize, double mean, double stdev) = evaluator.PerformStatisticAnalysisOnErrors(outputPath);
			Debug.WriteLine($"Errors: sample size = {sampleSize}, mean = {mean}, stdev = {stdev}");
		}

		[Fact]
		public static void TestFfnnStatistics()
		{
			string outputPath = "C:\\Users\\Serafeim\\Desktop\\AISolve\\Surrogate testing\\ffnn_errors_NET.txt";
			//string outputPath1 = "C:\\Users\\Serafeim\\Desktop\\AISolve\\Surrogate testing\\ffnn_errors_python.txt";
			(double[,] solutions, double[,] parameters, double[,] latentSpace) = ReadData();
			var ffnn = new FFNN(8);

			var evaluator = new SurrogateModelEvaluator(1000, outputPath);
			evaluator.RunExperiments(ffnn, parameters, latentSpace);
			(int sampleSize, double mean, double stdev) = evaluator.PerformStatisticAnalysisOnErrors(outputPath);
			Debug.WriteLine($"Errors: sample size = {sampleSize}, mean = {mean}, stdev = {stdev}");
		}

		[Fact]
		public static void TestFullSurrogateStatistics()
		{
			string outputPath = "C:\\Users\\Serafeim\\Desktop\\AISolve\\Surrogate testing\\surrogate_errors_NET.txt";
			(double[,] solutions, double[,] parameters, double[,] latentSpace) = ReadData();
			var surrogateBuilder = new CaeFffnSurrogate.Builder();
			//surrogateBuilder.CaeNumEpochs = 10;
			surrogateBuilder.CaeNumEpochs = 40;
			CaeFffnSurrogate surrogate = surrogateBuilder.BuildSurrogate();

			var evaluator = new SurrogateModelEvaluator(100, outputPath);
			evaluator.RunExperiments(surrogate, parameters, solutions);

			//(int sampleSize, double mean, double stdev) = evaluator.PerformStatisticAnalysisOnErrors(outputPath);
			//Debug.WriteLine($"CAE errors: sample size = {sampleSize}, mean = {mean}, stdev = {stdev}");
			//(sampleSize, mean, stdev) = evaluator.PerformStatisticAnalysisOnErrors(outputPath);
			//Debug.WriteLine($"Full surrogate errors: sample size = {sampleSize}, mean = {mean}, stdev = {stdev}");
		}

		[Fact]
		public static void TestBiot()
		{
			(double[,] solutions, double[,] parameters, double[,] latentSpace) = ReadData();
			int numSamples = solutions.GetLength(0);
			Assert.Equal(parameters.GetLength(0), numSamples);
			int solutionSpaceDim = solutions.GetLength(1);
			int parametricSpaceDim = parameters.GetLength(1);

			int seed = 1234;
			//tf.set_random_seed(seed);
			var surrogateBuilder = new CaeFffnSurrogate.Builder();
			surrogateBuilder.TensorFlowSeed = seed;
			var surrogate = surrogateBuilder.BuildSurrogate();
			Dictionary<string, double> errors = surrogate.TrainAndEvaluate(parameters, solutions, surrogateBuilder.Splitter);
		}

		private static (double[,] solutions, double[,] parameters, double[,] latentSpace) ReadData()
		{
			string folder = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.FullName 
				+ "\\MGroup.Constitutive.Structural.MachineLearning.Tests\\InputFiles\\CaeFfnnSurrogate\\";
			string solutionsPath = folder + "solutions.npy";
			string parametersPath = folder + "parameters.npy";
			string latentSpacePath = folder + "latentSpace.npy";

			double[,,] solutionVectors = np.Load<double[,,]>(solutionsPath); // [numSamples, 1, numDofs]
			double[,] parameters = np.Load<double[,]>(parametersPath); // [numSamples, numParameters]
			double[,] latentSpace = np.Load<double[,]>(latentSpacePath); // [numSamples, latentSpaceSize]

			return (solutionVectors.RemoveEmptyDimension(1), parameters, latentSpace);
		}
	}
}
