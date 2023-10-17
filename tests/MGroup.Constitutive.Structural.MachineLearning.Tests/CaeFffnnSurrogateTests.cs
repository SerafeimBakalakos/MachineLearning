namespace MGroup.Constitutive.Structural.MachineLearning.Tests
{
	using System;
	using System.Collections.Generic;
	using System.Linq;
	using System.Text;
	using System.Threading.Tasks;

	using MGroup.Constitutive.Structural.MachineLearning.Surrogates;
	using MGroup.MachineLearning.Utilities;

	using NumSharp;

	using Tensorflow;

	using Xunit;

	[Collection("Run sequentially")]
	public static class CaeFffnnSurrogateTests
	{
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
			surrogate.Initialize(solutionSpaceDim, parametricSpaceDim);
			//surrogate.Train(solutions, parameters, numSamples);
			surrogate.TrainNetworksIndependently(solutions, parameters, latentSpace, numSamples);
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
