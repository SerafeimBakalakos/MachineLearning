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

	using Xunit;

	[Collection("Run sequentially")]
	public static class CaeFffnnSurrogateTests
	{
		[Fact]
		public static void TestBiot()
		{
			(double[,] solutions, double[,] parameters) = ReadData();
			int numSamples = solutions.GetLength(0);
			Assert.Equal(parameters.GetLength(0), numSamples);
			int solutionSpaceDim = solutions.GetLength(1);
			int parametricSpaceDim = parameters.GetLength(1);

			var surrogateBuilder = new CaeFffnSurrogate.Builder();
			surrogateBuilder.TensorFlowSeed = 1234;
			var surrogate = surrogateBuilder.BuildSurrogate();
			surrogate.Initialize(solutionSpaceDim, parametricSpaceDim);
			surrogate.Train(solutions, parameters, numSamples);
		}

		private static (double[,] solutions, double[,] parameters) ReadData()
		{
			string solutionsPath = "C:\\Users\\Serafeim\\Desktop\\AISolve\\Solutions.npy";
			string parametersPath = "C:\\Users\\Serafeim\\Desktop\\AISolve\\Parameters.npy";

			double[,,] solutionVectors = np.Load<double[,,]>(solutionsPath); // [numSamples, 1, numDofs]
			double[,] parameters = np.Load<double[,]>(parametersPath); // [numSamples, numParameters]

			return (solutionVectors.RemoveEmptyDimension(1), parameters);
		}
	}
}
