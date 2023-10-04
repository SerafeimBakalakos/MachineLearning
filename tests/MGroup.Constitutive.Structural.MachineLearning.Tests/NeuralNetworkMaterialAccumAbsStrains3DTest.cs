using System;
using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
using MGroup.Constitutive.Structural.Continuum;
using MGroup.LinearAlgebra.Matrices;
using Xunit;
using System.Reflection;
using MGroup.MachineLearning;

namespace MGroup.Constitutive.Structural.MachineLearning.Tests
{
	[Collection("Run sequentially")]
	public static class NeuralNetworkMaterialAccumAbsStrains3DTest
	{
		[Fact]
		public static void RunTest()
		{
			// these files are used to generate an already trained FeedForwardNeuralNetwork which was created using strain-stress pairs from an ElasticMaterial3D(youngModulus:20, poissonRatio:0.2)
			string initialPath = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location).Split(new string[] { "\\bin" }, StringSplitOptions.None)[0];
			var folderName = "SavedFiles";
			var netPathName = "network_architecture_acc_strains";
			netPathName = Path.Combine(initialPath, folderName, netPathName);
			var weightsPathName = "trained_weights_acc_strains";
			weightsPathName = Path.Combine(initialPath, folderName, weightsPathName);
			var normalizationPathName = "normalization_acc_strains";
			normalizationPathName = Path.Combine(initialPath, folderName, normalizationPathName);

			var neuralNetwork = new FeedForwardNeuralNetwork();
			neuralNetwork.LoadNetwork(netPathName, weightsPathName, normalizationPathName);

			var neuralNetworkMaterial = new NeuralNetworkMaterialAccumAbsStrains3D(neuralNetwork, new double[0]);
			var elasticMaterial = new ElasticMaterial3D(20, 0.2);

			CheckNeuralNetworkMaterialAccuracy(neuralNetworkMaterial, elasticMaterial);
		}

		private static void CheckNeuralNetworkMaterialAccuracy(NeuralNetworkMaterialAccumAbsStrains3D neuralNetworkMaterial, ElasticMaterial3D elasticMaterial)
		{
			var increments = 5;
			var cases = 6;
			var stressesNeuralNetwork = new double[cases * increments, 6];
			var constitutiveNeuralNetwork = new Matrix[cases * increments];
			var stressesElastic = new double[cases * increments, 6];
			var constitutiveElastic = new Matrix[cases * increments];
			var strainCases = new double[6][] { new double[6]{ 0.001, 0.00, 0.00, 0.00, 0.00, 0.00 }, new double[6] { 0.00, 0.001, 0.00, 0.00, 0.00, 0.00 }, new double[6] { 0.00, 0.00, 0.001, 0.00, 0.00, 0.00 },
												new double[6]{ 0.00, 0.00, 0.00, 0.001, 0.00, 0.00 }, new double[6]{ 0.00, 0.00, 0.00, 0.00, 0.001, 0.00 }, new double[6]{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.001 }};
			for (int k = 0; k < strainCases.GetLength(0); k++)
			{
				for (int i = 0; i < increments; i++)
				{

					elasticMaterial.UpdateConstitutiveMatrixAndEvaluateResponse(strainCases[k]);
					neuralNetworkMaterial.UpdateConstitutiveMatrixAndEvaluateResponse(strainCases[k]);
					for (int j = 0; j < 6; j++)
					{
						stressesNeuralNetwork[k * increments + i, j] = neuralNetworkMaterial.Stresses[j];
						stressesElastic[k * increments + i, j] = elasticMaterial.Stresses[j];
					}
					constitutiveNeuralNetwork[k * increments + i] = (Matrix)neuralNetworkMaterial.ConstitutiveMatrix.Copy();
					constitutiveElastic[k * increments + i] = (Matrix)elasticMaterial.ConstitutiveMatrix.Copy();
					neuralNetworkMaterial.CreateState();
					elasticMaterial.CreateState();
				}
				neuralNetworkMaterial.ClearState();
				elasticMaterial.ClearState();
			}

			var stressDeviation = 0d;
			var constitutiveDeviation = 0d;
			for (int k = 0; k < strainCases.GetLength(0); k++)
			{
				for (int i = 0; i < increments; i++)
				{
					for (int j1 = 0; j1 < 6; j1++)
					{
						stressDeviation += Math.Pow((stressesNeuralNetwork[k * increments + i, j1] - stressesElastic[k * increments + i, j1]), 2);
						for (int j2 = 0; j2 < 6; j2++)
						{
							constitutiveDeviation += Math.Pow((constitutiveNeuralNetwork[k * increments + i][j1, j2] - constitutiveElastic[k * increments + i][j1, j2]), 2);
						}
					}
				}
			}

			stressDeviation = stressDeviation / (increments * strainCases.GetLength(0));
			constitutiveDeviation = constitutiveDeviation / (increments * strainCases.GetLength(0));

			Assert.True(stressDeviation < 1e-5 && constitutiveDeviation < 3e-1);
		}
	}
}
