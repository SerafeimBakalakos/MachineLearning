using System;
using Xunit;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;
using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
using MGroup.MachineLearning.Preprocessing;
using MGroup.MachineLearning.TensorFlow.KerasLayers;
using Tensorflow;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using System.Reflection;

namespace MGroup.MachineLearning.Tests
{
	[Collection("Run sequentially")]
	public class AutoencoderTest
	{

		[Fact]
		public static void AutoencoderWithAdam() => TestAutoencoder(new Autoencoder(new NullNormalization(),
				new TensorFlow.Keras.Optimizers.Adam(dataType: Tensorflow.TF_DataType.TF_DOUBLE),
				keras.losses.MeanSquaredError(), 
				new INetworkLayer[]
				{
							new InputLayer(new int[]{784}),
							new DenseLayer(64, ActivationType.RelU),
				},
				new INetworkLayer[]
				{
							new DenseLayer(784, ActivationType.Sigmoid),
				},
				batchSize: 32, epochs: 100, seed: 1, classification: false));

		private static void TestAutoencoder(Autoencoder neuralNetwork)
		{
			(double[,] trainX, double[,] testX) = PrepareData();
			neuralNetwork.Train(trainX);
			var loss = neuralNetwork.ValidateNetwork(testX);
			//var reducedX = neuralNetwork.MapFullToReduced(trainX);
			Assert.True(loss < 0.05);
			//CheckResponseGradientAccuracy(testGradient, gradients);
		}

		//private static void CheckResponseGradientAccuracy(double[][,] dataGradient, double[][,] gradient)
		//{
		//	var norm = new double[gradient.GetLength(0)];
		//	var normTotal = 0d;
		//	for (int k = 0; k < gradient.GetLength(0); k++)
		//	{
		//		var deviation = new double[gradient[k].GetLength(0), gradient[k].GetLength(1)];
		//		for (int i = 0; i < gradient[k].GetLength(0); i++)
		//		{
		//			for (int j = 0; j < gradient[k].GetLength(1); j++)
		//			{
		//				deviation[i, j] = (dataGradient[k][i, j] - gradient[k][i, j]);
		//				norm[k] += Math.Pow(deviation[i, j], 2);
		//			}
		//		}
		//		normTotal += norm[k];
		//	}
		//	normTotal = normTotal / gradient.GetLength(0);

		//	Assert.True(normTotal < 0.3, $"Gradient norm was above threshold (norm value: {norm})");
		//}

		public static (double[,] trainX, double[,] testX) PrepareData()
		{
			string initialPath = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location).Split(new string[] { "\\bin" }, StringSplitOptions.None)[0];
			var folderName = "SavedFiles";
			var trainXName = "mnist_trainX";
			trainXName = Path.Combine(initialPath, folderName, trainXName);
			folderName = "SavedFiles";
			var testXName = "mnist_testX";
			testXName = Path.Combine(initialPath, folderName, testXName);

			var trainX = new double[200, 28, 28, 1];
			var testX = new double[100, 28, 28, 1];

			using (Stream stream = File.Open(trainXName, FileMode.Open))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				trainX = (double[,,,])binaryFormatter.Deserialize(stream);
			}

			using (Stream stream = File.Open(testXName, FileMode.Open))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				testX = (double[,,,])binaryFormatter.Deserialize(stream);
			}

			var trainX2 = new double[200, 784];
			var testX2 = new double[100, 784];
			for (int k = 0; k < trainX.GetLength(0); k++)
			{
				var count = 0;
				for (int i = 0; i < trainX.GetLength(1); i++)
				{
					for (int j = 0; j < trainX.GetLength(2); j++)
					{
						trainX2[k, count] = trainX[k, i, j, 0];
						count = count + 1;
					}
				}
			}
			for (int k = 0; k < testX.GetLength(0); k++)
			{
				var count = 0;
				for (int i = 0; i < testX.GetLength(1); i++)
				{
					for (int j = 0; j < testX.GetLength(2); j++)
					{
						testX2[k, count] = testX[k, i, j, 0];
						count = count + 1;
					}
				}
			}

			return (trainX2, testX2);
		}
	}
}
