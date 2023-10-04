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
	public class ConvolutionalNeuralNetworkTest
	{

		[Fact]
		public static void ConvolutionalNeuralNetworkWithAdam() => TestConvolutionalNeuralNetwork(new ConvolutionalNeuralNetwork(new NullNormalization(), new NullNormalization(),
				new TensorFlow.Keras.Optimizers.Adam(dataType: Tensorflow.TF_DataType.TF_DOUBLE),
				keras.losses.SparseCategoricalCrossentropy(from_logits: true), new INetworkLayer[]
				{
							new InputLayer(new int[]{28, 28, 1}),
							//new RescalingLayer(scale: 1/255),
							new Convolutional2DLayer(filters: 32, kernelSize: (5, 5), ActivationType.RelU),
							new MaxPooling2DLayer(strides: (2,2)),
							new Convolutional2DLayer(filters: 64, kernelSize: (3, 3), ActivationType.RelU),
							new MaxPooling2DLayer(strides: (2,2)),
							new FlattenLayer(),
							new DenseLayer(1024, ActivationType.RelU),
							//new DropoutLayer(0.5, seed: 1),
							new DenseLayer(10, ActivationType.SoftMax),
				},
				batchSize: 32, epochs: 50, seed: 1, classification: true));

		private static void TestConvolutionalNeuralNetwork(ConvolutionalNeuralNetwork neuralNetwork)
		{
			(double[,,,] trainX, double[,] trainY, double[,,,] testX, double[,] testY) = PrepareData();
			neuralNetwork.Train(trainX, trainY);
			var accuracy = neuralNetwork.ValidateNetwork(testX, testY);
			Assert.True(accuracy > 0.85);
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

		public static (double[,,,] trainX, double[,] trainY, double[,,,] testX, double[,] testY) PrepareData()
		{
			string initialPath = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location).Split(new string[] { "\\bin" }, StringSplitOptions.None)[0];
			var folderName = "SavedFiles";
			var trainXName = "mnist_trainX";
			trainXName = Path.Combine(initialPath, folderName, trainXName);
			folderName = "SavedFiles";
			var trainYName = "mnist_trainY";
			trainYName = Path.Combine(initialPath, folderName, trainYName);
			folderName = "SavedFiles";
			var testXName = "mnist_testX";
			testXName = Path.Combine(initialPath, folderName, testXName);
			folderName = "SavedFiles";
			var testYName = "mnist_testY";
			testYName = Path.Combine(initialPath, folderName, testYName);

			var trainX = new double[200, 28, 28, 1];
			var trainY = new double[200,1];
			var testX = new double[100, 28, 28, 1];
			var testY = new double[100,1];

			using (Stream stream = File.Open(trainXName, FileMode.Open))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				trainX = (double[,,,])binaryFormatter.Deserialize(stream);
			}

			using (Stream stream = File.Open(trainYName, FileMode.Open))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				trainY = (double[,])binaryFormatter.Deserialize(stream);
			}

			using (Stream stream = File.Open(testXName, FileMode.Open))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				testX = (double[,,,])binaryFormatter.Deserialize(stream);
			}

			using (Stream stream = File.Open(testYName, FileMode.Open))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				testY = (double[,])binaryFormatter.Deserialize(stream);
			}

			return (trainX, trainY, testX, testY);
		}
	}
}
