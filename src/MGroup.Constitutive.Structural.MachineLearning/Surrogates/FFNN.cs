namespace MGroup.Constitutive.Structural.MachineLearning.Surrogates
{
	using System;
	using System.Collections.Generic;
	using System.Diagnostics;
	using System.IO;
	using System.Security.Cryptography;
	using System.Text;

	using MGroup.MachineLearning.Interfaces;
	using MGroup.MachineLearning.Preprocessing;
	using MGroup.MachineLearning.TensorFlow.Keras.Optimizers;
	using MGroup.MachineLearning.TensorFlow.KerasLayers;
	using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
	using MGroup.MachineLearning.Utilities;

	using Tensorflow;
	using Tensorflow.Keras.Losses;

	public class FFNN : ISurrogateModel2DTo2D
	{
		private const TF_DataType DataType = TF_DataType.TF_DOUBLE;

		private readonly int _ffnnBatchSize;
		private readonly int _ffnnNumEpochs;
		private readonly int _ffnnNumHiddenLayers;
		private readonly int _ffnnHiddenLayerSize;
		private readonly float _ffnnLearningRate;
		private readonly int _latentSpaceSize = 8;
		private readonly int? _tfSeed;
		private readonly Func<StreamWriter> _initOutputStream;

		private FeedForwardNeuralNetwork _ffnn;

		public FFNN(int latentSpaceSize, int ffnnBatchSize = 20, int ffnnNumEpochs = 3000, int ffnnNumHiddenLayers = 6, 
			int ffnnHiddenLayerSize = 64, float ffnnLearningRate = 1E-4f, int? tfSeed = null)
		{
			_ffnnBatchSize = ffnnBatchSize;
			_ffnnNumEpochs = ffnnNumEpochs;
			_ffnnNumHiddenLayers = ffnnNumHiddenLayers;
			_ffnnHiddenLayerSize = ffnnHiddenLayerSize;
			_ffnnLearningRate = ffnnLearningRate;
			_latentSpaceSize = latentSpaceSize;
			_tfSeed = tfSeed;

			_initOutputStream = () => new DebugTextWriter();
		}

		public IReadOnlyList<string> ErrorNames => new string[] { "Surrogate error" };

		public Dictionary<string, double> TrainAndEvaluate(double[,] parameterSpaceDataset, double[,] latentSpaceDataset, 
			DatasetSplitter splitter)
		{
			if (splitter == null)
			{
				splitter = new DatasetSplitter();
				splitter.MinTestSetPercentage = 0.2;
				splitter.MinValidationSetPercentage = 0.0;
			}

			int numTotalSamples = parameterSpaceDataset.GetLength(0);
			int parameterSpaceSize = parameterSpaceDataset.GetLength(1);
			if (latentSpaceDataset.GetLength(0) != numTotalSamples)
			{
				throw new ArgumentException(
					"The first dimension of the input and output datasets must be equal to the number of samples");
			}

			if (latentSpaceDataset.GetLength(1) != _latentSpaceSize)
			{
				throw new ArgumentException(
					$"The 2nd dimension of the output dataset must be equal to the size of the latent space {_latentSpaceSize}");
			}

			splitter.SetupSplittingRules(numTotalSamples);
			(double[,] trainParameters, double[,] testParameters, _) = splitter.SplitDataset(parameterSpaceDataset);
			(double[,] trainLatent, double[,] testLatent, _) = splitter.SplitDataset(latentSpaceDataset);

			BuildFfnn(parameterSpaceSize);
			Train(trainParameters, trainLatent);
			double error = Evaluate(testParameters, testLatent);

			return new Dictionary<string, double>()
			{
				{ "Surrogate error", error }
			};
		}

		private void BuildFfnn(int parameterSpaceDim)
		{
			var layers = new List<INetworkLayer>();
			layers.Add(new InputLayer(new int[] { parameterSpaceDim }));
			for (int i = 0; i < _ffnnNumHiddenLayers; i++)
			{
				layers.Add(new DenseLayer(_ffnnHiddenLayerSize, ActivationType.RelU));
			};
			layers.Add(new DenseLayer(_latentSpaceSize, ActivationType.Linear)); // Output layer. Activation: f(x) = x

			INormalization normalizationX = new NullNormalization();
			INormalization normalizationY = new NullNormalization();
			var optimizer = new Adam(dataType: DataType, learning_rate: _ffnnLearningRate);
			ILossFunc lossFunction = KerasApi.keras.losses.MeanSquaredError();
			_ffnn = new FeedForwardNeuralNetwork(normalizationX, normalizationY, optimizer, lossFunction, layers.ToArray(),
				_ffnnNumEpochs, _ffnnBatchSize, _tfSeed, shuffleTrainingData: true);
		}

		private void Train(double[,] trainParameters, double[,] trainLatent)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Training feed forward neural network:");
			watch.Restart();
			_ffnn.Train(trainParameters, trainLatent); // python code also used the encoded test data as validation set here.
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.Close();
		}

		private double Evaluate(double[,] testParameters, double[,] testLatent)
		{
			//testLatent set is different than python
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Testing FFNN:");
			watch.Start();
			double[,] predictions = _ffnn.EvaluateResponses(testParameters);
			double error = ErrorMetrics.CalculateMeanNorm2Error(testLatent, predictions);

			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine($"Mean norm2 error = 1/numSamples * sumOverSamples( norm2(FFNN(theta) - w) / norm2(w) = {error}");
			writer.Close();

			return error;
		}
	}
}
