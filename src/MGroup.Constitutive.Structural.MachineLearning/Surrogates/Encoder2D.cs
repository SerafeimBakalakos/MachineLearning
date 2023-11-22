namespace MGroup.Constitutive.Structural.MachineLearning.Surrogates
{
	using System;
	using System.Collections.Generic;
	using System.Diagnostics;
	using System.IO;
	using System.Text;
	using MGroup.MachineLearning.Preprocessing;
	using MGroup.MachineLearning.TensorFlow.KerasLayers;
	using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
	using MGroup.MachineLearning.Utilities;

	using Tensorflow.Keras.Losses;

	using Tensorflow;
	using MGroup.MachineLearning.TensorFlow.Keras.Optimizers;
	using MGroup.MachineLearning.Interfaces;

	public class Encoder2D : ISurrogateModel2DTo2D
	{
		private const TF_DataType DataType = TF_DataType.TF_DOUBLE;

		private readonly int _caeBatchSize;
		private readonly int _caeNumEpochs;
		private readonly float _caeLearningRate;
		private readonly int _caeKernelSize;
		private readonly int _caeStrides;
		private readonly ConvolutionPaddingType _caePadding;
		private readonly int[] _encoderFilters;
		private readonly int _latentSpaceSize = 8;
		private readonly int? _tfSeed;
		private readonly Func<StreamWriter> _initOutputStream;

		private ConvolutionalNeuralNetwork _encoder;

		public Encoder2D(int latentSpaceSize, int caeBatchSize = 10, int caeNumEpochs = 40, float caeLearningRate = 5E-4f,
			int caeKernelSize = 5, int caeStrides = 1, ConvolutionPaddingType caePadding = ConvolutionPaddingType.Same,
			int[] encoderFilters = null, int? tfSeed = null)
		{
			_caeBatchSize = caeBatchSize;
			_caeNumEpochs = caeNumEpochs;
			_caeLearningRate = caeLearningRate;
			_caeKernelSize = caeKernelSize;
			_caeStrides = caeStrides;
			_caePadding = caePadding;
			_encoderFilters = encoderFilters;
			_latentSpaceSize = latentSpaceSize;
			_tfSeed = tfSeed;

			if (encoderFilters == null)
			{
				encoderFilters = new int[] { 128, 64, 32, 16 };
			}

			_encoderFilters = encoderFilters;

			_initOutputStream = () => new DebugTextWriter();
		}

		public IReadOnlyList<string> ErrorNames => new string[] { "Surrogate error" };

		public Dictionary<string, double> TrainAndEvaluate(double[,] solutionSpaceDataset, double[,] latentSpaceDataset, 
			DatasetSplitter splitter)
		{
			if (splitter == null)
			{
				splitter = new DatasetSplitter();
				splitter.MinTestSetPercentage = 0.2;
				splitter.MinValidationSetPercentage = 0.0;
			}

			int numTotalSamples = solutionSpaceDataset.GetLength(0);
			int solutionSpaceSize = solutionSpaceDataset.GetLength(1);
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
			(double[,] trainSolutions, double[,] testSolutions, _) = splitter.SplitDataset(solutionSpaceDataset);
			(double[,] trainLatent, double[,] testLatent, _) = splitter.SplitDataset(latentSpaceDataset);

			BuildEncoder(solutionSpaceSize);
			Train(trainSolutions, trainLatent);
			double error = Evaluate(testSolutions, testLatent);

			return new Dictionary<string, double>()
			{
				{ "Surrogate error", error }
			};
		}

		private void BuildEncoder(int solutionSpaceDim)
		{
			var encoderLayers = new List<INetworkLayer>();
			encoderLayers.Add(new InputLayer(new int[] { 1, 1, solutionSpaceDim})); // This was not needed in the original python code
			for (int i = 0; i < _encoderFilters.Length; ++i)
			{
				encoderLayers.Add(new Convolutional2DLayer(_encoderFilters[i], (_caeKernelSize, 1), ActivationType.RelU,
					dilationRate:1, padding: _caePadding.GetNameForTensorFlow()));
			}
			encoderLayers.Add(new FlattenLayer());
			encoderLayers.Add(new DenseLayer(_latentSpaceSize, ActivationType.Linear)); // Output layer. Activation: f(x) = x

			INormalization normalizationX = new NullNormalization();
			INormalization normalizationY = new NullNormalization();
			var optimizer = new Adam(dataType: DataType, learning_rate: _caeLearningRate);
			ILossFunc lossFunction = KerasApi.keras.losses.MeanSquaredError();
			_encoder = new ConvolutionalNeuralNetwork(normalizationX, normalizationY, optimizer, lossFunction,
				encoderLayers.ToArray(), _caeNumEpochs, _caeBatchSize, _tfSeed, shuffleTrainingData: true);
		}

		private void Train(double[,] trainSolutions, double[,] trainLatent)
		{
			//double[,,,] trainX = trainSolutions.AddEmptyDimensions(false, false, true, true);
			double[,,,] trainX = trainSolutions.AddEmptyDimensions(false, true, true, false);
			double[,] trainY = trainLatent;

			//testLatent set is different than python
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Training feed forward neural network:");
			watch.Restart();
			_encoder.Train(trainX, trainY); // python code also used the encoded test data as validation set here.
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.Close();
		}

		private double Evaluate(double[,] testSolutions, double[,] testLatent)
		{
			//double[,,,] testX = testSolutions.AddEmptyDimensions(false, false, true, true);
			double[,,,] testX = testSolutions.AddEmptyDimensions(false, true, true, false);

			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Testing Encoder2D:");
			watch.Start();
			double[,] predictions = _encoder.EvaluateResponses(testX);
			double error = ErrorMetrics.CalculateMeanNorm2Error(testLatent, predictions);

			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine($"Mean norm2 error = 1/numSamples * sumOverSamples( norm2(FFNN(theta) - w) / norm2(w) = {error}");
			writer.Close();

			return error;
		}
	}
}
