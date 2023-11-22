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

	public class Decoder2D : ISurrogateModel2DTo2D
	{
		private const TF_DataType DataType = TF_DataType.TF_DOUBLE;

		private readonly int _caeBatchSize;
		private readonly int _caeNumEpochs;
		private readonly float _caeLearningRate;
		private readonly int _caeKernelSize;
		private readonly int _caeStrides;
		private readonly ConvolutionPaddingType _caePadding;
		private readonly int[] _decoderFiltersWithoutOutput;
		private readonly int _latentSpaceSize = 8;
		private readonly int? _tfSeed;
		private readonly Func<StreamWriter> _initOutputStream;

		private ConvolutionalNeuralNetwork _decoder;

		public Decoder2D(int latentSpaceSize, int caeBatchSize = 10, int caeNumEpochs = 40, float caeLearningRate = 5E-4f,
			int caeKernelSize = 5, int caeStrides = 1, ConvolutionPaddingType caePadding = ConvolutionPaddingType.Same,
			int[] decoderFiltersWithoutOutput = null, int? tfSeed = null)
		{
			_caeBatchSize = caeBatchSize;
			_caeNumEpochs = caeNumEpochs;
			_caeLearningRate = caeLearningRate;
			_caeKernelSize = caeKernelSize;
			_caeStrides = caeStrides;
			_caePadding = caePadding;
			_decoderFiltersWithoutOutput = decoderFiltersWithoutOutput;
			_latentSpaceSize = latentSpaceSize;
			_tfSeed = tfSeed;

			if (decoderFiltersWithoutOutput == null)
			{
				decoderFiltersWithoutOutput = new int[] { 32, 64, 128 };
			}

			_decoderFiltersWithoutOutput = decoderFiltersWithoutOutput;

			_initOutputStream = () => new DebugTextWriter();
		}

		public IReadOnlyList<string> ErrorNames => new string[] { "Surrogate error" };

		public Dictionary<string, double> TrainAndEvaluate(double[,] latentSpaceDataset, double[,] solutionSpaceDataset, 
			DatasetSplitter? splitter)
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
					$"The 2nd dimension of the input dataset must be equal to the size of the latent space {_latentSpaceSize}");
			}

			splitter.SetupSplittingRules(numTotalSamples);
			(double[,] trainSolutions, double[,] testSolutions, _) = splitter.SplitDataset(solutionSpaceDataset);
			(double[,] trainLatent, double[,] testLatent, _) = splitter.SplitDataset(latentSpaceDataset);

			BuildDecoder(solutionSpaceSize);
			Train(trainSolutions, trainLatent);
			double error = Evaluate(testSolutions, testLatent);

			return new Dictionary<string, double>()
			{
				{ "Surrogate error", error }
			};
		}

		private void BuildDecoder(int solutionSpaceDim)
		{
			// Decoder layers
			var decoderLayers = new List<INetworkLayer>();
			decoderLayers.Add(new InputLayer(new int[] { _latentSpaceSize }));
			int denseLayerSize = _decoderFiltersWithoutOutput[0] / 2; // In python code, it did not divide over 2. 
			decoderLayers.Add(new DenseLayer(denseLayerSize, ActivationType.RelU)); // This is 16 in the paper
			decoderLayers.Add(new ReshapeLayer(new int[] { 1, 1, denseLayerSize }));
			for (int i = 0; i < _decoderFiltersWithoutOutput.Length; ++i)
			{
				decoderLayers.Add(new Convolutional2DTransposeLayer(_decoderFiltersWithoutOutput[i], (_caeKernelSize, 1),
					ActivationType.RelU, strides: (_caeStrides, 1), padding: _caePadding.GetNameForTensorFlow(), dilationRate: 1));
			}
			decoderLayers.Add(new Convolutional2DTransposeLayer(solutionSpaceDim, (_caeKernelSize, 1),
				ActivationType.Linear, strides: (_caeStrides, 1), padding: _caePadding.GetNameForTensorFlow(), dilationRate: 1));

			INormalization normalizationX = new NullNormalization();
			INormalization normalizationY = new NullNormalization();
			var optimizer = new Adam(dataType: DataType, learning_rate: _caeLearningRate);
			ILossFunc lossFunction = KerasApi.keras.losses.MeanSquaredError();
			_decoder = new ConvolutionalNeuralNetwork(normalizationX, normalizationY, optimizer, lossFunction,
				decoderLayers.ToArray(), _caeNumEpochs, _caeBatchSize, _tfSeed, shuffleTrainingData: true);
		}

		private void Train(double[,] trainSolutions, double[,] trainLatent)
		{
			double[,] trainX = trainLatent;
			//double[,,] trainX = trainLatent.AddEmptyDimensions(false, false, true);
			double[,,,] trainY = trainSolutions.AddEmptyDimensions(false, true, true, false);

			//testLatent set is different than python
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Training feed forward neural network:");
			watch.Restart();
			_decoder.Train(trainX, trainY);
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.Close();
		}

		private double Evaluate(double[,] testSolutions, double[,] testLatent)
		{
			//double[,,,] testY = testSolutions.AddEmptyDimensions(false, false, true, true);
			double[,,,] testY = testSolutions.AddEmptyDimensions(false, true, true, false);

			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Testing Decoder 2D:");
			watch.Start();
			double[,,,] predictions = _decoder.EvaluateResponses(testLatent);
			double error = ErrorMetrics.CalculateMeanNorm2Error(testY, predictions);

			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine($"Mean norm2 error = 1/numSamples * sumOverSamples( norm2(FFNN(theta) - w) / norm2(w) = {error}");
			writer.Close();

			return error;
		}
	}
}
