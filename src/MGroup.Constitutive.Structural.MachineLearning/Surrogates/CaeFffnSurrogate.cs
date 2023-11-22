namespace MGroup.Constitutive.Structural.MachineLearning.Surrogates
{
	using System;
	using System.Collections.Generic;
	using System.Diagnostics;
	using System.IO;
	using System.Text;
	using System.Text.RegularExpressions;
	using System.Timers;

	using DotNumerics.LinearAlgebra;

	using MGroup.MachineLearning.Interfaces;
	using MGroup.MachineLearning.Preprocessing;
	using MGroup.MachineLearning.TensorFlow.Keras.Optimizers;
	using MGroup.MachineLearning.TensorFlow.KerasLayers;
	using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
	using MGroup.MachineLearning.Utilities;

	using Tensorflow;
	using Tensorflow.Clustering;
	using Tensorflow.Keras.Losses;
	using Tensorflow.Operations.Initializers;

	public class CaeFffnSurrogate : ISurrogateModel2DTo2D
	{
		private const TF_DataType DataType = TF_DataType.TF_DOUBLE;

		private readonly int _caeBatchSize;
		private readonly int _caeNumEpochs;
		private readonly float _caeLearningRate;
		private readonly int _caeKernelSize;
		private readonly int _caeStrides;
		private readonly ConvolutionPaddingType _caePadding;

		/// <summary>
		/// Another convolutional layer (with linear activation) from the last hidden layer to the output space will be 
		/// automatically added to the end.
		/// </summary>
		private readonly int[] _decoderFiltersWithoutOutput = { 32, 64, 128 };
		private readonly int[] _encoderFilters = { 128, 64, 32, 16 };

		private readonly int _ffnnBatchSize = 20;
		private readonly int _ffnnNumEpochs = 3000;
		private readonly int _ffnnNumHiddenLayers = 6;
		private readonly int _ffnnHiddenLayerSize = 64;
		private readonly float _ffnnLearningRate = 1E-4f;
		private readonly int _latentSpaceSize = 8;
		private readonly DatasetSplitter _splitter;
		private readonly int? _tfSeed;
		private readonly Func<StreamWriter> _initOutputStream;

		private ConvolutionalAutoencoder _cae;
		private FeedForwardNeuralNetwork _ffnn;

		//Delete
		private ConvolutionalNeuralNetwork _encoder;

		public IReadOnlyList<string> ErrorNames => new string[] { "CAE error", "Surrogate error" };

		public CaeFffnSurrogate(int caeBatchSize, int caeNumEpochs, float caeLearningRate, int caeKernelSize, int caeStrides,
			ConvolutionPaddingType caePadding, int[] decoderFiltersWithoutOutput, int[] encoderFilters,
			int ffnnBatchSize, int ffnnNumEpochs, int ffnnNumHiddenLayers, int ffnnHiddenLayerSize, float ffnnLearningRate,
			int latentSpaceDim, DatasetSplitter splitter, int? tfSeed, Func<StreamWriter> initOutputStream)
		{
			#region DEBUG
			//TODO: remove the option to set stride altogether. This is not a traditional convolution
			if (caeStrides != 1)
			{
				throw new ArgumentException("CAE stride must be 1");
			}
			#endregion

			_caeBatchSize = caeBatchSize;
			_caeNumEpochs = caeNumEpochs;
			_caeLearningRate = caeLearningRate;
			_caeKernelSize = caeKernelSize;
			_caeStrides = caeStrides;
			_caePadding = caePadding;
			_decoderFiltersWithoutOutput = decoderFiltersWithoutOutput;
			_encoderFilters = encoderFilters;
			_ffnnBatchSize = ffnnBatchSize;
			_ffnnNumEpochs = ffnnNumEpochs;
			_ffnnNumHiddenLayers = ffnnNumHiddenLayers;
			_ffnnHiddenLayerSize = ffnnHiddenLayerSize;
			_ffnnLearningRate = ffnnLearningRate;
			_latentSpaceSize = latentSpaceDim;
			_splitter = splitter;
			_tfSeed = tfSeed;

			_initOutputStream = initOutputStream;
		}

		public Dictionary<string, double> TrainAndEvaluate(double[,] inputDataset, double[,] outputDataset, 
			DatasetSplitter? splitter)
		{
			if (splitter == null)
			{
				splitter = _splitter;
			}

			int numTotalSamples = inputDataset.GetLength(0);
			if (outputDataset.GetLength(0) != numTotalSamples)
			{
				throw new ArgumentException(
					"The first dimension of the input and ouput dataset arrays must be the same and equal to the number of " +
					$"samples, but were {numTotalSamples} and {outputDataset.GetLength(0)} respectively instead");
			}

			// Define the networks
			double[,] parameters = inputDataset;
			double[,] solutionVectors = outputDataset;
			int parameterSpaceDim = parameters.GetLength(1);
			int solutionSpaceDim = solutionVectors.GetLength(1);
			BuildAutoEncoder(solutionSpaceDim);
			BuildFfnn(parameterSpaceDim);

			// Split the input and output into train-test sets
			splitter.SetupSplittingRules(numTotalSamples);
			(double[,] trainSolutions, double[,] testSolutions, _) = splitter.SplitDataset(solutionVectors);
			(double[,] trainParameters, double[,] testParameters, _) = splitter.SplitDataset(parameters);
			double[,,,] caeTrainX = trainSolutions.AddEmptyDimensions(false, true, true, false);
			double[,,,] caeTestX = testSolutions.AddEmptyDimensions(false, true, true, false);

			// Train
			TrainCae(caeTrainX);
			TrainFfnn(caeTrainX, trainParameters, testParameters);

			// Evaluate
			var result = new Dictionary<string, double>();
			result["CAE error"] = TestCae(caeTestX);
			result["Surrogate error"] = TestFullSurrogate(testSolutions, testParameters);
			return result;
		}

		private void BuildAutoEncoder(int solutionSpaceDim)
		{
			// Encoder layers
			var encoderLayers = new List<INetworkLayer>();
			encoderLayers.Add(new InputLayer(new int[] { 1, 1, solutionSpaceDim}));
			for (int i = 0; i < _encoderFilters.Length; ++i)
			{
				encoderLayers.Add(new Convolutional2DLayer(_encoderFilters[i], (_caeKernelSize, 1), ActivationType.RelU,
					strides: (_caeStrides, 1), padding: _caePadding.GetNameForTensorFlow(), dilationRate:1));
			}
			encoderLayers.Add(new FlattenLayer());
			encoderLayers.Add(new DenseLayer(_latentSpaceSize, ActivationType.Linear)); // Output layer. Activation: f(x) = x

			// Decoder layers
			var decoderLayers = new List<INetworkLayer>();
			//decoderLayers.Add(new InputLayer(new int[] { _latentSpaceSize })); // CAE class does this itself
			int denseLayerSize = _decoderFiltersWithoutOutput[0] / 2; // In python code, it did not divide over 2. 
			decoderLayers.Add(new DenseLayer(denseLayerSize, ActivationType.RelU)); // This is 16 in the paper
			decoderLayers.Add(new ReshapeLayer(new int[] { 1, 1, denseLayerSize }));
			for (int i = 0; i < _decoderFiltersWithoutOutput.Length; ++i)
			{
				decoderLayers.Add(new Convolutional2DTransposeLayer(_decoderFiltersWithoutOutput[i], (_caeKernelSize, 1),
					ActivationType.RelU, strides: (_caeStrides, 1), padding: _caePadding.GetNameForTensorFlow(), dilationRate:1));
			}

			// Output layer. Activation: f(x) = x
			decoderLayers.Add(new Convolutional2DTransposeLayer(solutionSpaceDim, (_caeKernelSize, 1), 
				ActivationType.Linear, strides: (_caeStrides, 1), padding: _caePadding.GetNameForTensorFlow(), dilationRate: 1)); 

			INormalization normalizationX = new NullNormalization();
			var optimizer = new Adam(dataType: DataType, learning_rate: _caeLearningRate);
			ILossFunc lossFunction = KerasApi.keras.losses.MeanSquaredError();
			_cae = new ConvolutionalAutoencoder(normalizationX, optimizer, lossFunction, encoderLayers.ToArray(),
				decoderLayers.ToArray(), _caeNumEpochs, _caeBatchSize, _tfSeed, shuffleTrainingData: true);
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
				_ffnnNumEpochs, _ffnnBatchSize, _tfSeed, shuffleTrainingData:true);
		}

		private void TrainCae(double[,,,] trainSolutions)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Training convolutional autoencoder:");
			watch.Start();
			_cae.Train(trainSolutions); // python code also used the test data as validation set here.
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine();

			writer.Close();
		}

		private void TrainFfnn(double[,,,] trainSolutions, double[,] trainParameters, double[,] testParameters)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Prepare FFNN output data using convolutional encoder");
			watch.Start();

			// RemoveEmptyDimensions() did not work correctly for [160, 1, 1, 8]. It produced [160,1]. Must test all add/remove empty dimension methods
			//double[,] ffnnTrainY = _cae
			//	.MapFullToReduced(trainSolutions)
			//	.RemoveEmptyDimensions(1, 2); 

			double[,] ffnnTrainY = _cae.MapFull4DToReduced2D(trainSolutions);
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);

			writer.WriteLine("Training feed forward neural network:");
			watch.Restart();
			_ffnn.Train(trainParameters, ffnnTrainY); // python code also used the encoded test data as validation set here.
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);

			writer.Close();
		}

		private double TestCae(double[,,,] testSolutions)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Testing convolutional autoencoder:");
			watch.Start();
			double[,,,] caePredictions = _cae.EvaluateResponses(testSolutions);
			double error = ErrorMetrics.CalculateMeanNorm2Error(testSolutions, caePredictions);
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine($"Mean error = 1/numSamples * sumOverSamples( norm2(CAE(u) - u) / norm2(u) = {error}");
			writer.WriteLine();

			writer.Close();
			return error;
		}

		private double TestFullSurrogate(double[,] testSolutions, double[,] testParameters)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Testing convolutional autoencoder:");
			watch.Start();
			double[,] ffnnPredictions = _ffnn.EvaluateResponses(testParameters);
			double[,,,] surrogatePredictions = _cae.MapReduced2DToFull4D(ffnnPredictions);
			double error = ErrorMetrics.CalculateMeanNorm2Error(
				testSolutions.AddEmptyDimensions(false, true, true, false), surrogatePredictions);
			
			//double[,,,] surrogatePredictions = _cae.MapReducedToFull(
			//	ffnnPredictions.AddEmptyDimensions(false, true, false, true));
			//double error = ErrorMetrics.CalculateMeanNorm2Error(
			//	testSolutions.AddEmptyDimensions(false, true, false, true), surrogatePredictions);

			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine($"Mean error = 1/numSamples * sumOverSamples( norm2(surrogate(theta) - u) / norm2(u) = {error}");

			writer.Close();
			return error;
		}

		public class Builder
		{
			public Builder()
			{
				//GetOutputStream = () =>
				//{
				//	var writer = new StreamWriter(Console.OpenStandardOutput());
				//	writer.AutoFlush = true;
				//	Console.SetOut(writer);
				//	return writer;
				//};
				GetOutputStream = () => new DebugTextWriter();

				Splitter = new DatasetSplitter();
				Splitter.MinTestSetPercentage = 0.2;
				Splitter.MinValidationSetPercentage = 0.0;
				Splitter.SetOrderToContiguous(DataSubsetType.Training, DataSubsetType.Test);
			}

			public int CaeBatchSize { get; set; } = 10;

			public int CaeNumEpochs { get; set; } = 40;

			public float CaeLearningRate { get; set; } = 5E-4f;

			public int CaeKernelSize { get; set; } = 5;

			public int CaeStrides { get; set; } = 1;

			public ConvolutionPaddingType CaePadding { get; set; } = ConvolutionPaddingType.Same;

			/// <summary>
			/// Another convolutional layer (with linear activation) from the last hidden layer to the output space will be 
			/// automatically added to the end.
			/// </summary>
			public int[] DecoderFiltersWithoutOutput { get; set; } = { 32, 64, 128 };

			public int[] EncoderFilters { get; set; } = { 128, 64, 32, 16 };

			public int FfnnBatchSize { get; set; } = 20;

			public int FfnnNumEpochs { get; set; } = 3000;

			public int FfnnNumHiddenLayers { get; set; } = 6;

			public int FfnnHiddenLayerSize { get; set; } = 64;

			public float FfnnLearningRate { get; set; } = 1E-4f;

			public Func<StreamWriter> GetOutputStream { get; set; }

			public int LatentSpaceDim { get; set; } = 8;

			public DatasetSplitter Splitter { get; set; }

			public int? TensorFlowSeed { get; set; } = null;

			public CaeFffnSurrogate BuildSurrogate()
			{
				return new CaeFffnSurrogate(CaeBatchSize, CaeNumEpochs, CaeLearningRate, CaeKernelSize, CaeStrides, CaePadding, 
					DecoderFiltersWithoutOutput, EncoderFilters, FfnnBatchSize, FfnnNumEpochs, FfnnNumHiddenLayers, 
					FfnnHiddenLayerSize, FfnnLearningRate, LatentSpaceDim, Splitter, TensorFlowSeed, GetOutputStream);
			}
		}
	}
}
