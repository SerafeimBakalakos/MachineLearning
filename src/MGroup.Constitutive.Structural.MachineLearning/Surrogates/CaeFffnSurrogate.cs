namespace MGroup.Constitutive.Structural.MachineLearning.Surrogates
{
	using System;
	using System.Collections.Generic;
	using System.Text;

	using Tensorflow;

	using MGroup.MachineLearning.Preprocessing;
	using MGroup.MachineLearning.TensorFlow.Keras.Optimizers;
	using MGroup.MachineLearning.TensorFlow.KerasLayers;
	using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
	using Tensorflow.Keras.Losses;
	using Tensorflow.Operations.Initializers;
	using DotNumerics.LinearAlgebra;

	using System.Diagnostics;

	using MGroup.MachineLearning.Utilities;
	using System.IO;
	using Tensorflow.Clustering;
	using System.Timers;
	using System.Text.RegularExpressions;
	using OneOf.Types;

	public class CaeFffnSurrogate
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
		private readonly int[] _decoderFiltersWithoutOutput = { 128, 64, 32, 16 };
		private readonly int[] _encoderFilters = { 32, 64, 128 };

		private readonly int _ffnnBatchSize = 20;
		private readonly int _ffnnNumEpochs = 3000;
		private readonly int _ffnnNumHiddenLayers = 6;
		private readonly int _ffnnHiddenLayerSize = 64;
		private readonly float _ffnnLearningRate = 1E-4f;
		private readonly int _latentSpaceDim = 8;
		private readonly int? _tfSeed;
		private readonly Func<StreamWriter> _initOutputStream;

		private ConvolutionalAutoencoder _cae;
		private FeedForwardNeuralNetwork _ffnn;

		//Delete
		private ConvolutionalNeuralNetwork _encoder;

		public CaeFffnSurrogate(int caeBatchSize, int caeNumEpochs, float caeLearningRate, int caeKernelSize, int caeStrides,
			ConvolutionPaddingType caePadding, int[] decoderFiltersWithoutOutput, int[] encoderFilters,
			int ffnnBatchSize, int ffnnNumEpochs, int ffnnNumHiddenLayers, int ffnnHiddenLayerSize, float ffnnLearningRate,
			int latentSpaceDim, int? tfSeed, Func<StreamWriter> initOutputStream)
		{
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
			_latentSpaceDim = latentSpaceDim;
			_tfSeed = tfSeed;

			_initOutputStream = initOutputStream;
		}

		public void Initialize(int solutionSpaceDim, int parameterSpaceDim)
		{
			BuildAutoEncoder(solutionSpaceDim);
			BuildFfnn(parameterSpaceDim);
			BuildEncoder(solutionSpaceDim);
		}

		public void Train(double[,] solutionVectors, double[,] parameters, int numTotalSamples, double testSetPercentage = 0.2)
		{
			if (solutionVectors.GetLength(0) != numTotalSamples)
			{
				throw new ArgumentException("The first dimension of the solution vectors input array must be equal to the number" +
					$" of samples {numTotalSamples}, but was {solutionVectors.GetLength(0)} instead");
			}
			if (parameters.GetLength(0) != numTotalSamples)
			{
				throw new ArgumentException("The first dimension of the paramters input array must be equal to the number" +
					$" of samples {numTotalSamples}, but was {parameters.GetLength(0)} instead");
			}

			var splitter = new DatasetSplitter();
			splitter.MinTestSetPercentage = testSetPercentage;
			splitter.MinValidationSetPercentage = 0.0;
			splitter.SetOrderToContiguous(DataSubsetType.Training, DataSubsetType.Test);
			splitter.SetupSplittingRules(numTotalSamples);
			(double[,] trainSolutions, double[,] testSolutions, _) = splitter.SplitDataset(solutionVectors);
			(double[,] trainParameters, double[,] testParameters, _) = splitter.SplitDataset(parameters);

			TrainCae(trainSolutions, testSolutions);
			TestCae(testSolutions);
			TrainFfnn(trainSolutions, testSolutions, trainParameters, testParameters);
			TestFullSurrogate(testSolutions, testParameters);
		}

		//Delete
		public void TrainNetworksIndependently(double[,] solutionVectors, double[,] parameters, double[,] latentSpace, 
			int numTotalSamples, double testSetPercentage = 0.2)
		{
			if (solutionVectors.GetLength(0) != numTotalSamples)
			{
				throw new ArgumentException("The first dimension of the solution vectors input array must be equal to the number" +
					$" of samples {numTotalSamples}, but was {solutionVectors.GetLength(0)} instead");
			}
			if (parameters.GetLength(0) != numTotalSamples)
			{
				throw new ArgumentException("The first dimension of the paramters input array must be equal to the number" +
					$" of samples {numTotalSamples}, but was {parameters.GetLength(0)} instead");
			}

			var splitter = new DatasetSplitter();
			splitter.MinTestSetPercentage = testSetPercentage;
			splitter.MinValidationSetPercentage = 0.0;
			splitter.SetOrderToContiguous(DataSubsetType.Training, DataSubsetType.Test);
			splitter.SetupSplittingRules(numTotalSamples);
			(double[,] trainSolutions, double[,] testSolutions, _) = splitter.SplitDataset(solutionVectors);
			(double[,] trainParameters, double[,] testParameters, _) = splitter.SplitDataset(parameters);
			(double[,] trainLatent, double[,] testLatent, _) = splitter.SplitDataset(latentSpace);

			TrainOnlyFfnn(trainParameters, testParameters, trainLatent, testLatent);
			//TrainOnlyEncoder(trainSolutions, testSolutions, trainLatent, testLatent);
		}

		//Delete
		private void BuildEncoder(int solutionSpaceDim)
		{
			// Encoder layers
			var encoderLayers = new List<INetworkLayer>();
			encoderLayers.Add(new InputLayer(new int[] { 1, solutionSpaceDim })); // This was not needed in the original python code
			for (int i = 0; i < _encoderFilters.Length; ++i)
			{
				encoderLayers.Add(new Convolutional1DLayer(_encoderFilters[i], _caeKernelSize, ActivationType.RelU,
					strides: _caeStrides, padding: _caePadding));
			}
			encoderLayers.Add(new FlattenLayer()); // not sure if needed. The input is 1D and there is no such layer in the paper.
			encoderLayers.Add(new DenseLayer(_latentSpaceDim, ActivationType.Linear)); // Output layer. Activation: f(x) = x

			INormalization normalizationX = new NullNormalization();
			INormalization normalizationY = new NullNormalization();
			var optimizer = new Adam(dataType: DataType, learning_rate: _caeLearningRate);
			ILossFunc lossFunction = KerasApi.keras.losses.MeanSquaredError();
			_encoder = new ConvolutionalNeuralNetwork(normalizationX, normalizationY, optimizer, lossFunction, encoderLayers.ToArray(),
				_caeNumEpochs, batchSize:_caeBatchSize, seed:_tfSeed, shuffleTrainingData: true);
		}

		//Delete
		private void BuildDecoder(int solutionSpaceDim)
		{
			// Decoder layers
			var decoderLayers = new List<INetworkLayer>();
			decoderLayers.Add(new InputLayer(new int[] { _latentSpaceDim }));
			decoderLayers.Add(new DenseLayer(32, ActivationType.RelU)); // This is 16 in the paper
			decoderLayers.Add(new ReshapeLayer(new int[] { 1, 32 }));
			for (int i = 0; i < _decoderFiltersWithoutOutput.Length; ++i)
			{
				decoderLayers.Add(new Convolutional1DTransposeLayer(_decoderFiltersWithoutOutput[i], _caeKernelSize,
					ActivationType.RelU, strides: _caeStrides, padding: _caePadding));
			}
			decoderLayers.Add(new Convolutional1DTransposeLayer(solutionSpaceDim, _caeKernelSize,
				ActivationType.Linear, strides: _caeStrides, padding: _caePadding)); // Output layer. Activation: f(x) = x

			INormalization normalizationX = new NullNormalization();
			INormalization normalizationY = new NullNormalization();
			var optimizer = new Adam(dataType: DataType, learning_rate: _caeLearningRate);
			ILossFunc lossFunction = KerasApi.keras.losses.MeanSquaredError();
			_encoder = new ConvolutionalNeuralNetwork(normalizationX, normalizationY, optimizer, lossFunction, decoderLayers.ToArray(),
				_caeNumEpochs, batchSize: _caeBatchSize, seed: _tfSeed, shuffleTrainingData: true);
		}

		private void BuildAutoEncoder(int solutionSpaceDim)
		{
			// Encoder layers
			var encoderLayers = new List<INetworkLayer>();
			encoderLayers.Add(new InputLayer(new int[] { 1, solutionSpaceDim})); // This was not needed in the original python code
			for (int i = 0; i < _encoderFilters.Length; ++i)
			{
				encoderLayers.Add(new Convolutional1DLayer(_encoderFilters[i], _caeKernelSize, ActivationType.RelU,
					strides: _caeStrides, padding: _caePadding));
			}
			encoderLayers.Add(new FlattenLayer()); // not sure if needed. The input is 1D and there is no such layer in the paper.
			encoderLayers.Add(new DenseLayer(_latentSpaceDim, ActivationType.Linear)); // Output layer. Activation: f(x) = x

			// Decoder layers
			var decoderLayers = new List<INetworkLayer>();
			decoderLayers.Add(new InputLayer(new int[] { _latentSpaceDim }));
			decoderLayers.Add(new DenseLayer(32, ActivationType.RelU)); // This is 16 in the paper
			decoderLayers.Add(new ReshapeLayer(new int[] { 1, 32 }));
			for (int i = 0; i < _decoderFiltersWithoutOutput.Length; ++i)
			{
				decoderLayers.Add(new Convolutional1DTransposeLayer(_decoderFiltersWithoutOutput[i], _caeKernelSize, 
					ActivationType.RelU, strides: _caeStrides, padding: _caePadding));
			}
			decoderLayers.Add(new Convolutional1DTransposeLayer(solutionSpaceDim, _caeKernelSize, 
				ActivationType.Linear, strides: _caeStrides, padding: _caePadding)); // Output layer. Activation: f(x) = x

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
			layers.Add(new DenseLayer(_latentSpaceDim, ActivationType.Linear)); // Output layer. Activation: f(x) = x

			INormalization normalizationX = new NullNormalization();
			INormalization normalizationY = new NullNormalization();
			var optimizer = new Adam(dataType: DataType, learning_rate: _ffnnLearningRate);
			ILossFunc lossFunction = KerasApi.keras.losses.MeanSquaredError();
			_ffnn = new FeedForwardNeuralNetwork(normalizationX, normalizationY, optimizer, lossFunction, layers.ToArray(),
				_ffnnNumEpochs, _ffnnBatchSize, _tfSeed, shuffleTrainingData:true);
		}

		private void TrainCae(double[,] trainSolutions, double[,] testSolutions)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Training convolutional autoencoder:");
			watch.Start();
			double[,,,] validationSolutions = testSolutions.AddEmptyDimensions(false, true, false, true);
			_cae.Train(trainSolutions.AddEmptyDimensions(false, true, false, true)); // python code also used the test data as validation set here.
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine();

			writer.Close();
		}

		private void TrainFfnn(double[,] trainSolutions, double[,] testSolutions,
			double[,] trainParameters, double[,] testParameters)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Prepare FFNN output data using convolutional encoder");
			watch.Start();
			double[,] ffnnTrainY = _cae
				.MapFullToReduced(trainSolutions.AddEmptyDimensions(false, true, false, true))
				.RemoveEmptyDimensions(1, 3);
			double[,] ffnnTestY = _cae
				.MapFullToReduced(testSolutions.AddEmptyDimensions(false, true, false, true))
				.RemoveEmptyDimensions(1, 3);
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);

			writer.WriteLine("Training feed forward neural network:");
			watch.Restart();
			_ffnn.Train(trainParameters, ffnnTrainY); // python code also used the encoded test data as validation set here.
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);

			writer.Close();
		}

		//Delete
		private void TrainOnlyFfnn(double[,] trainParameters, double[,] testParameters, double[,] trainLatent, double[,] testLatent)
		{
			//testLatent set is different than python
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Training feed forward neural network:");
			watch.Restart();
			_ffnn.Train(trainParameters, trainLatent); // python code also used the encoded test data as validation set here.
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);

			writer.WriteLine("Testing FFNN:");
			watch.Restart();
			double[,] ffnnPredictions = _ffnn.EvaluateResponses(testParameters);
			double error = ErrorMetrics.CalculateMeanNorm2Error(testLatent, ffnnPredictions);

			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine($"Mean error = 1/numSamples * sumOverSamples( norm2(FFNN(theta) - w) / norm2(w) = {error}");

			writer.Close();
		}

		//Delete
		private void TrainOnlyEncoder(double[,] trainSolutions, double[,] testSolutions, double[,] trainLatent, double[,] testLatent)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Training convolutional autoencoder:");
			watch.Start();
			double[,,,] validationSolutions = testSolutions.AddEmptyDimensions(false, true, false, true);
			_encoder.Train(
				trainSolutions.AddEmptyDimensions(false, true, false, true),
				trainLatent
				); // python code also used the test data as validation set here.
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine();

			writer.Close();
		}
		private void TestCae(double[,] testSolutions)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Testing convolutional autoencoder:");
			watch.Start();
			double[,,,] caePredictions = _cae.EvaluateResponses(testSolutions.AddEmptyDimensions(false, true, false, true));
			double error = ErrorMetrics.CalculateMeanNorm2Error(
				testSolutions.AddEmptyDimensions(false, true, false, true), caePredictions);
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine($"Mean error = 1/numSamples * sumOverSamples( norm2(CAE(u) - u) / norm2(u) = {error}");
			writer.WriteLine();

			writer.Close();
		}

		private void TestFullSurrogate(double[,] testSolutions, double[,] testParameters)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Testing convolutional autoencoder:");
			watch.Start();
			double[,] ffnnPredictions = _ffnn.EvaluateResponses(testParameters);
			double[,,,] surrogatePredictions = _cae.MapReducedToFull(
				ffnnPredictions.AddEmptyDimensions(false, true, false, true));
			double error = ErrorMetrics.CalculateMeanNorm2Error(
				testSolutions.AddEmptyDimensions(false, true, false, true), surrogatePredictions);

			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine($"Mean error = 1/numSamples * sumOverSamples( norm2(surrogate(theta) - u) / norm2(u) = {error}");

			writer.Close();
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

			public int? TensorFlowSeed { get; set; } = null;

			public CaeFffnSurrogate BuildSurrogate()
			{
				return new CaeFffnSurrogate(CaeBatchSize, CaeNumEpochs, CaeLearningRate, CaeKernelSize, CaeStrides, CaePadding, 
					DecoderFiltersWithoutOutput, EncoderFilters, FfnnBatchSize, FfnnNumEpochs, FfnnNumHiddenLayers, 
					FfnnHiddenLayerSize, FfnnLearningRate, LatentSpaceDim, TensorFlowSeed, GetOutputStream);
			}
		}
	}
}
