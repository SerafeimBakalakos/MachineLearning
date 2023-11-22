namespace MGroup.MachineLearning.TensorFlow.KerasLayers
{
	using System;

	using Tensorflow;
	using Tensorflow.Keras.ArgsDefinition;
	using Tensorflow.Keras.Layers;

	[Serializable]
	public class Convolutional1DLayer : INetworkLayer
	{
		public Convolutional1DLayer(int filters, int kernelSize, ActivationType activationType, 
			int rank = 1, int dilationRate = 1, int strides = 1, ConvolutionPaddingType padding = ConvolutionPaddingType.Valid)
		{
			Filters = filters;
			KernelSize = kernelSize;
			ActivationType = activationType;
			Rank = rank;
			DilationRate = dilationRate;
			Strides = strides;
			Padding = padding;
		}

		public int Filters { get; }

		public int KernelSize { get; }

		public ActivationType ActivationType { get; }

		public int Rank { get; }

		public int DilationRate { get; }

		public int Strides { get; }

		public ConvolutionPaddingType Padding { get; }

		public Tensors BuildLayer(Tensors inputs)
		{
			var args = new Conv1DArgs()
			{
				Rank = this.Rank,
				Filters = this.Filters,
				KernelSize = this.KernelSize,
				Activation = ActivationType.GetActivationFunc(),
				DilationRate = this.DilationRate,
				Strides = this.Strides,
				DType = TF_DataType.TF_DOUBLE,
				Padding = this.Padding.GetNameForTensorFlow(),
			};
			var kerasLayer = new Conv1D(args);
			Tensors result = kerasLayer.Apply(inputs);
			return result;
		}
	}
}
