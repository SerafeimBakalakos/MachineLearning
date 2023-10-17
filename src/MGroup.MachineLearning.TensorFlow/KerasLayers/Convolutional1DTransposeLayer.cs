namespace MGroup.MachineLearning.TensorFlow.KerasLayers
{
	using System;

	using Tensorflow;
	using Tensorflow.Keras.ArgsDefinition;
	using Tensorflow.Keras.Layers;

	[Serializable]
	public class Convolutional1DTransposeLayer : INetworkLayer
	{
		public Convolutional1DTransposeLayer(int filters, int kernelSize, ActivationType activationType, 
			int rank = 1, int dilationRate = 1, int strides = 1, ConvolutionPaddingType padding = ConvolutionPaddingType.Valid,
			bool isConvolutionWindowAlongHeight = true)
		{
			Filters = filters;
			KernelSize = kernelSize;
			ActivationType = activationType;
			Rank = rank;
			DilationRate = dilationRate;
			Strides = strides;

			if (!(padding == ConvolutionPaddingType.Valid || padding == ConvolutionPaddingType.Same))
			{
				throw new ArgumentException("Padding type for 1D transpose convolution must be 'valid' or 'same'");
			}
			Padding = padding;

			IsConvolutionWindowAlongHeight = isConvolutionWindowAlongHeight;
		}

		public int Filters { get; }

		public int KernelSize { get; }

		public ActivationType ActivationType { get; }

		public int Rank { get; }

		public int DilationRate { get; }

		public int Strides { get; }

		public ConvolutionPaddingType Padding { get; }

		public bool IsConvolutionWindowAlongHeight { get; }

		public Tensors BuildLayer(Tensors inputs)
		{
			if (IsConvolutionWindowAlongHeight)
			{
				return new Conv2DTranspose(new Conv2DArgs()
				{
					//Rank = 2,
					Filters = this.Filters,
					KernelSize = (this.KernelSize, 1),
					Activation = ActivationType.GetActivationFunc(),
					Strides = (this.Strides, 1),
					DilationRate = (this.DilationRate, 1),
					DType = TF_DataType.TF_DOUBLE,
					Padding = this.Padding.ToString(),
				}).Apply(inputs);
			}
			else
			{
				return new Conv2DTranspose(new Conv2DArgs()
				{
					//Rank = 2,
					Filters = this.Filters,
					KernelSize = (1, this.KernelSize),
					Activation = ActivationType.GetActivationFunc(),
					Strides = (1, this.Strides),
					DilationRate = (1, this.DilationRate),
					DType = TF_DataType.TF_DOUBLE,
					Padding = this.Padding.ToString(),
				}).Apply(inputs);
			}
		}
	}
}
