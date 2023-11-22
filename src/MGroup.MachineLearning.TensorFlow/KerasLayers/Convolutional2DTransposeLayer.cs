using System;
using System.Collections.Generic;
using System.Text;

using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using Tensorflow.Operations.Initializers;

namespace MGroup.MachineLearning.TensorFlow.KerasLayers
{
	[Serializable]
	public class Convolutional2DTransposeLayer : INetworkLayer
	{
		public int Filters { get; }
		public (int, int) KernelSize { get; }
		public ActivationType ActivationType { get; }
		public (int, int)? Strides { get; }
		public int DilationRate { get; }
		public string Padding { get; }

		public Convolutional2DTransposeLayer(int filters, (int, int) kernelSize, ActivationType activationType, 
			(int, int)? strides = null, int dilationRate = 1, string padding = "valid")
		{
			Filters = filters;
			KernelSize = kernelSize;
			ActivationType = activationType;
			Strides = strides;
			DilationRate = dilationRate;
			Padding = padding;
		}

		public Tensors BuildLayer(Tensors output)
		{
			var args = new Conv2DArgs()
			{
				Filters = Filters,
				KernelSize = KernelSize,
				Activation = GetActivationByName(ActivationType),
				Strides = this.Strides ?? (1, 1),
				Padding = Padding,
				DilationRate = this.DilationRate,
				DType = TF_DataType.TF_DOUBLE,
			};
			var kerasLayer = new Conv2DTranspose(args);
			Tensors result = kerasLayer.Apply(output);
			return result;
		}

		private Activation GetActivationByName(ActivationType activation)
		{
			return activation switch
			{
				ActivationType.Linear => KerasApi.keras.activations.Linear,
				ActivationType.RelU => KerasApi.keras.activations.Relu,
				ActivationType.Sigmoid => KerasApi.keras.activations.Sigmoid,
				ActivationType.TanH => KerasApi.keras.activations.Tanh,
				ActivationType.SoftMax => KerasApi.keras.activations.Softmax,
				_ => throw new Exception($"Activation '{activation}' not found"),
			};
		}
	}
}
