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
		public string Padding { get; }

		public Convolutional2DTransposeLayer(int filters, (int, int) kernelSize, ActivationType activationType, string padding = "valid")
		{
			Filters = filters;
			KernelSize = kernelSize;
			ActivationType = activationType;
			Padding = padding;
		}

		public Tensors BuildLayer(Tensors output) => new Conv2DTranspose(new Conv2DArgs()
		{
			Filters = Filters,
			KernelSize = KernelSize,
			Activation = GetActivationByName(ActivationType),
			Padding = Padding,
			DilationRate = 0,
			DType = TF_DataType.TF_DOUBLE,
		}).Apply(output);

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
