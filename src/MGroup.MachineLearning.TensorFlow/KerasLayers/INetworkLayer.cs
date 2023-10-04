using System;

using Tensorflow;
using Tensorflow.Keras;

namespace MGroup.MachineLearning.TensorFlow.KerasLayers
{
	public interface INetworkLayer
	{
		Tensors BuildLayer(Tensors output);
	}

	public enum ActivationType
	{
		NotSet,
		Linear,
		RelU,
		Sigmoid,
		TanH,
		SoftMax
	}

	public static class ActivationTypeExtensions
	{
		public static Activation GetActivationFunc(this ActivationType activation)
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
