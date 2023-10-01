using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras;
using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;

namespace MGroup.MachineLearning.TensorFlow.KerasLayers
{
	[Serializable]
	public class UpSampling2DLayer : INetworkLayer
	{
		public (int, int)? Size { get; }

		public UpSampling2DLayer((int, int)? Size = null)
		{
			this.Size = Size ?? (2, 2);
		}

		public Tensors BuildLayer(Tensors output) => new UpSampling2D(new UpSampling2DArgs()
		{
			Size = this.Size,
			DType = TF_DataType.TF_DOUBLE
		}).Apply(output);

	}
}
