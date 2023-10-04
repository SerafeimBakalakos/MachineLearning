namespace MGroup.MachineLearning.TensorFlow.KerasLayers
{
	using System;
	using System.Collections.Generic;
	using System.Text;

	using Tensorflow;
	using Tensorflow.Keras.ArgsDefinition;
	using Tensorflow.Keras.Layers;

	public class ReshapeLayer : INetworkLayer
	{
		public int[] TargetShape { get; }

		public ReshapeLayer(int[] targetShape)
		{
			this.TargetShape = targetShape;
		}

		public Tensors BuildLayer(Tensors output) => new Reshape(new ReshapeArgs()
		{
			TargetShape = this.TargetShape,
		}).Apply(output);
	}
}
