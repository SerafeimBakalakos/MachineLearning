namespace MGroup.MachineLearning.TensorFlow.KerasLayers
{
	using Tensorflow;
	using Tensorflow.Keras.ArgsDefinition;
	using Tensorflow.Keras.Layers;

	public class ReshapeLayer : INetworkLayer
	{
		public ReshapeLayer(int[] targetShape)
		{
			this.TargetShape = targetShape;
		}

		public int[] TargetShape { get; }

		public Tensors BuildLayer(Tensors output) => new Reshape(new ReshapeArgs()
		{
			TargetShape = this.TargetShape,
		}).Apply(output);
	}
}
