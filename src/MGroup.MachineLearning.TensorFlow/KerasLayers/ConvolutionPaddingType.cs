namespace MGroup.MachineLearning.TensorFlow.KerasLayers
{
	using System;

	public enum ConvolutionPaddingType
	{
		/// <summary>
		/// No padding
		/// </summary>
		Valid,


		/// <summary>
		/// Padding with zeros evenly to the left/right or up/down of the input, such that output has the same height/width
		/// dimension as the input
		/// </summary>
		Same,

		/// <summary>
		/// Results in causal (dilated) convolutions, e.g. `output[t]` does not depend on `input[t+1:]`.
		/// Useful when modeling temporal data where the model should not violate the temporal order.
		/// </summary>
		/// <remarks>
		/// See [WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499).
		/// </remarks>
		Causal,
	}

	public static class ConvolutionPaddingTypeExtensions
	{
		public static string GetNameForTensorFlow(this ConvolutionPaddingType padding) => padding switch
		{
			ConvolutionPaddingType.Valid => "valid",
			ConvolutionPaddingType.Same => "same",
			ConvolutionPaddingType.Causal => "causal",
			_ => throw new ArgumentOutOfRangeException(nameof(padding), $"Unexpected padding type: {padding}")
		};
	}

}
