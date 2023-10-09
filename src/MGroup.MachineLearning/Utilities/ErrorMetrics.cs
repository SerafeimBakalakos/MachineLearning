namespace MGroup.MachineLearning.Utilities
{
	using System;
	using System.Collections.Generic;
	using System.Text;

	public static class ErrorMetrics
	{
		/// <summary>
		/// The property Array.GetLength(0) of the input data <paramref name="expected"/> and <paramref name="predicted"/> must
		/// be equal to the number of samples/predictions.
		/// </summary>
		/// <param name="expected"><pa</param>
		/// <param name="predicted"></param>
		/// <returns></returns>
		public static double CalculateMeanError(double[,] expected, double[,] predicted)
		{
			int numSamples = expected.GetLength(0);
			int dimension = expected.GetLength(1);
			if (predicted.GetLength(0) != numSamples)
			{
				throw new ArgumentException($"Received {numSamples)} expected samples, " +
					$"but {predicted.GetLength(0)} corresponding predicted ones");
			}
			if (predicted.GetLength(1) != dimension)
			{
				throw new ArgumentException($"Each expected sample has dimensions [1, {dimension}], " +
					$"but each corresponding predicted sample has dimensions [1, {predicted.GetLength(1)}]");
			}

			double meanError = 0.0;
			for (int s = 0; s < numSamples; s++)
			{
				double norm = 0.0;
				double normDiff = 0.0;
				for (int i = 0; i < dimension; i++)
				{
					norm += expected[s, i] * expected[s, i];
					double diff = expected[s, i] - predicted[s, i];
					normDiff += diff * diff;
				}

				norm = Math.Sqrt(norm);
				normDiff = Math.Sqrt(normDiff);
				meanError += normDiff / norm;
			}

			meanError /= numSamples;
			return meanError;
		}

		/// <summary>
		/// The property Array.GetLength(0) of the input data <paramref name="expected"/> and <paramref name="predicted"/> must
		/// be equal to the number of samples/predictions.
		/// </summary>
		/// <param name="expected"><pa</param>
		/// <param name="predicted"></param>
		/// <returns></returns>
		public static double CalculateMeanError(double[,,] expected, double[,,] predicted)
		{
			int numSamples = expected.GetLength(0);
			int dim1 = expected.GetLength(1);
			int dim2 = expected.GetLength(2);
			if (predicted.GetLength(0) != numSamples)
			{
				throw new ArgumentException($"Received {numSamples} expected samples, " +
					$"but {predicted.GetLength(0)} corresponding predicted ones");
			}
			if (predicted.GetLength(1) != dim1 || predicted.GetLength(2) != dim2)
			{
				throw new ArgumentException($"Each expected sample has dimensions [1, {dim1}, {dim2}], " +
					"but each corresponding predicted sample has dimensions " +
					$"[1, {predicted.GetLength(1)}, {predicted.GetLength(2)}]");
			}

			double meanError = 0.0;
			for (int s = 0; s < numSamples; s++)
			{
				double norm = 0.0;
				double normDiff = 0.0;
				for (int i = 0; i < dim1; i++)
				{
					for (int j = 0; j < dim2; j++)
					{
						norm += expected[s, i, j] * expected[s, i, j];
						double diff = expected[s, i, j] - predicted[s, i, j];
						normDiff += diff * diff;
					}
				}

				norm = Math.Sqrt(norm);
				normDiff = Math.Sqrt(normDiff);
				meanError += normDiff / norm;
			}

			meanError /= numSamples;
			return meanError;
		}

		/// <summary>
		/// The property Array.GetLength(0) of the input data <paramref name="expected"/> and <paramref name="predicted"/> must
		/// be equal to the number of samples/predictions.
		/// </summary>
		/// <param name="expected"><pa</param>
		/// <param name="predicted"></param>
		/// <returns></returns>
		public static double CalculateMeanError(double[,,,] expected, double[,,,] predicted)
		{
			int numSamples = expected.GetLength(0);
			int dim1 = expected.GetLength(1);
			int dim2 = expected.GetLength(2);
			int dim3 = expected.GetLength(3);
			if (predicted.GetLength(0) != numSamples)
			{
				throw new ArgumentException($"Received {numSamples} expected samples, " +
					$"but {predicted.GetLength(0)} corresponding predicted ones");
			}
			if (predicted.GetLength(1) != dim1 || predicted.GetLength(2) != dim2 || predicted.GetLength(3) != dim3)
			{
				throw new ArgumentException($"Each expected sample has dimensions [1, {dim1}, {dim2}, {dim3}], " +
					$"but each corresponding predicted sample has dimensions " +
					$"[1, {predicted.GetLength(1)}, {predicted.GetLength(2)}, {predicted.GetLength(3)}]");
			}

			double meanError = 0.0;
			for (int s = 0; s < numSamples; s++)
			{
				double norm = 0.0;
				double normDiff = 0.0;
				for (int i = 0; i < dim1; i++)
				{
					for (int j = 0; j < dim2; j++)
					{
						for (int k = 0; k < dim3; k++)
						{
							norm += expected[s, i, j, k] * expected[s, i, j, k];
							double diff = expected[s, i, j, k] - predicted[s, i, j, k];
							normDiff += diff * diff;
						}
					}
				}

				norm = Math.Sqrt(norm);
				normDiff = Math.Sqrt(normDiff);
				meanError += normDiff / norm;
			}

			meanError /= numSamples;
			return meanError;
		}
	}
}
