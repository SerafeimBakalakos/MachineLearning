namespace MGroup.MachineLearning.Interfaces
{
	using System;
	using System.Collections.Generic;
	using System.Text;

	using MGroup.MachineLearning.Utilities;

	public interface ISurrogateModel2DTo2D
	{
		IReadOnlyList<string> ErrorNames { get; }

		/// <summary>
		/// Trains the surrogate model, tests its various parts and returns the corresponding errors.
		/// </summary>
		/// <param name="inputDataset">
		/// GetLength(0) must be the number of combined samples (training, test, evaluation).
		/// </param>
		/// <param name="outputDataset">
		/// GetLength(0) must be the number of combined samples (training, test, evaluation).
		/// </param>
		/// <param name="splitter">If null, the surrogate will decide how to split the datasets.</param>
		/// <returns></returns>
		Dictionary<string, double> TrainAndEvaluate(double[,] inputDataset, double[,] outputDataset, DatasetSplitter splitter);
	}
}
