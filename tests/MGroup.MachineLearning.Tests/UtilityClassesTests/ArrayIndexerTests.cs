namespace MGroup.MachineLearning.Tests.UtilityClassesTests
{
	using System;
	using System.Collections.Generic;
	using System.Linq;
	using System.Text;
	using System.Threading.Tasks;

	using MGroup.MachineLearning.Utilities;

	using Xunit;

	public class ArrayIndexerTests
	{
		[Theory]
		[InlineData(true)]
		[InlineData(false)]
		public static void TestIndexer1D(bool fullArray)
		{
			int dim0 = 10;
			int[] lengths;
			int[] lowerBounds;

			if (fullArray)
			{
				lengths = new int[] { dim0 };
				lowerBounds = null;
			}
			else
			{
				lengths = new int[] { dim0 - 1 };
				lowerBounds = new int[] { 1 };
			}

			int seed = 0;
			var rng = new Random(seed);
			var arrayExpected = new int[dim0];
			for (int i = 0; i < arrayExpected.Length; i++)
			{
				arrayExpected[i] = rng.Next();
			}

			var arrayComputed = new int[arrayExpected.Length];
			var indexer = new ArrayIndexer(lengths);
			do
			{
				int val = indexer.GetValueAtCurrentIndex<int>(arrayExpected);
				indexer.SetValueAtCurrentIndex(arrayComputed, val);
			} while (indexer.MoveNext());

			Assert.Equal(arrayExpected, arrayComputed);
		}

		[Theory]
		[InlineData(true)]
		[InlineData(false)]
		public static void TestIndexer2D(bool fullArray)
		{
			int dim0 = 8;
			int dim1 = 12;
			int[] lengths;
			int[] lowerBounds;

			if (fullArray)
			{
				lengths = new int[] { dim0, dim1 };
				lowerBounds = null;
			}
			else
			{
				lengths = new int[] { dim0 - 1, dim1 - 2 };
				lowerBounds = new int[] { 0, 1 };
			}

			int seed = 0;
			var rng = new Random(seed);
			var arrayExpected = new int[dim0, dim1];
			for (int i = 0; i < dim0; i++)
			{
				for (int j = 0; j < dim1; j++)
				{
					arrayExpected[i, j] = rng.Next();
				}
			}

			var arrayComputed = new int[dim0, dim1];
			var indexer = new ArrayIndexer(lengths, lowerBounds);
			do
			{
				int val = indexer.GetValueAtCurrentIndex<int>(arrayExpected);
				indexer.SetValueAtCurrentIndex(arrayComputed, val);
			} while (indexer.MoveNext());

			Assert.Equal(arrayExpected, arrayComputed);
		}

		[Theory]
		[InlineData(true)]
		[InlineData(false)]
		public static void TestIndexer3D(bool fullArray)
		{
			int dim0 = 8;
			int dim1 = 12;
			int dim2 = 16;
			int[] lengths;
			int[] lowerBounds;

			if (fullArray)
			{
				lengths = new int[] { dim0, dim1, dim2 };
				lowerBounds = null;
			}
			else
			{
				lengths = new int[] { dim0 - 1, dim1 - 2, dim2 - 3 };
				lowerBounds = new int[] { 0, 1, 2 };
			}

			int seed = 0;
			var rng = new Random(seed);
			var arrayExpected = new int[dim0, dim1, dim2];
			for (int i = 0; i < dim0; i++)
			{
				for (int j = 0; j < dim1; j++)
				{
					for (int k = 0; k < dim2; k++)
					{
						arrayExpected[i, j, k] = rng.Next();
					}
				}
			}

			var arrayComputed = new int[dim0, dim1, dim2];
			var indexer = new ArrayIndexer(lengths, lowerBounds);
			do
			{
				int val = indexer.GetValueAtCurrentIndex<int>(arrayExpected);
				indexer.SetValueAtCurrentIndex(arrayComputed, val);
			} while (indexer.MoveNext());

			Assert.Equal(arrayExpected, arrayComputed);
		}

		[Theory]
		[InlineData(true)]
		[InlineData(false)]
		public static void TestIndexer4D(bool fullArray)
		{
			int dim0 = 8;
			int dim1 = 12;
			int dim2 = 16;
			int dim3 = 20;
			int[] lengths;
			int[] lowerBounds;

			if (fullArray)
			{
				lengths = new int[] { dim0, dim1, dim2, dim3 };
				lowerBounds = null;
			}
			else
			{
				lengths = new int[] { dim0 - 1, dim1 - 2, dim2 - 3, dim3 - 4 };
				lowerBounds = new int[] { 0, 1, 2, 3 };
			}

			int seed = 0;
			var rng = new Random(seed);
			var arrayExpected = new int[dim0, dim1, dim2, dim3];
			for (int i = 0; i < dim0; i++)
			{
				for (int j = 0; j < dim1; j++)
				{
					for (int k = 0; k < dim2; k++)
					{
						for (int m = 0; m < dim3; m++)
						{
							arrayExpected[i, j, k, m] = rng.Next();
						}
					}
				}
			}

			var arrayComputed = new int[dim0, dim1, dim2, dim3];
			var indexer = new ArrayIndexer(lengths, lowerBounds);
			do
			{
				int val = indexer.GetValueAtCurrentIndex<int>(arrayExpected);
				indexer.SetValueAtCurrentIndex(arrayComputed, val);
			} while (indexer.MoveNext());

			Assert.Equal(arrayExpected, arrayComputed);
		}

		[Theory]
		[InlineData(true)]
		[InlineData(false)]
		public static void TestIndexer5D(bool fullArray)
		{
			int dim0 = 8;
			int dim1 = 12;
			int dim2 = 16;
			int dim3 = 20;
			int dim4 = 24;
			int[] lengths;
			int[] lowerBounds;

			if (fullArray)
			{
				lengths = new int[] { dim0, dim1, dim2, dim3, dim4 };
				lowerBounds = null;
			}
			else
			{
				lengths = new int[] { dim0 - 1, dim1 - 2, dim2 - 3, dim3 - 4, dim4 - 5 };
				lowerBounds = new int[] { 0, 1, 2, 3, 4 };
			}

			int seed = 0;
			var rng = new Random(seed);
			var arrayExpected = new int[dim0, dim1, dim2, dim3, dim4];
			for (int i = 0; i < dim0; i++)
			{
				for (int j = 0; j < dim1; j++)
				{
					for (int k = 0; k < dim2; k++)
					{
						for (int m = 0; m < dim2; m++)
						{
							for (int n = 0; n < dim3; n++)
							{
								arrayExpected[i, j, k, m, n] = rng.Next();
							}
						}
					}
				}
			}

			var arrayComputed = new int[dim0, dim1, dim2, dim3, dim4];
			var indexer = new ArrayIndexer(lengths, lowerBounds);
			do
			{
				int val = indexer.GetValueAtCurrentIndex<int>(arrayExpected);
				indexer.SetValueAtCurrentIndex(arrayComputed, val);
			} while (indexer.MoveNext());

			Assert.Equal(arrayExpected, arrayComputed);
		}

		[Theory]
		[InlineData(true)]
		[InlineData(false)]
		public static void TestIndexer6D(bool fullArray)
		{
			int dim0 = 8;
			int dim1 = 12;
			int dim2 = 16;
			int dim3 = 20;
			int dim4 = 24;
			int dim5 = 28;
			int[] lengths;
			int[] lowerBounds;

			if (fullArray)
			{
				lengths = new int[] { dim0, dim1, dim2, dim3, dim4, dim5 };
				lowerBounds = null;
			}
			else
			{
				lengths = new int[] { dim0 - 1, dim1 - 2, dim2 - 3, dim3 - 4, dim4 - 5, dim5 - 5 };
				lowerBounds = new int[] { 0, 1, 2, 3, 4, 5 };
			}

			int seed = 0;
			var rng = new Random(seed);
			var arrayExpected = new int[dim0, dim1, dim2, dim3, dim4, dim5];
			for (int i = 0; i < dim0; i++)
			{
				for (int j = 0; j < dim1; j++)
				{
					for (int k = 0; k < dim2; k++)
					{
						for (int m = 0; m < dim2; m++)
						{
							for (int n = 0; n < dim3; n++)
							{
								for (int o = 0; o < dim3; o++)
								{
									arrayExpected[i, j, k, m, n, o] = rng.Next();
								}
							}
						}
					}
				}
			}

			var arrayComputed = new int[dim0, dim1, dim2, dim3, dim4, dim5];
			var indexer = new ArrayIndexer(lengths, lowerBounds);
			do
			{
				int val = indexer.GetValueAtCurrentIndex<int>(arrayExpected);
				indexer.SetValueAtCurrentIndex(arrayComputed, val);
			} while (indexer.MoveNext());

			Assert.Equal(arrayExpected, arrayComputed);
		}
	}
}
