#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional
namespace MGroup.MachineLearning.Utilities
{
	using System;
	using System.Collections.Generic;

	/// <summary>
	/// Perhaps System.Numerics has something similar using tensors.
	/// Maybe user Buffer.Copy() when possible for better performance.
	/// </summary>
	public static class ChangeArrayDimensionsExtensions
	{
		public static T[] Add1EmptyDimension<T>(T scalar) => new T[] { scalar };

		public static T[,] Add2EmptyDimensions<T>(T scalar) => new T[,] { { scalar } };
		
		public static T[,,] Add3EmptyDimensions<T>(T scalar) => new T[,,] { { { scalar } } };

		public static T[,,,] Add4EmptyDimensions<T>(T scalar) => new T[,,,] { { { { scalar } } } };

		public static T[,] AddEmptyDimensions<T>(this T[] array, bool makeDim0Empty, bool makeDim1Empty)
		{
			CheckAddedEmptyDimensions(array, makeDim0Empty, makeDim1Empty); // check that exactly 1 dimension is non-empty
			if (!makeDim0Empty)
			{
				var result = new T[array.Length, 1];
				for (int i = 0; i < array.Length; i++)
				{
					result[i, 0] = array[i];
				}
				return result;
			}
			else
			{
				var result = new T[1, array.Length];
				for (int i = 0; i < array.Length; i++)
				{
					result[0, i] = array[i];
				}
				return result;
			}
		}

		public static T[,,] AddEmptyDimensions<T>(this T[] array, bool makeDim0Empty, bool makeDim1Empty, bool makeDim2Empty)
		{
			CheckAddedEmptyDimensions(array, makeDim0Empty, makeDim1Empty, makeDim2Empty); // check that exactly 1 dimension is non-empty
			if (!makeDim0Empty)
			{
				var result = new T[array.Length, 1, 1];
				for (int i = 0; i < array.Length; i++)
				{
					result[i, 0, 0] = array[i];
				}
				return result;
			}
			else if (!makeDim1Empty)
			{
				var result = new T[1, array.Length, 1];
				for (int i = 0; i < array.Length; i++)
				{
					result[0, i, 0] = array[i];
				}
				return result;
			}
			else
			{
				var result = new T[1, 1, array.Length];
				for (int i = 0; i < array.Length; i++)
				{
					result[0, 0, i] = array[i];
				}
				return result;
			}
		}

		public static T[,,,] AddEmptyDimensions<T>(this T[] array, bool makeDim0Empty, bool makeDim1Empty, bool makeDim2Empty, 
			bool makeDim3Empty)
		{
			CheckAddedEmptyDimensions(array, makeDim0Empty, makeDim1Empty, makeDim2Empty, makeDim3Empty); // check that exactly 1 dimension is non-empty
			if (!makeDim0Empty)
			{
				var result = new T[array.Length, 1, 1, 1];
				for (int i = 0; i < array.Length; i++)
				{
					result[i, 0, 0, 0] = array[i];
				}
				return result;
			}
			else if (!makeDim1Empty)
			{
				var result = new T[1, array.Length, 1, 1];
				for (int i = 0; i < array.Length; i++)
				{
					result[0, i, 0, 0] = array[i];
				}
				return result;
			}
			else if (!makeDim2Empty)
			{
				var result = new T[1, 1, array.Length, 1];
				for (int i = 0; i < array.Length; i++)
				{
					result[0, 0, i, 0] = array[i];
				}
				return result;
			}
			else
			{
				var result = new T[1, 1, 1, array.Length];
				for (int i = 0; i < array.Length; i++)
				{
					result[0, 0, 0, i] = array[i];
				}
				return result;
			}
		}

		public static T[,,] AddEmptyDimensions<T>(this T[,] array, bool makeDim0Empty, bool makeDim1Empty, bool makeDim2Empty)
		{
			CheckAddedEmptyDimensions(array, makeDim0Empty, makeDim1Empty, makeDim2Empty); // check that exactly 1 dimension is empty
			int m = array.GetLength(0);
			int n = array.GetLength(1);
			if (makeDim0Empty)
			{
				var result = new T[1, m, n];
				for (int i = 0; i < m; i++)
				{
					for (int j = 0; j < n; j++)
					{
						result[0, i, j] = array[i, j];
					}
				}
				return result;
			}
			else if (makeDim1Empty)
			{
				var result = new T[m, 1, n];
				for (int i = 0; i < m; i++)
				{
					for (int j = 0; j < n; j++)
					{
						result[i, 0, j] = array[i, j];
					}
				}
				return result;
			}
			else
			{
				var result = new T[m, n, 1];
				for (int i = 0; i < m; i++)
				{
					for (int j = 0; j < n; j++)
					{
						result[i, j, 0] = array[i, j];
					}
				}
				return result;
			}
		}

		public static T[,,,] AddEmptyDimensions<T>(this T[,] array, bool makeDim0Empty, bool makeDim1Empty, bool makeDim2Empty,
			bool makeDim3Empty)
		{
			CheckAddedEmptyDimensions(array, makeDim0Empty, makeDim1Empty, makeDim2Empty, makeDim3Empty); // check that exactly 2 dimensions are empty
			int m = array.GetLength(0);
			int n = array.GetLength(1);
			if (makeDim0Empty) // [1, ?, ?, ?] combinations, where exactly one ? is 1 and the two other ? are m or n
			{
				if (makeDim1Empty)
				{
					var result = new T[1, 1, m, n];
					for (int i = 0; i < m; i++)
					{
						for (int j = 0; j < n; j++)
						{
							result[0, 0, i, j] = array[i, j];
						}
					}
					return result;
				}
				else if (makeDim2Empty)
				{
					var result = new T[1, m, 1, n];
					for (int i = 0; i < m; i++)
					{
						for (int j = 0; j < n; j++)
						{
							result[0, i, 0, j] = array[i, j];
						}
					}
					return result;
				}
				else
				{
					var result = new T[1, m, n, 1];
					for (int i = 0; i < m; i++)
					{
						for (int j = 0; j < n; j++)
						{
							result[0, i, j, 0] = array[i, j];
						}
					}
					return result;
				}
			}
			else // [m, ?, ?, ?] combinations, where exactly one ? is n and the two other ? are 1
			{
				if (!makeDim1Empty)
				{
					var result = new T[m, n, 1, 1];
					for (int i = 0; i < m; i++)
					{
						for (int j = 0; j < n; j++)
						{
							result[i, j, 0, 0] = array[i, j];
						}
					}
					return result;
				}
				else if (!makeDim2Empty)
				{
					var result = new T[m, 1, n, 1];
					for (int i = 0; i < m; i++)
					{
						for (int j = 0; j < n; j++)
						{
							result[i, 0, j, 0] = array[i, j];
						}
					}
					return result;
				}
				else
				{
					var result = new T[m, 1, 1, n];
					for (int i = 0; i < m; i++)
					{
						for (int j = 0; j < n; j++)
						{
							result[i, 0, 0, j] = array[i, j];
						}
					}
					return result;
				}
			}
		}

		public static T[,,,] AddEmptyDimensions<T>(this T[,,] array, bool makeDim0Empty, bool makeDim1Empty, bool makeDim2Empty, 
			bool makeDim3Empty)
		{
			CheckAddedEmptyDimensions(array, makeDim0Empty, makeDim1Empty, makeDim2Empty); // check that exactly 1 dimension is empty
			int m = array.GetLength(0);
			int n = array.GetLength(1);
			int p = array.GetLength(2);
			if (makeDim0Empty)
			{
				var result = new T[1, m, n, p];
				for (int i = 0; i < m; i++)
				{
					for (int j = 0; j < n; j++)
					{
						for (int k = 0; k < p; k++)
						{
							result[0, i, j, k] = array[i, j, k];
						}
					}
				}
				return result;
			}
			else if (makeDim1Empty)
			{
				var result = new T[m, 1, n, p];
				for (int i = 0; i < m; i++)
				{
					for (int j = 0; j < n; j++)
					{
						for (int k = 0; k < p; k++)
						{
							result[i, 0, j, j] = array[i, j, k];
						}
					}
				}
				return result;
			}
			else if (makeDim2Empty)
			{
				var result = new T[m, n, 1, p];
				for (int i = 0; i < m; i++)
				{
					for (int j = 0; j < n; j++)
					{
						for (int k = 0; k < p; k++)
						{
							result[i, j, 0, k] = array[i, j, k];
						}
					}
				}
				return result;
			}
			else
			{
				var result = new T[m, n, p, 1];
				for (int i = 0; i < m; i++)
				{
					for (int j = 0; j < n; j++)
					{
						for (int k = 0; k < p; k++)
						{
							result[i, j, k, 0] = array[i, j, k];
						}
					}
				}
				return result;
			}
		}

		public static T RemoveAllEmptyDimensions<T>(this T[] array)
		{
			CheckDimensionsAreEmpty(array, 0);
			return array[0];
		}

		public static T RemoveAllEmptyDimensions<T>(this T[,] array)
		{
			CheckDimensionsAreEmpty(array, 0, 1);
			return array[0, 0];
		}

		public static T RemoveAllEmptyDimensions<T>(this T[,,] array)
		{
			CheckDimensionsAreEmpty(array, 0, 1, 2);
			return array[0, 0, 0];
		}

		public static T RemoveAllEmptyDimensions<T>(this T[,,,] array)
		{
			CheckDimensionsAreEmpty(array, 0, 1, 2, 3);
			return array[0, 0, 0, 0];
		}

		public static T[] RemoveEmptyDimension<T>(this T[,] array, int dimToRemove)
		{
			CheckDimensionsAreEmpty(array, dimToRemove);
			if (dimToRemove == 0)
			{
				int n = array.GetLength(1);
				var result = new T[n];
				for (int i = 0; i < n; ++i)
				{
					result[i] = array[0, i];
				}
				return result;
			}
			else // dimToRemove == 1
			{
				int n = array.GetLength(0);
				var result = new T[n];
				for (int i = 0; i < n; ++i)
				{
					result[i] = array[i, 0];
				}
				return result;
			}
		}

		public static T[,] RemoveEmptyDimension<T>(this T[,,] array, int dimToRemove)
		{
			CheckDimensionsAreEmpty(array, dimToRemove);
			if (dimToRemove == 0)
			{
				int m = array.GetLength(1);
				int n = array.GetLength(2);
				var result = new T[m, n];
				for (int i = 0; i < m; ++i)
				{
					for (int j = 0; j < n; ++j)
					{
						result[i, j] = array[0, i, j];
					}
				}
				return result;
			}
			else if (dimToRemove == 1)
			{
				int m = array.GetLength(0);
				int n = array.GetLength(2);
				var result = new T[m, n];
				for (int i = 0; i < m; ++i)
				{
					for (int j = 0; j < n; ++j)
					{
						result[i, j] = array[i, 0, j];
					}
				}
				return result;
			}
			else // dimToRemove == 2
			{
				int m = array.GetLength(0);
				int n = array.GetLength(1);
				var result = new T[m, n];
				for (int i = 0; i < m; ++i)
				{
					for (int j = 0; j < n; ++j)
					{
						result[i, j] = array[i, j, 0];
					}
				}
				return result;
			}
		}

		public static T[] RemoveEmptyDimensions<T>(this T[,,] array, int firstDimToRemove, int secondDimToRemove)
		{
			CheckDimensionsAreEmpty(array, firstDimToRemove, secondDimToRemove);
			if (firstDimToRemove != 0 && secondDimToRemove != 0) // keep dim 0
			{
				int n = array.GetLength(0);
				var result = new T[n];
				for (int i = 0; i < n; ++i)
				{
					result[i] = array[i, 0, 0];
				}
				return result;
			}
			else if (firstDimToRemove != 1 && secondDimToRemove != 1) // keep dim 1
			{
				int n = array.GetLength(1);
				var result = new T[n];
				for (int i = 0; i < n; ++i)
				{
					result[i] = array[0, i, 0];
				}
				return result;
			}
			else // keep dim 2
			{
				int n = array.GetLength(2);
				var result = new T[n];
				for (int i = 0; i < n; ++i)
				{
					result[i] = array[0, 0, i];
				}
				return result;
			}
		}

		public static T[,,] RemoveEmptyDimension<T>(this T[,,,] array, int dimToRemove)
		{
			CheckDimensionsAreEmpty(array, dimToRemove);
			if (dimToRemove == 0)
			{
				int m = array.GetLength(1);
				int n = array.GetLength(2);
				int p = array.GetLength(3);
				var result = new T[m, n, p];
				for (int i = 0; i < m; ++i)
				{
					for (int j = 0; j < n; ++j)
					{
						for (int k = 0; k < p; ++k)
						{
							result[i, j, k] = array[0, i, j, k];
						}
					}
				}
				return result;
			}
			else if (dimToRemove == 1)
			{
				int m = array.GetLength(0);
				int n = array.GetLength(2);
				int p = array.GetLength(3);
				var result = new T[m, n, p];
				for (int i = 0; i < m; ++i)
				{
					for (int j = 0; j < n; ++j)
					{
						for (int k = 0; k < p; ++k)
						{
							result[i, j, k] = array[i, 0, j, k];
						}
					}
				}
				return result;
			}
			else if (dimToRemove == 2)
			{
				int m = array.GetLength(0);
				int n = array.GetLength(1);
				int p = array.GetLength(3);
				var result = new T[m, n, p];
				for (int i = 0; i < m; ++i)
				{
					for (int j = 0; j < n; ++j)
					{
						for (int k = 0; k < p; ++k)
						{
							result[i, j, k] = array[i, j, 0, k];
						}
					}
				}
				return result;
			}
			else // dimToRemove == 3
			{
				int m = array.GetLength(0);
				int n = array.GetLength(1);
				int p = array.GetLength(2);
				var result = new T[m, n, p];
				for (int i = 0; i < m; ++i)
				{
					for (int j = 0; j < n; ++j)
					{
						for (int k = 0; k < p; ++k)
						{
							result[i, j, k] = array[i, j, k, 0];
						}
					}
				}
				return result;
			}
		}

		public static T[,] RemoveEmptyDimensions<T>(this T[,,,] array, int firstDimToRemove, int secondDimToRemove)
		{
			CheckDimensionsAreEmpty(array, firstDimToRemove, secondDimToRemove);
			if (firstDimToRemove == 0)  // [1, ?, ?, ?] combinations, where exactly one ? is 1 and the two other ? are m or n
			{
				if (secondDimToRemove == 1) // [1, 1, m, n]
				{
					int m = array.GetLength(2);
					int n = array.GetLength(3);
					var result = new T[m, n];
					for (int i = 0; i < m; ++i)
					{
						for (int j = 0; j < n; ++j)
						{
							result[i, j] = array[0, 0, i, j];
						}
					}
					return result;
				}
				else if (secondDimToRemove == 2) // [1, m, 1, n]
				{
					int m = array.GetLength(1);
					int n = array.GetLength(3);
					var result = new T[m, n];
					for (int i = 0; i < m; ++i)
					{
						for (int j = 0; j < n; ++j)
						{
							result[i, j] = array[0, i, 0, j];
						}
					}
					return result;
				}
				else // [1, m, n, 1]
				{
					int m = array.GetLength(1);
					int n = array.GetLength(2);
					var result = new T[m, n];
					for (int i = 0; i < m; ++i)
					{
						for (int j = 0; j < n; ++j)
						{
							result[i, j] = array[0, i, j, 0];
						}
					}
					return result;
				}
			}
			else // [m, ?, ?, ?] combinations, where exactly one ? is n and the two other ? are 1
			{
				if (secondDimToRemove != 1) // [m, n, 1, 1]
				{
					int m = array.GetLength(0);
					int n = array.GetLength(1);
					var result = new T[m, n];
					for (int i = 0; i < m; ++i)
					{
						for (int j = 0; j < n; ++j)
						{
							result[i, j] = array[i, j, 0, 0];
						}
					}
					return result;
				}
				else if (secondDimToRemove != 2) // [m, 1, n, 1]
				{
					int m = array.GetLength(0);
					int n = array.GetLength(2);
					var result = new T[m, n];
					for (int i = 0; i < m; ++i)
					{
						for (int j = 0; j < n; ++j)
						{
							result[i, j] = array[i, 0, j, 0];
						}
					}
					return result;
				}
				else // [m, 1, 1, n]
				{
					int m = array.GetLength(0);
					int n = array.GetLength(3);
					var result = new T[m, n];
					for (int i = 0; i < m; ++i)
					{
						for (int j = 0; j < n; ++j)
						{
							result[i, j] = array[i, 0, 0, j];
						}
					}
					return result;
				}
			}
		}

		public static T[] RemoveEmptyDimensions<T>(this T[,,,] array, int firstDimToRemove, int secondDimToRemove, 
			int thirdDimToRemove)
		{
			CheckDimensionsAreEmpty(array, firstDimToRemove, secondDimToRemove, thirdDimToRemove);
			if (firstDimToRemove != 0 && secondDimToRemove != 0 && thirdDimToRemove != 0) // keep dim 0
			{
				int n = array.GetLength(0);
				var result = new T[n];
				for (int i = 0; i < n; ++i)
				{
					result[i] = array[i, 0, 0, 0];
				}
				return result;
			}
			else if (firstDimToRemove != 1 && secondDimToRemove != 1 && thirdDimToRemove != 1) // keep dim 1
			{
				int n = array.GetLength(1);
				var result = new T[n];
				for (int i = 0; i < n; ++i)
				{
					result[i] = array[0, i, 0, 0];
				}
				return result;
			}
			else if (firstDimToRemove != 2 && secondDimToRemove != 2 && thirdDimToRemove != 2) // keep dim 2
			{
				int n = array.GetLength(2);
				var result = new T[n];
				for (int i = 0; i < n; ++i)
				{
					result[i] = array[0, 0, i, 0];
				}
				return result;
			}
			else // keep dim 3
			{
				int n = array.GetLength(3);
				var result = new T[n];
				for (int i = 0; i < n; ++i)
				{
					result[i] = array[0, 0, 0, i];
				}
				return result;
			}
		}

		private static void CheckAddedEmptyDimensions(Array originalArray, params bool[] areEmptyDimensions)
		{
			int numIn = originalArray.Rank;
			int numOut = areEmptyDimensions.Length;
			int numEmpty = 0;
			for (int i = 0; i < areEmptyDimensions.Length; i++)
			{
				if (areEmptyDimensions[i])
				{
					numEmpty++;
				}
			}

			if (numEmpty == numOut - numIn)
			{
				return;
			}
			else
			{
				throw new ArgumentException(
					$"When changing a {numIn}d array to a {numOut}d array, exactly {numOut - numIn} dimensions must be empty," +
					$" but {numEmpty} empty dimensions were specified.");
			}
		}

		private static void CheckDimensionsAreEmpty(Array array, params int[] emptyDimensions)
		{
			int rank = array.Rank;
			var uniqueDimensions = new SortedSet<int>();
			foreach (int dim in emptyDimensions)
			{
				if (dim < 0 || dim >= rank)
				{
					throw new ArgumentException(
						$"Cannot remove dimension {dim} of a {rank}d array, since it has dimensions 0 ... {rank-1}");
				}
				if (array.GetLength(dim) != 1)
				{
					throw new ArgumentException($"The array's length along dimension {dim} is {array.GetLength(dim)}, " +
						$"thus this dimension is not empty and cannot be removed.");
				}
				if (uniqueDimensions.Contains(dim))
				{
					throw new ArgumentException($"Each dimension can be declared at most one time for removal, " +
						$"but dimension {dim} has been declared multiple times.");
				}
				uniqueDimensions.Add(dim);
			}
		}
	}
}
#pragma warning restore CA1814 // Prefer jagged arrays over multidimensional
