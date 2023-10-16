namespace MGroup.MachineLearning.Utilities
{
	using System;
	using System.Collections.Generic;
	using System.Diagnostics;
	using System.Text;

	public class ArrayIndexer
	{
		private readonly int[] _lengths;
		private readonly int[] _lowerBounds;
		private readonly int _rank;

		private int[] _currentIdx;
		private bool _finishedIterating;

		public ArrayIndexer(int[] arrayLengthAlongEachDimension, int[] lowerBoundsAlongEachDimension = null)
		{
			// Check that lengths and loweBounds are valid
			if (arrayLengthAlongEachDimension == null)
			{
				throw new ArgumentException("The lengths array can not be null");
			}
			_rank = arrayLengthAlongEachDimension.Length;

			if (lowerBoundsAlongEachDimension == null)
			{
				lowerBoundsAlongEachDimension = new int[_rank];
			}

			if (lowerBoundsAlongEachDimension.Length != arrayLengthAlongEachDimension.Length) 
			{
				throw new ArgumentException(
					"The length of the lengths array must be equal to the length of the lower bounds array");
			}

			for (int d = 0; d < _rank; ++d)
			{
				if (lowerBoundsAlongEachDimension[d] < 0)
				{
					throw new ArgumentException($"The lower bound along dimension {d} must be >= 0");
				}

				if (arrayLengthAlongEachDimension[d] < 1)
				{
					throw new ArgumentException($"The array length along dimension {d} must be > 1");
				}
			}

			_lengths = arrayLengthAlongEachDimension;
			_lowerBounds = lowerBoundsAlongEachDimension;

			_currentIdx = new int[_rank];
			Array.Copy(_lowerBounds, _currentIdx, _rank);
			_finishedIterating = false;
		}

		/// <summary>
		/// Do not use this to access arrays, if <see cref="FinishedIterating"/> is false or the last call to 
		/// <see cref="MoveNext"/> returned false.
		/// </summary>
		public int[] CurrentIndex => _currentIdx;

		public bool FinishedIterating => _finishedIterating;

		public T GetValueAtCurrentIndex<T>(T[] array)
		{
			CheckArrayShape(array);
			return array[_currentIdx[0]];
		}

		public T GetValueAtCurrentIndex<T>(T[,] array)
		{
			CheckArrayShape(array);
			return array[_currentIdx[0], _currentIdx[1]];
		}

		public T GetValueAtCurrentIndex<T>(T[,,] array)
		{
			CheckArrayShape(array);
			return array[_currentIdx[0], _currentIdx[1], _currentIdx[2]];
		}

		public T GetValueAtCurrentIndex<T>(T[,,,] array)
		{
			CheckArrayShape(array);
			return array[_currentIdx[0], _currentIdx[1], _currentIdx[2], _currentIdx[3]];
		}

		public T GetValueAtCurrentIndex<T>(T[,,,,] array)
		{
			CheckArrayShape(array);
			return array[_currentIdx[0], _currentIdx[1], _currentIdx[2], _currentIdx[3], _currentIdx[4]];
		}

		public T GetValueAtCurrentIndex<T>(Array array)
		{
			CheckArrayShape(array);
			if (_rank == 1)
			{
				return (T)(array.GetValue(_currentIdx[0]));
			}
			else
			{
				return (T)(array.GetValue(_currentIdx));
			}
		}

		public bool IsArrayCompatible(Array array)
		{
			if (array.Rank != _rank)
			{
				return false;
			}

			for (int d = 0; d < _rank; d++)
			{
				if (array.GetLength(d) < _lowerBounds[d] + _lengths[d])
				{
					return false;
				}
			}
			return true;
		}

		/// <summary>
		/// Moves the index to the next entry of the compatible array(s). Returns true if there are more elements in the array
		/// after the move.
		/// </summary>
		/// <returns>Returns true if there are more elements in the array after the move.</returns>
		public bool MoveNext()
		{
			if (_finishedIterating)
			{
				return false;
			}

			for (int d = _rank - 1; d >= 0; --d)
			{
				if (_currentIdx[d] < _lengths[d] - 1)
				{
					// There are more entries along this direction
					++_currentIdx[d];
					return true;
				}
				else
				{
					// Restart at 0 for this dimension and move to the previous one to check if that one has leftover entries.
					_currentIdx[d] = _lowerBounds[d];
				}
			}

			// At this point, we found that there are no more entries along all dimensions and the index is
			// set to { lb0, lb1, ... }. Set the index to an invalid one and stop the iteration
			for (int d = 0; d < _rank; ++d)
			{
				_currentIdx[d] = int.MaxValue;
			}
			_finishedIterating = true;
			return false;
		}

		public void Restart()
		{
			_currentIdx = new int[_rank];
			Array.Copy(_lowerBounds, _currentIdx, _rank);
			_finishedIterating = false;
		}

		public void SetValueAtCurrentIndex<T>(T[] array, T value)
		{
			CheckArrayShape(array);
			array[_currentIdx[0]] = value;
		}

		public void SetValueAtCurrentIndex<T>(T[,] array, T value)
		{
			CheckArrayShape(array);
			array[_currentIdx[0], _currentIdx[1]] = value;
		}

		public void SetValueAtCurrentIndex<T>(T[,,] array, T value)
		{
			CheckArrayShape(array);
			array[_currentIdx[0], _currentIdx[1], _currentIdx[2]] = value;
		}

		public void SetValueAtCurrentIndex<T>(T[,,,] array, T value)
		{
			CheckArrayShape(array);
			array[_currentIdx[0], _currentIdx[1], _currentIdx[2], _currentIdx[3]] = value;
		}

		public void SetValueAtCurrentIndex<T>(T[,,,,] array, T value)
		{
			CheckArrayShape(array);
			array[_currentIdx[0], _currentIdx[1], _currentIdx[2], _currentIdx[3], _currentIdx[4]] = value;
		}

		public void SetValueAtCurrentIndex<T>(Array array, T value)
		{
			CheckArrayShape(array);
			if (_rank == 1)
			{
				array.SetValue(value, _currentIdx[0]);
			}
			else
			{
				array.SetValue(value, _currentIdx);
			}
		}

		[Conditional("DEBUG")]
		private void CheckArrayShape(Array array)
		{
			bool isCompatible = IsArrayCompatible(array);
			if (!isCompatible)
			{
				throw new ArgumentException($"The shape of the provided array {DescribeShape(array)} is incompatible with the " +
					$" shape of this indexer {DescribeShape()}. The indexer must not exceed the array's length in any dimension");
			}
		}

		private string DescribeShape()
		{
			var text = new StringBuilder();
			text.Append("(");
			text.Append(_lowerBounds[0]);
			text.Append("-");
			text.Append(_lowerBounds[0] + _lengths[0] - 1);
			for (int d = 1; d < _rank; d++)
			{
				text.Append(", ");
				text.Append(_lowerBounds[d]);
				text.Append("-");
				text.Append(_lowerBounds[d] + _lengths[d] - 1);
			}
			text.Append(")");
			return text.ToString();
		}

		private string DescribeShape(Array array)
		{
			var text = new StringBuilder();
			text.Append("(");
			text.Append(array.GetLength(0));
			for (int d = 1; d < array.Rank; d++)
			{
				text.Append(", ");
				text.Append(array.GetLength(d));
			}
			text.Append(")");
			return text.ToString();
		}
	}
}
