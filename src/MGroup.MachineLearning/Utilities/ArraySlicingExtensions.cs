namespace MGroup.MachineLearning.Utilities
{
	using System;
	using System.Runtime.CompilerServices;

	public static class ArraySlicingExtensions
	{
		/// <param name="start">Inclusive.</param>
		/// <param name="end">Exclusive.</param>
		public static T[] Slice<T>(this T[] array, int start, int end)
		{
			CheckRange(start, end, null);

			int resultLength = end - start;
			var result = new T[resultLength];
			for (int i = 0; i < resultLength; i++)
			{
				result[i] = array[start + i];
			}

			return result;
		}

		/// <param name="start">Inclusive. If null: start of array.</param>
		/// <param name="end">Exclusive. If null: end of array.</param>
		public static T[] Slice<T>(this T[] array, int? start, int? end)
			=> Slice(array, start ?? 0, end ?? array.Length);

		/// <param name="range">
		/// (start, end), where: Start is inclusive and can be null for start of the array.
		/// End is exclusive and can be null for end of the array.
		/// If the range itself is null, all entries will be taken.
		/// </param>
		public static T[] Slice<T>(this T[] array, (int? start, int? end)? range)
			=> Slice(array, EnsureRange(range, 0, array));

		/// <param name="startDim0">Start along dimension 0. Inclusive.</param>
		/// <param name="endDim0">End along dimension 0. Exclusive.</param>
		/// <param name="startDim1">Start along dimension 1. Inclusive.</param>
		/// <param name="endDim1">End along dimension 1. Exclusive.</param>
		public static T[,] Slice<T>(this T[,] array, int startDim0, int endDim0, int startDim1, int endDim1)
		{
			CheckRange(startDim0, endDim0, 0);
			CheckRange(startDim1, endDim1, 1);

			int resultLength0 = endDim0 - startDim0;
			int resultLength1 = endDim1 - startDim1;
			var result = new T[resultLength0, resultLength1];
			for (int i = 0; i < resultLength0; i++)
			{
				int offset0 = startDim0 + i;
				for (int j = 0; j < resultLength1; j++)
				{
					result[i, j] = array[offset0, startDim1 + j];
				}
			}

			return result;
		}

		/// <param name="rangeDim0">
		/// (start, end) along dimension 0, where: Start is inclusive and can be null for start of the array along dimension 0.
		/// End is exclusive and can be null for end of the array along dimension 0.
		/// If the range itself is null, all entries along dimension 0 will be taken.
		/// </param>
		/// <param name="rangeDim1">
		/// (start, end) along dimension 1, where: Start is inclusive and can be null for start of the array along dimension 1.
		/// End is exclusive and can be null for end of the array along dimension 1.
		/// If the range itself is null, all entries along dimension 1 will be taken.
		/// </param>
		public static T[,] Slice<T>(this T[,] array, (int? start, int? end)? rangeDim0, (int? start, int? end)? rangeDim1)
		{
			(int start0, int end0) = EnsureRange(rangeDim0, 0, array);
			(int start1, int end1) = EnsureRange(rangeDim1, 1, array);
			return Slice(array, start0, end0, start1, end1);
		}

		/// <param name="startDim0">Start along dimension 0. Inclusive.</param>
		/// <param name="endDim0">End along dimension 0. Exclusive.</param>
		/// <param name="startDim1">Start along dimension 1. Inclusive.</param>
		/// <param name="endDim1">End along dimension 1. Exclusive.</param>
		/// <param name="startDim2">Start along dimension 2. Inclusive.</param>
		/// <param name="endDim2">End along dimension 2. Exclusive.</param>
		public static T[,,] Slice<T>(this T[,,] array, int startDim0, int endDim0, int startDim1, int endDim1,
			int startDim2, int endDim2)
		{
			CheckRange(startDim0, endDim0, 0);
			CheckRange(startDim1, endDim1, 1);
			CheckRange(startDim2, endDim2, 2);

			int resultLength0 = endDim0 - startDim0;
			int resultLength1 = endDim1 - startDim1;
			int resultLength2 = endDim2 - startDim2;
			var result = new T[resultLength0, resultLength1, resultLength2];
			for (int i = 0; i < resultLength0; i++)
			{
				int offset0 = startDim0 + i;
				for (int j = 0; j < resultLength1; j++)
				{
					int offset1 = startDim1 + j;
					for (int k = 0; k < resultLength2; k++)
					{
						result[i, j, k] = array[offset0, offset1, startDim2 + k];
					}
				}
			}

			return result;
		}

		/// <param name="rangeDim0">
		/// (start, end) along dimension 0, where: Start is inclusive and can be null for start of the array along dimension 0.
		/// End is exclusive and can be null for end of the array along dimension 0.
		/// If the range itself is null, all entries along dimension 0 will be taken.
		/// </param>
		/// <param name="rangeDim1">
		/// (start, end) along dimension 1, where: Start is inclusive and can be null for start of the array along dimension 1.
		/// End is exclusive and can be null for end of the array along dimension 1.
		/// If the range itself is null, all entries along dimension 1 will be taken.
		/// </param>
		/// /// <param name="rangeDim2">
		/// (start, end) along dimension 2, where: Start is inclusive and can be null for start of the array along dimension 2.
		/// End is exclusive and can be null for end of the array along dimension 2.
		/// If the range itself is null, all entries along dimension 2 will be taken.
		/// </param>
		public static T[,,] Slice<T>(this T[,,] array, (int? start, int? end)? rangeDim0, (int? start, int? end)? rangeDim1,
			(int? start, int? end)? rangeDim2)
		{
			(int start0, int end0) = EnsureRange(rangeDim0, 0, array);
			(int start1, int end1) = EnsureRange(rangeDim1, 1, array);
			(int start2, int end2) = EnsureRange(rangeDim2, 2, array);
			return Slice(array, start0, end0, start1, end1, start2, end2);
		}

		/// <param name="startDim0">Start along dimension 0. Inclusive.</param>
		/// <param name="endDim0">End along dimension 0. Exclusive.</param>
		/// <param name="startDim1">Start along dimension 1. Inclusive.</param>
		/// <param name="endDim1">End along dimension 1. Exclusive.</param>
		/// <param name="startDim2">Start along dimension 2. Inclusive.</param>
		/// <param name="endDim2">End along dimension 2. Exclusive.</param>
		/// <param name="startDim3">Start along dimension 3. Inclusive.</param>
		/// <param name="endDim3">End along dimension 3. Exclusive.</param>
		public static T[,,,] Slice<T>(this T[,,,] array, int startDim0, int endDim0, int startDim1, int endDim1,
			int startDim2, int endDim2, int startDim3, int endDim3)
		{
			CheckRange(startDim0, endDim0, 0);
			CheckRange(startDim1, endDim1, 1);
			CheckRange(startDim2, endDim2, 2);
			CheckRange(startDim3, endDim3, 3);

			int resultLength0 = endDim0 - startDim0;
			int resultLength1 = endDim1 - startDim1;
			int resultLength2 = endDim2 - startDim2;
			int resultLength3 = endDim3 - startDim3;
			var result = new T[resultLength0, resultLength1, resultLength2, resultLength3];
			for (int i = 0; i < resultLength0; i++)
			{
				int offset0 = startDim0 + i;
				for (int j = 0; j < resultLength1; j++)
				{
					int offset1 = startDim1 + j;
					for (int k = 0; k < resultLength2; k++)
					{
						int offset2 = startDim2 + k;
						for (int m = 0; m < resultLength3; m++)
						{
							result[i, j, k, m] = array[offset0, offset1, offset2, startDim3 + m];
						}
					}
				}
			}

			return result;
		}

		/// <param name="rangeDim0">
		/// (start, end) along dimension 0, where: Start is inclusive and can be null for start of the array along dimension 0.
		/// End is exclusive and can be null for end of the array along dimension 0.
		/// If the range itself is null, all entries along dimension 0 will be taken.
		/// </param>
		/// <param name="rangeDim1">
		/// (start, end) along dimension 1, where: Start is inclusive and can be null for start of the array along dimension 1.
		/// End is exclusive and can be null for end of the array along dimension 1.
		/// If the range itself is null, all entries along dimension 1 will be taken.
		/// </param>
		/// /// <param name="rangeDim2">
		/// (start, end) along dimension 2, where: Start is inclusive and can be null for start of the array along dimension 2.
		/// End is exclusive and can be null for end of the array along dimension 2.
		/// If the range itself is null, all entries along dimension 2 will be taken.
		/// </param>
		/// <param name="rangeDim3">
		/// (start, end) along dimension 3, where: Start is inclusive and can be null for start of the array along dimension 3.
		/// End is exclusive and can be null for end of the array along dimension 3.
		/// If the range itself is null, all entries along dimension 3 will be taken.
		/// </param>
		public static T[,,,] Slice<T>(this T[,,,] array, (int? start, int? end)? rangeDim0, (int? start, int? end)? rangeDim1,
			(int? start, int? end)? rangeDim2, (int? start, int? end)? rangeDim3)
		{
			(int start0, int end0) = EnsureRange(rangeDim0, 0, array);
			(int start1, int end1) = EnsureRange(rangeDim1, 1, array);
			(int start2, int end2) = EnsureRange(rangeDim2, 2, array);
			(int start3, int end3) = EnsureRange(rangeDim2, 3, array);
			return Slice(array, start0, end0, start1, end1, start2, end2, start3, end3);
		}

		/// <param name="startDim0">Start along dimension 0. Inclusive.</param>
		/// <param name="endDim0">End along dimension 0. Exclusive.</param>
		/// <param name="startDim1">Start along dimension 1. Inclusive.</param>
		/// <param name="endDim1">End along dimension 1. Exclusive.</param>
		/// <param name="startDim2">Start along dimension 2. Inclusive.</param>
		/// <param name="endDim2">End along dimension 2. Exclusive.</param>
		/// <param name="startDim3">Start along dimension 3. Inclusive.</param>
		/// <param name="endDim3">End along dimension 3. Exclusive.</param>
		/// <param name="startDim4">Start along dimension 4. Inclusive.</param>
		/// <param name="endDim4">End along dimension 4. Exclusive.</param>
		public static T[,,,,] Slice<T>(this T[,,,,] array, int startDim0, int endDim0, int startDim1, int endDim1,
			int startDim2, int endDim2, int startDim3, int endDim3, int startDim4, int endDim4)
		{
			CheckRange(startDim0, endDim0, 0);
			CheckRange(startDim1, endDim1, 1);
			CheckRange(startDim2, endDim2, 2);
			CheckRange(startDim3, endDim3, 3);
			CheckRange(startDim4, endDim4, 3);

			int resultLength0 = endDim0 - startDim0;
			int resultLength1 = endDim1 - startDim1;
			int resultLength2 = endDim2 - startDim2;
			int resultLength3 = endDim3 - startDim3;
			int resultLength4 = endDim4 - startDim4;
			var result = new T[resultLength0, resultLength1, resultLength2, resultLength3, resultLength4];
			for (int i = 0; i < resultLength0; i++)
			{
				int offset0 = startDim0 + i;
				for (int j = 0; j < resultLength1; j++)
				{
					int offset1 = startDim1 + j;
					for (int k = 0; k < resultLength2; k++)
					{
						int offset2 = startDim2 + k;
						for (int m = 0; m < resultLength3; m++)
						{
							int offset3 = startDim3 + m;
							for (int n = 0; n < resultLength4; n++)
							{
								result[i, j, k, m, n] = array[offset0, offset1, offset2, offset3, startDim4 + n];
							}
						}
					}
				}
			}

			return result;
		}

		/// <param name="rangeDim0">
		/// (start, end) along dimension 0, where: Start is inclusive and can be null for start of the array along dimension 0.
		/// End is exclusive and can be null for end of the array along dimension 0.
		/// If the range itself is null, all entries along dimension 0 will be taken.
		/// </param>
		/// <param name="rangeDim1">
		/// (start, end) along dimension 1, where: Start is inclusive and can be null for start of the array along dimension 1.
		/// End is exclusive and can be null for end of the array along dimension 1.
		/// If the range itself is null, all entries along dimension 1 will be taken.
		/// </param>
		/// /// <param name="rangeDim2">
		/// (start, end) along dimension 2, where: Start is inclusive and can be null for start of the array along dimension 2.
		/// End is exclusive and can be null for end of the array along dimension 2.
		/// If the range itself is null, all entries along dimension 2 will be taken.
		/// </param>
		/// <param name="rangeDim3">
		/// (start, end) along dimension 3, where: Start is inclusive and can be null for start of the array along dimension 3.
		/// End is exclusive and can be null for end of the array along dimension 3.
		/// If the range itself is null, all entries along dimension 3 will be taken.
		/// </param>
		/// <param name="rangeDim4">
		/// (start, end) along dimension 4, where: Start is inclusive and can be null for start of the array along dimension 4.
		/// End is exclusive and can be null for end of the array along dimension 4.
		/// If the range itself is null, all entries along dimension 4 will be taken.
		/// </param>
		public static T[,,,,] Slice<T>(this T[,,,,] array, (int? start, int? end)? rangeDim0, (int? start, int? end)? rangeDim1,
			(int? start, int? end)? rangeDim2, (int? start, int? end)? rangeDim3, (int? start, int? end)? rangeDim4)
		{
			(int start0, int end0) = EnsureRange(rangeDim0, 0, array);
			(int start1, int end1) = EnsureRange(rangeDim1, 1, array);
			(int start2, int end2) = EnsureRange(rangeDim2, 2, array);
			(int start3, int end3) = EnsureRange(rangeDim2, 3, array);
			(int start4, int end4) = EnsureRange(rangeDim2, 4, array);
			return Slice(array, start0, end0, start1, end1, start2, end2, start3, end3, start4, end4);
		}

		private static (int start, int end) EnsureRange((int? start, int? end)? range, int dimension, Array array)
		{
			if (range == null)
			{
				return (0, array.GetLength(dimension));
			}
			else
			{
				return (range.Value.start ?? 0, range.Value.end ?? array.GetLength(dimension));
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static void CheckRange(int start, int end, int? dim)
		{
			if (end < start)
			{
				string dimensionText = dim != null ? " along dimension " + dim : string.Empty;
				throw new ArgumentException("End entry must be greater than start entry" + dimensionText + ".");
			}
		}
	}
}
