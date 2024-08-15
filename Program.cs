using System;
using NumFlat;
using OpenBlasSharp;

static class Program
{
    unsafe static void Main(string[] args)
    {
        Mat<double> a =
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ];

        var s = new Vec<double>(Math.Min(a.RowCount, a.ColCount));
        var u = new Mat<double>(a.RowCount, a.RowCount);
        var vt = new Mat<double>(a.ColCount, a.ColCount);
        var work = new Vec<double>(s.Count - 1);

        fixed (double* pa = a.Memory.Span)
        fixed (double* ps = s.Memory.Span)
        fixed (double* pu = u.Memory.Span)
        fixed (double* pvt = vt.Memory.Span)
        fixed (double* pwork = work.Memory.Span)
        {
            Lapack.Dgesvd(
                MatrixLayout.ColMajor,
                'A', 'A',
                a.RowCount, a.ColCount,
                pa, a.Stride, // The content of 'a' will be destroyed.
                ps,
                pu, u.Stride,
                pvt, vt.Stride,
                pwork);
        }

        Console.WriteLine(u * s.ToDiagonalMatrix() * vt);
    }
}
