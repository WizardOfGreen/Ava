#ifndef MATRIX_CPP
#define MATRIX_CPP

#include "inc/Matrix.h"

template class Matrix<float>;
template class Matrix<int>;
template class Matrix<double>;

template <typename T>
Matrix<T>::Matrix()
{
}

template <typename T>
const std::vector<std::vector<T>> Matrix<T>::MultiplyMatrix(const std::vector<std::vector<T>> &x, const std::vector<std::vector<T>> &y)
{
    int x1 = x.size();
    int x2 = x[0].size();
    int y1 = y[0].size();

    if (x1 != y1)
    {
        std::cerr << "Matrixes cannot be multiplied\n";
        exit(-1);
    }

    std::vector<std::vector<T>> Result;
    T Res;

    for (int i = 0; i < x1; i++)
    {
        Result.push_back(std::vector<T>());
        for (int j = 0; j < y1; j++)
        {
            Result[i].push_back(0);
        }
    }

    for (int i = 0; i < x1; i++)
    {
        for (int j = 0; j < y1; j++)
        {
            Res = 0;
            for (int k = 0; k < x2; k++)
            {
                Res += (x[i][k] * y[k][j]);
            }
            Result[i][j] = Res;
        }
    }

    Cleanup(Result);
    return Result;
}

template <typename T>
const std::vector<std::vector<T>> Matrix<T>::ScalarMult(T s, const std::vector<std::vector<T>> &x)
{
    int row = x.size();
    int col = x[0].size();

    std::vector<std::vector<T>> R;
    for (int i = 0; i < row; i++)
    {
        R.push_back(std::vector<T>());
        for (int j = 0; j < col; j++)
        {
            R[i].push_back(s * x[i][j]);
        }
    }
    return R;
}

template <typename T>
const std::vector<std::vector<T>> Matrix<T>::TransposeMatrix(const std::vector<std::vector<T>> &x)
{
    std::vector<std::vector<T>> TransposedMatrix;

    int row = x.size();
    int col = x[0].size();

    for (int i = 0; i < col; i++)
    {
        TransposedMatrix.push_back(std::vector<T>());
        for (int j = 0; j < row; j++)
        {
            TransposedMatrix[i].push_back(x[j][i]);
        }
    }
    Cleanup(TransposedMatrix);
    return TransposedMatrix;
}

template <typename T>
std::vector<std::vector<T>> Matrix<T>::GetIdentityMatrix(int x)
{
    std::vector<std::vector<T>> R;
    for (int i = 0; i < x; i++)
    {
        R.push_back(std::vector<T>());
        for (int j = 0; j < x; j++)
        {
            if (j == i)
                R[i].push_back(1);
            else
                R[i].push_back(0);
        }
    }
    Cleanup(R);
    return R;
}

// Time : O(n!)
// Space : O(n^2)
template <typename T>
const T Matrix<T>::Det(const std::vector<std::vector<T>> &x)
{
    int n = x.size();

    if (n == 1)
        return x[0][0];

    T det = 0;
    int sign = 1;
    if (n == 2)
    {
        det = (x[0][0] * x[1][1]) - (x[0][1] * x[1][0]);
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            det += (x[0][i] * Det(CreateSmallerMatrix(x, 0, i))) * sign;
            sign *= -1;
        }
    }

    return det;
}

template <typename T>
const T Matrix<T>::GaussElimDet(const std::vector<std::vector<T>> &x)
{
    std::vector<std::vector<T>> R = CopyOfMatrix(x);

    int h = 0;
    int k = 0;

    int sizey = x.size();
    int sizex = x[0].size();

    int i = 0;
    int j = 0;
    int sign = 1;

    printMatrixes(R);

    while (h <= (sizey - 1) && k <= (sizex - 1))
    {
        int i_max = MaxH(h, k, R);
        if (R[i_max][k] == 0)
        {
            k++;
        }
        else
        {
            if (i_max != h)
            {
                R = this->RowInter(R, h, i_max);

                sign *= -1;
            }
            for (i = h + 1; i < sizey; i++)
            {
                T f = R[i][k] / R[h][k];
                R[i][k] = 0;
                for (j = k + 1; j < sizex; j++)
                {
                    R[i][j] = R[i][j] - (R[h][j] * f);
                }
            }
            h++;
            k++;
        }
    }

    T Det = 1;
    for (int i = 0; i < sizey; i++)
    {
        Det *= R[i][i];
    }

    if (Det != 0)
        Det *= sign;

    return Det;
}

template <typename T>
int Matrix<T>::MaxH(int h, int k, const std::vector<std::vector<T>> &x)
{
    int i_max = h;
    T val = std::abs(x[h][k]);
    int size = x.size();

    for (int i = h + 1; i < size; ++i)
    {
        if (std::abs(x[i][k]) > val)
        {
            val = std::abs(x[i][k]);
            i_max = i;
        }
    }

    return i_max;
}

template <typename T>
const std::vector<std::vector<T>> Matrix<T>::Cofactor(const std::vector<std::vector<T>> &x)
{
    std::vector<std::vector<T>> R;

    int row = x.size();
    int col = x[0].size();
    T D;

    for (int i = 0; i < row; i++)
    {
        R.push_back(std::vector<T>());
        for (int j = 0; j < col; j++)
        {
            D = Det(CreateSmallerMatrix(x, i, j));
            if (D != 0)
                D *= pow(-1, i + j);

            R[i].push_back(D);
        }
    }
    Cleanup(R);
    return R;
}

template <typename T>
const std::vector<std::vector<T>> Matrix<T>::Adjoin(const std::vector<std::vector<T>> &x)
{
    std::vector<std::vector<T>> R = TransposeMatrix(Cofactor(x));
    Cleanup(R);
    return R;
}

template <typename T>
const std::vector<std::vector<T>> Matrix<T>::Inverse(const std::vector<std::vector<T>> &x)
{
    std::vector<std::vector<T>> R = ScalarMult((1 / GaussElimDet(x)), Adjoin(x));
    Cleanup(R);
    return R;
}

template <typename T>
void Matrix<T>::Cleanup(std::vector<std::vector<T>> &x)
{
    T Epsilon = static_cast<T>(1e-12);

    int sizey = x.size();
    int sizex = x[0].size();

    for (int i = 0; i < sizey; i++)
    {
        for (int j = 0; j < sizex; j++)
        {
            if (std::abs(x[i][j]) < Epsilon)
            {
                x[i][j] = 0;
            }
        }
    }
}

template <typename T>
std::vector<std::vector<T>> Matrix<T>::RowInter(std::vector<std::vector<T>> &x, int a, int b)
{
    std::vector<std::vector<T>> R;
    int row = x.size();
    int col = x[0].size();

    for (int i = 0; i < row; i++)
    {
        R.push_back(std::vector<T>());
        for (int j = 0; j < col; j++)
        {
            if (i == a)
            {
                R[i].push_back(x[b][j]);
            }
            else if (i == b)
            {
                R[i].push_back(x[a][j]);
            }
            else
            {
                R[i].push_back(x[i][j]);
            }
        }
    }
    return R;
}

template <typename T>
std::vector<std::vector<T>> Matrix<T>::RowMult(std::vector<std::vector<T>> &x, int row, int b)
{
    std::vector<std::vector<T>> R;

    int r = x.size();
    int c = x[0].size();

    for (int i = 0; i < r; i++)
    {
        R.push_back(std::vector<T>());
        for (int j = 0; j < c; j++)
        {
            if (i == row)
            {
                R[i].push_back(x[i][j] * b);
            }
            else
            {
                R[i].push_back(x[i][j]);
            }
        }
    }

    return R;
}

template <typename T>
std::vector<std::vector<T>> Matrix<T>::RowDiv(std::vector<std::vector<T>> &x, int row, int b)
{
    std::vector<std::vector<T>> R;

    int r = x.size();
    int c = x[0].size();

    for (int i = 0; i < r; i++)
    {
        R.push_back(std::vector<T>());
        for (int j = 0; j < c; j++)
        {
            if (i == row)
            {
                R[i].push_back(x[i][j] / b);
            }
            else
            {
                R[i].push_back(x[i][j]);
            }
        }
    }

    return R;
}

template <typename T>
std::vector<std::vector<T>> Matrix<T>::RowAdd(std::vector<std::vector<T>> &x, int row1, int row2)
{
    std::vector<std::vector<T>> R;

    int r = x.size();
    int c = x[0].size();

    for (int i = 0; i < r; i++)
    {
        R.push_back(std::vector<T>());
        for (int j = 0; j < c; j++)
        {
            if (i == row1)
            {
                R[i].push_back((x[i][j] + x[row2][j]));
            }
            else
            {
                R[i].push_back(x[i][j]);
            }
        }
    }

    return R;
}

// Time     :   O(row * col) [Worst Case] , O(n)
// Space    :   O((m - 1) * (n - 1))
template <typename T>
const std::vector<std::vector<T>> Matrix<T>::CreateSmallerMatrix(const std::vector<std::vector<T>> &x, int ignx, int igny)
{
    std::vector<std::vector<T>> R;

    int row = x.size();
    int col = x[0].size();

    int y1;
    y1 = 0;

    for (int i = 0; i < row; i++)
    {
        if (ignx != i)
            R.push_back(std::vector<T>());
        for (int j = 0; j < col; j++)
        {
            if ((ignx != i) && (igny != j))
            {
                R[y1].push_back(x[i][j]);
            }
        }
        if (ignx != i)
            y1++;
    }
    return R;
}

template <typename T>
std::vector<std::vector<T>> Matrix<T>::CopyOfMatrix(const std::vector<std::vector<T>> &x)
{
    std::vector<std::vector<T>> R;
    int sizey = x.size();
    int sizex = x[0].size();

    std::vector<std::vector<T>> Idt = this->GetIdentityMatrix(sizey);
    for (int i = 0; i < sizey; i++)
    {
        R.push_back(std::vector<T>());
        for (int j = 0; j < sizex; j++)
        {
            R[i].push_back(x[i][j]);
        }
    }
    return R;
}

template <typename T>
Matrix<T>::~Matrix()
{
}

template <typename T>
void Matrix<T>::printMatrixes(const std::vector<std::vector<T>> &x)
{
    for (std::vector<T> row : x)
    {
        for (T col : row)
        {
            std::cout << col << '\t';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
bool Matrix<T>::TestInverse(const std::vector<std::vector<T>> &x, const std::vector<std::vector<T>> &y)
{
    int y1 = x.size();
    int y2 = y.size();

    if (y1 == 0 || y2 == 0)
        return false;

    int x1 = x[0].size();
    int x2 = y[0].size();

    if (y1 != x1 || y2 != x2)
        return false;

    if (y1 != y2)
        return false;

    std::vector<std::vector<T>> Inv = MultiplyMatrix(x, y);

    for (int i = 0; i < y1; i++)
    {
        for (int j = 0; j < x1; j++)
        {
            if (i == j)
                if (Inv[i][j] != 1)
                    return false;

            if (i != j)
                if (Inv[i][j] != 0)
                    return false;
        }
    }
    return true;
}

#endif
