#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>

template <typename T>
class Matrix
{
private:
public:
    Matrix();
    ~Matrix();

    std::vector<std::vector<T>> GetIdentityMatrix(int x);
    const std::vector<std::vector<T>> TransposeMatrix(const std::vector<std::vector<T>> &x); // Error Coded

    const std::vector<std::vector<T>> MultiplyMatrix(const std::vector<std::vector<T>> &x, const std::vector<std::vector<T>> &y); // Error Coded
    const std::vector<std::vector<T>> ScalarMult(T s, const std::vector<std::vector<T>> &x);                                      // Error Coded

    const T Det(const std::vector<std::vector<T>> &x);                                // Error Coded
    const T GaussElimDet(const std::vector<std::vector<T>> &x);                       // Error Coded
    int MaxH(int h, int k, const std::vector<std::vector<T>> &x);                     // Error Coded
    const std::vector<std::vector<T>> Cofactor(const std::vector<std::vector<T>> &x); // Error COded

    const std::vector<std::vector<T>> Adjoin(const std::vector<std::vector<T>> &x);  // Error Coded
    const std::vector<std::vector<T>> Inverse(const std::vector<std::vector<T>> &x); // Error COded

    std::vector<std::vector<T>> RowInter(std::vector<std::vector<T>> &x, int a, int b);     // Error Coded
    std::vector<std::vector<T>> RowMult(std::vector<std::vector<T>> &x, int row, int b);    // Error Coded
    std::vector<std::vector<T>> RowDiv(std::vector<std::vector<T>> &x, int row, int b);     // Error Coded
    std::vector<std::vector<T>> RowAdd(std::vector<std::vector<T>> &x, int row1, int row2); // Error Coded

    const std::vector<std::vector<T>> CreateSmallerMatrix(const std::vector<std::vector<T>> &x, int ignx, int igny); // Error COded
    std::vector<std::vector<T>> CopyOfMatrix(const std::vector<std::vector<T>> &x);                                  // Error Coded
    void Cleanup(std::vector<std::vector<T>> &x);                                                                    // Error COded
    void printMatrixes(const std::vector<std::vector<T>> &x);                                                        // Error Coded

    bool TestInverse(const std::vector<std::vector<T>> &x , const std::vector<std::vector<T>> &y) ; 
    

    // UNIT TESTS

#define IS_TRUE(x)                                                                    \
    {                                                                                 \
        if (!(x))                                                                     \
            std::cout << "" __FUNCTION__ << " has failed on " << __LINE << std::endl; \
    };
};

#endif