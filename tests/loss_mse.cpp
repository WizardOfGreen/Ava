#include <iostream>
#include "Helper.h"
#include "LossFunctions.h"

#include <vector>

int main()
{

    bool failed = false;

    std::vector<double> actual = {3, 5, 2, 7};
    std::vector<double> predicted = {2, 4, 3, 8};

    double final = 1;

    LossFunction *mse = new MSE();

    double res = mse->calculateLoss(actual, predicted);

    if (compare(final, res))
    {
        std::cout << "MSE Test 1 Failed!\n";
        failed = true;
    }
    else
    {
        std::cout << "MSE Test 1 Passed!\n";
    }

    actual = {10, 20, 30, 40, 50};
    predicted = {12, 18, 33, 35, 55};
    final = 13.4;
    res = mse->calculateLoss(actual, predicted);

    if (compare(final, res))
    {
        std::cout << "MSE Test 2 Failed!\n";
        failed = true;
    }
    else
    {
        std::cout << "MSE Test 2 Passed!\n";
    }

    return failed;
}