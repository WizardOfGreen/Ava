#include <iostream>
#include <iomanip>
#include "Helper.h"
#include "LossFunctions.h"

#include <vector>

int main()
{

    bool failed = false;

    std::vector<double> actual = {10, 20, 30, 40, 50};
    std::vector<double> predicted = {12, 18, 33, 35, 55};

    double final = 3.660601044;

    LossFunction *root_mse = new RootMSE();

    double res = root_mse->calculateLoss(actual, predicted);

    if (compare(final, res))
    {
        std::cout << "Root MSE Test 1 Failed!\n";
        failed = true;
    }
    else
    {
        std::cout << "Root MSE Test 1 Passed!\n";
    }

    return failed;
}