#include <iostream>
#include <vector>

#include "Helper.h"
#include "LossFunctions.h"

int main()
{

    bool failed = false;

    std::vector<double> actual = {10, 15, 20, 25};
    std::vector<double> predicted = {12, 14, 23, 20};

    double final = 2.75;

    LossFunction * mae  = new MAE();

    double res = mae->calculateLoss(actual, predicted);

    if (compare(final, res))
    {
        std::cout << "MAE Test 1 Failed!\n";
        failed = true;
    }
    else
    {
        std::cout << "MAE Test 1 Passed!\n";
    }

    return failed;
}