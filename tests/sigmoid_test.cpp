#include <iostream>
#include <cmath>

#include "ActFunctions.h"
#include "Helper.h"

int main()
{
    bool failed = false;
    ActivationFunction *sigmoid = new Sigmoid();

    double testX = 0.458;

    double testAct = sigmoid->activate(testX);
    double testDer = sigmoid->activateDerivative(testX);

    double testRes = 0.6125396134409533;
    double testDerRes = 0.2373348354065608;

    if (compare(testAct, testRes))
    {
        failed = true;
    }

    if (compare(testDer, testDerRes))
    {
        failed = true;
    }

    if (failed)
        std::cout << "Sigmoid Test Failed!\n";
    else
        std::cout << "Sigmoid Test Passed!\n";

    return failed;
}