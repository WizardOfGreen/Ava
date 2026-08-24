#include <iostream>
#include <cmath>

#include "ActFunctions.h"
#include "Helper.h"

int main()
{
    bool failed = false;
    ActivationFunction *relu = new ReLU();

    double testX = 0.458;

    double testAct = relu->activate(testX);
    double testDer = relu->activateDerivative(testX);

    double testRes = 0.458;
    double testDerRes = 1;

    if (compare(testAct, testRes))
    {
        failed = true;
    }

    if (compare(testDer, testDerRes))
    {
        failed = true;
    }

    if (failed)
        std::cout << "RelU Test Failed!\n";
    else
        std::cout << "RelU Test Passed!\n";

    return failed;
}