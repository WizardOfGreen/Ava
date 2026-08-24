#include <iostream>
#include <cmath>

#include "ActFunctions.h"
#include "Helper.h"

int main()
{
    bool failed = false;
    ActivationFunction *tanh = new TanH();

    double testX = 0.458;

    double testAct = tanh->activate(testX);
    double testDer = tanh->activateDerivative(testX);
    
    double testRes = 0.428452755108;
    double testDerRes = 0.816428236640483200247 ;

    if (compare(testAct, testRes))
    {
        failed = true;
    }

    if (compare(testDer, testDerRes))
    {
        failed = true;
    }

    if ( failed) 
        std::cout << "TanH Test Failed!\n" ;
    else
        std::cout << "TanH Test Passed!\n" ;

    return failed;
}