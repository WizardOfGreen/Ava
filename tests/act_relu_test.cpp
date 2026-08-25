#include <iostream>
#include <cmath>
#include <vector>

#include "ActFunctions.h"
#include "Helper.h"

int main()
{
    bool failed = false;
    ActivationFunction *relu = new ReLU();

    std::vector<double> inp = { -3.7, 1} ; 
    std::vector<double> outAct = {0,0,0} ;
    std::vector<double> outDer = {0,0,0} ;  

    std::vector<double> resAct = {0.594113663,-0.652706706,-0.10955847} ;
    std::vector<double> resDer = {0.647028956,0.573973956,0.987996942} ; 

    for ( int i = 0 ; i < inp.size() ; i++)
    {
        outAct[i] = relu->activate( inp[i]) ; 
        outDer[i] = relu->activateDerivative( inp[i]) ; 

        if (compare(outAct[i], resAct[i]))
        {
            std::cout << "ReLu Activation " << (i + 1) << " Failed!\n";
            std::cout << "Result : " << outAct[i] << " True Val : " << resAct[i] << std::endl;
            failed = true;
        }
        else
        {
            std::cout << "ReLu Activation " << (i + 1) << " Passed!\n";
        }

        if (compare(outDer[i], resDer[i]))
        {
            std::cout << "ReLu Derivative " << (i + 1) << " Failed!\n";
            std::cout << "Result : " << outDer[i] << " True Val : " << resDer[i] << std::endl;
            failed = true;
        }
        else
        {
            std::cout << "ReLu Derivative " << (i + 1) << " Passed!\n";
        }

        std::cout << std::endl;
    }

    return failed;
}