#include <iostream>
#include <vector>
#include "Helper.h"

#include "ActFunctions.h"

int main()
{
    bool failed = false; 
    std::vector<double> inp = {0.5, -0.5};
    std::vector<double> out = {0, 0};
    std::vector<double> res = {1, 0};

    ActivationFunction * stepfunc = new StepFunc ; 

    for ( int i = 0 ; i < inp.size() ; i++)
    {
        out[i] = stepfunc->activate(inp[i]) ; 

        if ( compare(out[i] , res[i]))
        {
            std::cout << "Step Function " << (i+1) << " Failed!\n" ;
            std::cout << "Result : " << out[i] << " True Value : " << res[i] << std::endl ;
            failed = true ; 
        }
        else
        {
            std::cout << "Step Function " << (i+1) << " Passed!\n" ;
        }

    }

    return failed;
}