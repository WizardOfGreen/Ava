#include <iostream>
#include <cmath>
#include <vector>

#include "ActFunctions.h"
#include "Helper.h"

int main()
{
    bool failed = false;
    ActivationFunction *softplus = new SoftPlus();

    std::vector<double> inp = {0, 2, -4};

    std::vector<double> out1 = {0, 0, 0};
    std::vector<double> out2 = {0, 0, 0};

    std::vector<double> outact = {0.693147181, 2.126928011, 0.018149928};
    std::vector<double> outder = {0.5, 0.880797078, 0.01798621};

    for (int i = 0; i < inp.size() ; i++)
    {
        out1[i] = softplus->activate(inp[i]);
        out2[i] = softplus->activateDerivative(inp[i]);

        if (compare(out1[i], outact[i]))
        {
            std::cout << "Sofplus Activation " << (i + 1) << " Failed!\n";
            std::cout << "Result : " << out1[i] << " True Val : " << outact[i] << std::endl ; 
            failed = true;
        }
        else
        {
            std::cout << "Sofplus Activation " << (i + 1) << " Passed!\n";
        }
         
        if (compare(out2[i], outder[i]))
        {
            std::cout << "Sofplus Derivative " << (i + 1) << " Failed!\n";
            std::cout << "Result : " << out2[i] << " True Val : " << outder[i] << std::endl ; 
            failed = true;
        }
        else
        {
            std::cout << "Sofplus Derivative " << (i + 1) << " Passed!\n";
        }
        std::cout << std::endl ;
    }

    return failed;
}