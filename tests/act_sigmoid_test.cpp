#include <iostream>
#include <cmath>
#include <vector>

#include "ActFunctions.h"
#include "Helper.h"

int main()
{
    bool failed = false;
    ActivationFunction *sigmoid = new Sigmoid();

    std::vector<double> inp = {0.485, 0.612, 1.2568};

    std::vector<double> out1 = {0, 0, 0};
    std::vector<double> out2 = {0, 0, 0};

    std::vector<double> outact = {0.618927855, 0.648396894, 0.778474754};
    std::vector<double> outder = {0.235856165, 0.227978362, 0.172451811};

    for (int i = 0; i < inp.size(); i++)
    {
        out1[i] = sigmoid->activate(inp[i]);
        out2[i] = sigmoid->activateDerivative(inp[i]);

        if (compare(out1[i], outact[i]))
        {
            std::cout << "Sigmoid Activation " << (i + 1) << " Failed!\n";
            std::cout << "Result : " << out1[i] << " True Val : " << outact[i] << std::endl;
            failed = true;
        }
        else
        {
            std::cout << "Sigmoid Activation " << (i + 1) << " Passed!\n";
        }

        if (compare(out2[i], outder[i]))
        {
            std::cout << "Sigmoid Derivative " << (i + 1) << " Failed!\n";
            std::cout << "Result : " << out2[i] << " True Val : " << outder[i] << std::endl;
            failed = true;
        }
        else
        {
            std::cout << "Sigmoid Derivative " << (i + 1) << " Passed!\n";
        }
        std::cout << std::endl;
    }

    return failed;
}