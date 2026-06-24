#include <iostream>
#include <string>

#include "Matrix.h"
#include "ActFunctions.h"
#include "Perceptron.h"

#include "NeuralNetwork.h"

Perceptron P;

void runEpoch(std::vector<std::vector<double>> inp, std::vector<double> out, int amount)
{
    int i = 0;
    while (i <= amount)
    {
        for (int j = 0; j < 4; j++)
        {
            P.TrainPerceptron(inp[j], out[j]);
        }
        i++;
    }
}

void XORSolved()
{
    NeuralNetwork N;
    std::vector<std::vector<double>> inp = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    N.addLayer(2, new StepFunc());
    N.addLayer(1, new StepFunc());

    N.KnownXORAnswer();

    for (int i = 0; i < 4; i++)
    {
        N.setInputsLayer(inp[i]);
        N.PassThrough();

        std::cout << "OUTPUT\n";
        for (double d : N.getOutputs())
        {
            std::cout << d << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    XORSolved();

    std::cout << "Program Ran to Completion\n";
    system("pause");
    system("cls");
    return 0;
}