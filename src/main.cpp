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

void XORTraining()
{
    NeuralNetwork N;
    std::vector<std::vector<double>> inp = {{0.35, 0.7}};
    std::vector<std::vector<double>> out = {{0.5}};

    N.addLayer(2, new Sigmoid());
    N.addLayer(1, new Sigmoid());

    std::vector<double> firstWeights = {0.2, 0.2};
    std::vector<double> secondWeights = {0.3, 0.3};
    std::vector<double> thirdWeight = {0.3, 0.9};

    N.setInputsLayer(inp[0]);

    N.setLayerWeights(0, 0, firstWeights);
    N.setLayerWeights(0, 1, secondWeights);
    N.setLayerWeights(1, 0, thirdWeight);

    N.PassThrough();

    std::cout << "OUTPUT\n";
    for (double d : N.getOutputs())
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    N.TrainNN(inp, out, 1000);
    N.PassThrough();

    std::cout << "OUTPUT\n";
    for (double d : N.getOutputs())
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

int main()
{
    XORTraining();
    // XORSolved() ;

    std::cout << "Program Ran to Completion\n";
    system("pause");
    system("cls");
    return 0;
}