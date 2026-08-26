#include <iostream>
#include <vector>

#include "NeuralNetwork.h"
#include "Helper.h"

bool GetAndSet()
{
    NeuralNetwork N;
    LossFunction *lossFunc = new MSE();
    LossFunction *tester = new MSE();

    LossFunction *getter;

    N.AssignLossFunction(lossFunc);
    getter = N.ReturnLossFunction();

    if (typeid(getter) != typeid(tester))
        return false;

    std::vector<double> inp = {0, 1, 1, 0};
    N.setInputsLayer(inp);

    return true;
}

bool SingleLayerTest()
{
    NeuralNetwork N;
    N.AssignLossFunction(new MSE());
    std::vector<std::vector<double>> inp = {{1.0, 2.0}};
    std::vector<std::vector<double>> out = {{1}};

    N.addLayer(1, new Sigmoid());

    std::vector<double> firstWeights = {0.5, -0.3};
    double b = 0.1;

    std::vector<double> firstOut = N.getOutputs();

    N.setLayerWeights(0, 0, firstWeights);
    N.setLayerBias(0, 0, b);

    N.TrainNN(inp, out, 200, 0.1);

    std::vector<double> res = N.getOutputs();

    for (int i = 0; i < res.size(); i++)
    {
        if (firstOut[i] > res[i])
        {
            return false;
        }
    }

    return true;
}

bool MultipleLayerTest()
{
    NeuralNetwork N;
    N.AssignLossFunction(new MSE());
    std::vector<std::vector<double>> inp = {{0.35, 0.7, 0.6}};
    std::vector<std::vector<double>> out = {{0.5, 0.7}};

    N.addLayer(2, new Sigmoid());
    N.addLayer(2, new Sigmoid());
    N.addLayer(2, new Sigmoid());

    std::vector<double> firstWeights = {0.2, 0.2, 0, 7};
    std::vector<double> secondWeights = {0.3, 0.3, 0.6};

    std::vector<double> thirdWeight = {0.3, 0.9};
    std::vector<double> fourthWeight = {0.2, 0.1};

    std::vector<double> fifthWeight = {0.7, 0.55};
    std::vector<double> sixthWeight = {0.6, 0.35};

    N.setLayerWeights(0, 0, firstWeights);
    N.setLayerWeights(0, 1, secondWeights);
    N.setLayerWeights(1, 0, thirdWeight);
    N.setLayerWeights(1, 1, fourthWeight);
    N.setLayerWeights(2, 0, fifthWeight);
    N.setLayerWeights(2, 1, sixthWeight);

    std::vector<double> firstOut = N.getOutputs();
    N.TrainNN(inp, out, 5000, 0.1);
    std::vector<double> afterTrainOut = N.getOutputs();

    for (int i = 0; i < out.size(); i++)
    {
        for (int j = 0; j < out[i].size(); j++)
        {
            if (compare(afterTrainOut[j], out[i][j]))
            {
                return false;
            }
        }
    }

    return true;
}

bool XORTrainTest()
{
    NeuralNetwork N;
    N.AssignLossFunction(new MSE());
    std::vector<std::vector<double>> inp = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> out = {{0}, {1}, {1}, {0}};
    std::vector<double> outExp = {0.1, 0.9, 0.9, 0.1};
    std::vector<double> out1 = {0};

    N.addLayer(2, new Sigmoid());
    N.addLayer(1, new Sigmoid());

    N.setLayerWeights(0, 0, {-0.5, 0.3});
    N.setLayerWeights(0, 1, {0.1, -0.7});
    N.setLayerWeights(1, 0, {0.4, -0.2});

    N.setLayerBias(0, 0, -0.2);
    N.setLayerBias(0, 1, 0.6);
    N.setLayerBias(1, 0, 0.7);

    N.TrainNN(inp, out, 10000, 0.1);

    for (int i = 0; i < 4; i++)
    {
        N.setInputsLayer(inp[i]);
        out1 = N.getOutputs();

        if ((i == 0 || i == 3) && out1[0] >= outExp[i])
            return false;

        if ((i == 1 || i == 2) && out1[0] <= outExp[i])
            return false;
    }

    return true;
}

int main()
{
    if (SingleLayerTest())
    {
        std::cout << "Single Layer Training Test Passed!\n";
    }
    else
    {
        std::cout << "Single Layer Training Test Failed!\n";
        return 1;
    }

    if (MultipleLayerTest())
    {
        std::cout << "Multiple Layer Training Test Passed!\n";
    }
    else
    {
        std::cout << "Multiple Layer Training Test Failed!\n";
        return 1;
    }

    if (XORTrainTest())
    {
        std::cout << "XOR Training Test Passed!\n";
    }
    else
    {
        std::cout << "XOR Training Test Failed!\n";
        return 1;
    }

    if (GetAndSet())
    {
        std::cout << "Getter&Setter Test Passed!\n";
    }
    else
    {
        std::cout << "Getter&Setter Test Failed!\n";
        return 1;
    }

    return 0;
}