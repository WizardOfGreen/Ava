#include <iostream>
#include <string>

#include "Matrix.h"
#include "ActFunctions.h"
#include "LossFunctions.h"
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

void SimpleBackPropogation()
{
    NeuralNetwork N;
    N.AssignLossFunction(new RootMSE());
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

    N.setInputsLayer(inp[0]);
    N.PassThrough();

    std::cout << "OUTPUT\n";
    for (double d : N.getOutputs())
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

void MultipleHiddenLayers()
{
    NeuralNetwork N; // 0 Bias and 1 Learning Rate
    N.AssignLossFunction(new MSE()); // Change this later
    std::vector<std::vector<double>> inp = {{0.35, 0.7}};
    std::vector<std::vector<double>> out = {{0.5, 0.7}};

    N.addLayer(2, new Sigmoid()); // Hidden Layer 1
    N.addLayer(2, new Sigmoid()); // Hidden Layer 2
    N.addLayer(2, new Sigmoid()); // Output Layer

    std::vector<double> firstWeights = {0.2, 0.2};  // Hidden Layer 1
    std::vector<double> secondWeights = {0.3, 0.3}; // Hidden Layer 1

    std::vector<double> thirdWeight = {0.3, 0.9};  // Hidden Layer 2
    std::vector<double> fourthWeight = {0.2, 0.1}; // Hidden Layer 2

    std::vector<double> fifthWeight = {0.7, 0.55}; // Output Layer
    std::vector<double> sixthWeight = {0.6, 0.35}; // Output Layer

    N.setInputsLayer(inp[0]);

    N.setLayerWeights(0, 0, firstWeights);
    N.setLayerWeights(0, 1, secondWeights);
    N.setLayerWeights(1, 0, thirdWeight);
    N.setLayerWeights(1, 1, fourthWeight);
    N.setLayerWeights(2, 0, fifthWeight);
    N.setLayerWeights(2, 1, sixthWeight);

    N.PassThrough();

    N.checkWeights() ; 
    N.TrainNN(inp, out, 1);
    N.checkWeights() ;
    
}
void XORTraining()
{
    NeuralNetwork N;
    N.AssignLossFunction(new MSE());
    std::vector<std::vector<double>> inp = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> out = {{0}, {1}, {1}, {0}};

    N.addLayer(2, new Sigmoid());
    N.addLayer(1, new Sigmoid());

    N.setLayerWeights(0, 0, {5.0, 5.0});
    N.setLayerWeights(0, 1, {-5.0, -5.0});
    N.setLayerWeights(1, 0, {7.5, 7.5});

    N.TrainNN(inp, out, 16);
    for (int i = 0; i < 4; i++)
    {
        N.setInputsLayer(inp[i]);
        N.PassThrough();

        std::cout << "Inputs :" << inp[i][0] << " " << inp[i][1] << std::endl;
        std::cout << "Outputs:" << N.getOutputs()[0] << std::endl;
    }
}

void MSETest()
{
    std::cout << " --- MEAN SQUARED --- \n";
    LossFunction *mse = new MSE();
    std::vector<double> trueValues = {10, 20, 30, 40, 50};
    std::vector<double> predValues = {12, 18, 32, 38, 48};
    std::cout << "True Values : ";
    for (double d : trueValues)
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;
    std::cout << "Predicted Values : ";
    for (double d : predValues)
    {
        std::cout << d << " ";
    }

    std::cout << "\n-------------------\n";
    std::cout << "Mean Squared Error : ";
    std::cout << mse->calculateLoss(predValues, trueValues);
    std::cout << "\n-------------------\n";

    std::cout << "Root Mean Squared Error : ";
    std::cout << (new RootMSE())->calculateLoss(predValues, trueValues);
    std::cout << "\n-------------------\n";

    std::cout << "Root Mean Squared Error : ";
    std::cout << (new AbsMSE())->calculateLoss(predValues, trueValues);
    std::cout << "\n-------------------\n";
}

void LossFunctionTests()
{
    MSETest();
}

int main()
{
    // SimpleBackPropogation();
    MultipleHiddenLayers();
    // XORSolved() ;
    // XORTraining();
    // LossFunctionTests();

    std::cout << "Program Ran to Completion\n";
    system("pause");
    system("cls");
    return 0;
}