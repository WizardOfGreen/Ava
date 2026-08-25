#include <iostream>
#include <string>

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
        N.FeedForward();

        std::cout << "OUTPUT\n";
        for (double d : N.getOutputs())
        {
            std::cout << d << " ";
        }
        std::cout << std::endl;
    }
}

void SingleLayerBP()
{
    NeuralNetwork N;
    N.AssignLossFunction(new MSE());
    std::vector<std::vector<double>> inp = {{1.0, 2.0}};
    std::vector<std::vector<double>> out = {{1}};

    N.addLayer(1, new Sigmoid());

    std::vector<double> firstWeights = {0.5, -0.3};
    double b = 0.1;

    N.setInputsLayer(inp[0]);

    N.setLayerWeights(0, 0, firstWeights);
    N.setLayerBias(0, 0, b);

    N.PrintPredictionInfo(inp, out);
    N.PrintOutInfo();

    N.TrainNN(inp, out, 1, 0.1);

    N.PrintPredictionInfo(inp, out);
    N.PrintOutInfo();
}

void MultipleHiddenLayers()
{
    NeuralNetwork N;                 // Learning Rate 0.1 , Default Bias 0.5 on Trainable Bias.
    N.AssignLossFunction(new MSE()); // Change this later
    std::vector<std::vector<double>> inp = {{0.35, 0.7, 0.6}};
    std::vector<std::vector<double>> out = {{0.5, 0.7}};
    // N.getOutputs() -> Returns std::vector<double>

    N.addLayer(2, new Sigmoid()); // Hidden Layer 1
    N.addLayer(2, new Sigmoid()); // Hidden Layer 2
    N.addLayer(2, new Sigmoid()); // Hidden Layer 3

    std::vector<double> firstWeights = {0.2, 0.2, 0, 7}; // Hidden Layer 1
    std::vector<double> secondWeights = {0.3, 0.3, 0.6}; // Hidden Layer 1

    std::vector<double> thirdWeight = {0.3, 0.9};  // Hidden Layer 2
    std::vector<double> fourthWeight = {0.2, 0.1}; // Hidden Layer 2

    std::vector<double> fifthWeight = {0.7, 0.55}; // Output Layer
    std::vector<double> sixthWeight = {0.6, 0.35}; // Output Layer

    N.setLayerWeights(0, 0, firstWeights);
    N.setLayerWeights(0, 1, secondWeights);
    N.setLayerWeights(1, 0, thirdWeight);
    N.setLayerWeights(1, 1, fourthWeight);
    N.setLayerWeights(2, 0, fifthWeight);
    N.setLayerWeights(2, 1, sixthWeight);

    N.PrintPredictionInfo(inp, out);
    N.PrintOutInfo();

    N.TrainNN(inp, out, 5000, 0.1);

    N.PrintPredictionInfo(inp, out);
    N.PrintOutInfo();
}

void XORTraining()
{
    NeuralNetwork N;
    N.AssignLossFunction(new MSE());
    std::vector<std::vector<double>> inp = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> out = {{0}, {1}, {1}, {0}};

    N.addLayer(2, new Sigmoid());
    N.addLayer(1, new Sigmoid());

    N.setLayerWeights(0, 0, {-0.5, 0.3});
    N.setLayerWeights(0, 1, {0.1, -0.7});
    N.setLayerWeights(1, 0, {0.4, -0.2});

    N.setLayerBias(0, 0, -0.2);
    N.setLayerBias(0, 1, 0.6);
    N.setLayerBias(1, 0, 0.7);

    N.TrainNN(inp, out, 10000, 0.1);

    N.PrintPredictionInfo(inp, out);
    N.PrintOutInfo();
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

    std::cout << "MAE Squared Error : ";
    std::cout << (new MAE())->calculateLoss(predValues, trueValues);
    std::cout << "\n-------------------\n";
}

void LossFunctionTests()
{
    MSETest();
}

void PrintFeedForwardResults(NeuralNetwork &N, std::vector<double> &inp1, std::vector<double> &out1)
{
    std::cout << "INPUT : ";
    for (double d : inp1)
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    std::cout << "OUTPUT: ";
    for (double d : N.getOutputs())
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    std::cout << "Expected Output : ";
    for (double d : out1)
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}

void FFTest1()
{
    std::cout << "Linear TEST :\n";
    NeuralNetwork N;
    std::vector<double> inp1 = {2, 3};
    std::vector<double> w1 = {4, 5};
    std::vector<double> out1 = {24};
    double b1 = 1;

    N.addLayer(1, new StepFunc()); // Might not be a Step Function !
    N.setLayerWeights(0, 0, w1);
    N.setLayerBias(0, 0, b1);

    N.setInputsLayer(inp1);
    N.FeedForward();
    PrintFeedForwardResults(N, inp1, out1);
}

void FFTest2()
{
    std::cout << "Sigmoid TEST :\n";
    NeuralNetwork N;
    std::vector<double> inp1 = {1, 2};
    std::vector<double> w1 = {0.5, -0.25};
    std::vector<double> out1 = {0.5249791875};
    double b1 = 0.1;

    N.addLayer(1, new Sigmoid());
    N.setLayerWeights(0, 0, w1);
    N.setLayerBias(0, 0, b1);

    N.setInputsLayer(inp1);
    N.FeedForward();
    PrintFeedForwardResults(N, inp1, out1);
}

void FFTest3()
{
    std::cout << "RELU TEST :\n";
    NeuralNetwork N;
    std::vector<double> inp1 = {-1, 2};
    std::vector<double> w1 = {1, -2};
    std::vector<double> w2 = {-3, 4};
    std::vector<double> w3 = {2, 3};

    std::vector<double> out1 = {28};
    double b1 = 1;
    double b2 = -2;
    double b3 = 1;

    N.addLayer(2, new ReLU());
    N.addLayer(1, new ReLU());

    N.setLayerWeights(0, 0, w1);
    N.setLayerWeights(0, 1, w2);
    N.setLayerWeights(1, 0, w3);

    N.setLayerBias(0, 0, b1);
    N.setLayerBias(0, 1, b2);
    N.setLayerBias(1, 0, b3);

    N.setInputsLayer(inp1);
    N.FeedForward();
    PrintFeedForwardResults(N, inp1, out1);
}

void FFTest4()
{
    std::cout << "Final Sigmoid TEST :\n";
    NeuralNetwork N;
    std::vector<double> inp1 = {0.05, 0.10};
    std::vector<double> w1 = {0.15, 0.20};
    std::vector<double> w2 = {0.25, 0.30};
    std::vector<double> w3 = {0.40, 0.45};
    std::vector<double> w4 = {0.5, 0.55};

    std::vector<double> out1 = {0.751365070, 0.772928465};
    double b1 = 0.35;
    double b2 = 0.35;
    double b3 = 0.6;
    double b4 = 0.6;

    N.addLayer(2, new Sigmoid());
    N.addLayer(2, new Sigmoid());

    N.setLayerWeights(0, 0, w1);
    N.setLayerWeights(0, 1, w2);
    N.setLayerWeights(1, 0, w3);
    N.setLayerWeights(1, 1, w4);

    N.setLayerBias(0, 0, b1);
    N.setLayerBias(0, 1, b2);
    N.setLayerBias(1, 0, b3);
    N.setLayerBias(1, 1, b4);

    N.setInputsLayer(inp1);
    N.FeedForward();
    PrintFeedForwardResults(N, inp1, out1);
}

void FeedForwardTest()
{
    FFTest1();
    FFTest2();
    FFTest3();
    FFTest4();
}

void SigmoidTest()
{
    std::cout << "Sigmoid Activation Test : \n";

    ActivationFunction *act = new Sigmoid();
    std::vector<double> inputs = {-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0};
    std::vector<double> outputs = {0.006692851, 0.119202922, 0.268941421, 0.500000000, 0.731058579, 0.880797078, 0.993307149};
    std::vector<double> derivative = {0.006648057, 0.104993585, 0.196611933, 0.250000000, 0.196611933, 0.104993585, 0.006648057};

    for (int i = 0; i < inputs.size(); i++)
    {
        std::cout << "Input :\t" << inputs[i] << "\tExpected :\t" << outputs[i] << "\tResult :\t" << act->activate(inputs[i]) << std::endl;
    }

    std::cout << "Sigmoid Derivative Test : \n";
    for (int i = 0; i < outputs.size(); i++)
    {
        std::cout << "Input :\t" << outputs[i] << "\tExpected :\t" << derivative[i] << "\tResult :\t" << act->activateDerivative(inputs[i]) << std::endl;
    }
}

void ReluTest()
{
    std::cout << "ReLu Activation Test : \n";

    ActivationFunction *act = new ReLU();
    std::vector<double> inputs = {-10, -1, 0, 1, 5};
    std::vector<double> outputs = {0, 0, 0, 1, 5};
    std::vector<double> derivative = {0, 0, 0, 1, 1};

    for (int i = 0; i < inputs.size(); i++)
    {
        std::cout << "Input :\t" << inputs[i] << "\tExpected :\t" << outputs[i] << "\tResult :\t" << act->activate(inputs[i]) << std::endl;
    }

    std::cout << "ReLu Derivative Test : \n";
    for (int i = 0; i < outputs.size(); i++)
    {
        std::cout << "Input :\t" << outputs[i] << "\tExpected :\t" << derivative[i] << "\tResult :\t" << act->activateDerivative(inputs[i]) << std::endl;
    }
}

void TanHTest()
{
    std::cout << "TanH Activation Test : \n";

    ActivationFunction *act = new TanH();
    std::vector<double> inputs = {-5, -2, -1, 0, 1, 2, 5};
    std::vector<double> outputs = {-0.999909204, -0.964027580, -0.761594156, 0.000000000, 0.761594156, 0.964027580, 0.999909204};
    std::vector<double> derivative = {0.000181575, 0.070650825, 0.419974342, 1.000000000, 0.419974342, 0.070650825, 0.000181575};

    for (int i = 0; i < inputs.size(); i++)
    {
        std::cout << "Input :\t" << inputs[i] << "\tExpected :\t" << outputs[i] << "\tResult :\t" << act->activate(inputs[i]) << std::endl;
    }

    std::cout << "TanH Derivative Test : \n";
    for (int i = 0; i < outputs.size(); i++)
    {
        std::cout << "Input :\t" << outputs[i] << "\tExpected :\t" << derivative[i] << "\tResult :\t" << act->activateDerivative(inputs[i]) << std::endl;
    }
}

void SoftMaxTest()
{
}

void ActivationTests()
{
    SigmoidTest();
    ReluTest();
    TanHTest();
    SoftMaxTest();
}

void singlePerceptronTest()
{
    Perceptron P;
    std::vector<double> inp = {0.790009, 0.668941};
    std::vector<double> w = {0.6, 0.35};
    double b = 0.5;
    P.setActFunc(new Sigmoid());
    P.setWeights(w);
    P.setInp(inp);
    P.setBias(b);

    double exp = (inp[0] * w[0]) + (inp[1] * w[1]) + (1 * b);

    std::cout << "Inputs : " << inp[0] << " " << inp[1] << std::endl;
    std::cout << "OutputActivation : " << P.retAct() << std::endl;
    std::cout << "OutputNet : " << P.returnNet() << std::endl;
    std::cout << "Expected Output : " << exp << std::endl;
}

int main()
{
    // =======================================
    // Neural Networks :
    // =======================================
    // SingleLayerBP();                         // 1 Layer , 2 Inputs , 2 Output
    // MultipleHiddenLayers();                     // 3 Layers , 3 Inputs , 2 Outputs
    XORTraining();                           // 1 Layers , 2 Inputs ,1 Output.
    // XORSolved() ;                            // Solved Problem To Test Feed Forwarding.
    // FeedForwardTest();                       // Tests Feed Forward Pass
    // =======================================
    // Function Tests :
    // =======================================
    // LossFunctionTests();
    // ActivationTests();
    // =======================================
    // Perceptron Tests :
    // Perceptron P ;
    // P.setInp( { 1 , 0 } ) ;
    // std::cout << "Perceptron Test\n" ;

    // =======================================
    // singlePerceptronTest() ;
    // =======================================

    std::cout << "Program Ran Successfully\n";

    return 0;
}