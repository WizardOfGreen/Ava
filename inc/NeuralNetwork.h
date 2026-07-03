#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Perceptron.h"
#include "LossFunctions.h"
#include <iomanip>
#include <sstream>

class NeuralNetwork
{
private:
    std::vector<double> InputLayer;
    std::vector<std::vector<Perceptron>> HiddenLayers;
    std::vector<ActivationFunction *> Layer_Activations;

    ActivationFunction *act;
    LossFunction *lossFunc;

public:
    NeuralNetwork();
    ~NeuralNetwork();

    void AssignLossFunction(LossFunction *lossFunc);

    void setInputsLayer(const std::vector<double> &Inp);
    void addLayer(int amount, ActivationFunction *act);
    void setPercInputs(int index, std::vector<double> inp);

    void setLayerWeights(int indexOfLayer, int indexOfPerceptron, std::vector<double> newWeights);
    void setLayerBias(int indexOfLayer, int indexOfPerceptron, double newBias);

    std::vector<double> CalcLayerOutputs(int index);
    void FeedForward();

    void TrainNN(const std::vector<std::vector<double>> &inp, const std::vector<std::vector<double>> &out, int epoch, double learning_rate);

    std::vector<double> getOutputs();
    void PrintOutInfo()
    {
        std::cout << std::fixed << std::setprecision(5);

        std::cout << std::left
                  << std::setw(8) << "Layer"
                  << std::setw(10) << "Percep"
                  << std::setw(12) << "Bias"
                  << std::setw(30) << "Weights"
                  << std::setw(15) << "NetOutput"
                  << "ActivatedOutput\n";

        std::cout << std::string(90, '-') << '\n';

        for (int Layeridx = 0; Layeridx < HiddenLayers.size(); Layeridx++)
        {
            for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
            {
                std::vector<double> weights = HiddenLayers[Layeridx][PercIdx].returnWeights();

                std::cout << std::left
                          << std::setw(8) << Layeridx
                          << std::setw(10) << PercIdx
                          << std::setw(12) << HiddenLayers[Layeridx][PercIdx].returnBias();

                // Print weights without a trailing comma
                std::ostringstream weightStream;
                for (size_t i = 0; i < weights.size(); i++)
                {
                    if (i != 0)
                        weightStream << ", ";

                    weightStream << weights[i];
                }

                std::cout << std::setw(30) << weightStream.str()
                          << std::setw(15) << HiddenLayers[Layeridx][PercIdx].returnNet()
                          << HiddenLayers[Layeridx][PercIdx].retAct()
                          << '\n';
            }
        }
    }

    void PrintPredictionInfo(std::vector<std::vector<double>> &inp,
                             std::vector<std::vector<double>> &expected)
    {
        std::cout << std::fixed << std::setprecision(5);

        std::cout << "\n================ Prediction Results ================\n\n";

        for (size_t sample = 0; sample < inp.size(); sample++)
        {
            setInputsLayer(inp[sample]);
            FeedForward();

            std::vector<double> outputs = getOutputs();

            std::cout << "Sample " << sample << '\n';

            // Input
            std::cout << std::setw(18) << std::left << "Input:";
            for (size_t i = 0; i < inp[sample].size(); i++)
            {
                if (i != 0)
                    std::cout << ", ";

                std::cout << inp[sample][i];
            }
            std::cout << '\n';

            // Expected Output
            std::cout << std::setw(18) << std::left << "Expected Output:";
            for (size_t i = 0; i < expected[sample].size(); i++)
            {
                if (i != 0)
                    std::cout << ", ";

                std::cout << expected[sample][i];
            }
            std::cout << '\n';

            // Actual Output
            std::cout << std::setw(18) << std::left << "Network Output:";
            for (size_t i = 0; i < outputs.size(); i++)
            {
                if (i != 0)
                    std::cout << ", ";

                std::cout << outputs[i];
            }
            std::cout << "\n\n";
        }

        std::cout << "====================================================\n";
    }

    void KnownXORAnswer()
    {
        std::vector<double> w1 = {1, 1};
        std::vector<double> v1 = {1, -2};

        HiddenLayers[0][0].setWeights(w1);
        HiddenLayers[0][1].setWeights(w1);

        HiddenLayers[1][0].setWeights(v1);

        HiddenLayers[0][0].setBias(-0.5);
        HiddenLayers[0][1].setBias(-1.5);

        HiddenLayers[1][0].setBias(-0.5);
    }
};

#endif