#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Perceptron.h"
#include "LossFunctions.h"

class NeuralNetwork
{
private:
    /* data */
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
        for (int Layeridx = 0; Layeridx < HiddenLayers.size(); Layeridx++)
        {
            std::cout << "Layer " << Layeridx << std::endl;
            for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
            {
                std::cout << "Perceptron " << PercIdx << std::endl;
                std::cout << "Bias : " << HiddenLayers[Layeridx][PercIdx].returnBias() << std::endl ;
                std::vector<double> newWeights = HiddenLayers[Layeridx][PercIdx].returnWeights();
                for (int WeightIdx = 0; WeightIdx < newWeights.size(); WeightIdx++)
                {
                    std::cout << "Weight " << WeightIdx << " : " << newWeights[WeightIdx] << std::endl;
                }
            }
        }
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