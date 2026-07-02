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
        std::cout << "Layer\tPercep\tBias\t\tWeight\t\t\tNetOutput\tActivatedOutput\t\n";
        for (int Layeridx = 0; Layeridx < HiddenLayers.size(); Layeridx++)
        {
            std::cout << Layeridx << "\t";
            for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
            {
                std::cout << PercIdx << "\t";
                std::vector<double> newWeights = HiddenLayers[Layeridx][PercIdx].returnWeights();

                std::cout << HiddenLayers[Layeridx][PercIdx].returnBias() << "\t";
                for (int WeightIdx = 0; WeightIdx < newWeights.size(); WeightIdx++)
                {
                    std::cout << newWeights[WeightIdx] << ",";
                }
                std::cout << "\t";
                std::cout << HiddenLayers[Layeridx][PercIdx].returnNet() << "\t";
                std::cout << HiddenLayers[Layeridx][PercIdx].retAct() ;
                std::cout << std::endl << "\t" ;
            }
            std::cout << std::endl ; 
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