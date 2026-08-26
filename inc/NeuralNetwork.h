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
    LossFunction * ReturnLossFunction() ; 

    void setInputsLayer(const std::vector<double> &Inp);
    void addLayer(int amount, ActivationFunction *act);
    void setPercInputs(int index, std::vector<double> inp);

    void setLayerWeights(int indexOfLayer, int indexOfPerceptron, std::vector<double> newWeights);
    void setLayerBias(int indexOfLayer, int indexOfPerceptron, double newBias);

    std::vector<double> CalcLayerOutputs(int index);
    
    void FeedForward();
    void TrainNN(const std::vector<std::vector<double>> &inp, const std::vector<std::vector<double>> &out, int epoch, double learning_rate);

    std::vector<double> getOutputs();

};

#endif