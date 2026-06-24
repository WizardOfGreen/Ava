#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Perceptron.h"

class NeuralNetwork
{
private:
    /* data */
    std::vector<double> InputLayer;
    std::vector<std::vector<Perceptron>> HiddenLayers;

    ActivationFunction *act;

public:
    NeuralNetwork();
    ~NeuralNetwork();

    void setInputsLayer(const std::vector<double> &Inp);
    void addLayer(int amount , ActivationFunction * act) ; 
    void setPercInputs(int index, std::vector<double> inp);

    std::vector<double> CalcLayerOutputs(int index);
    void PassThrough();

    void TrainNN(std::vector<std::vector<double>> inp , std::vector<std::vector<double>> out , int epoch ) ; 

    std::vector<double> getOutputs() ; 

    void CheckHiddenLayerOutputs()
    {
        for (int i = 0; i < this->HiddenLayers.size(); i++)
        {
            for (int j = 0; j < this->HiddenLayers[i].size(); j++)
            {
                std::cout << "OUTPUT OF "<< i << " HIDDEN LAYER AND PERCEPTRON " << j << std::endl ;
                HiddenLayers[i][j].printInp(); 
                HiddenLayers[i][j].printY();
                std::cout << "END OF THIS PERCEPTON\n";
            }
        }
    }

    void KnownXORAnswer()
    {
        std::vector<double> w1 = {1, 1};
        std::vector<double> v1 = {1 , -2};

        HiddenLayers[0][0].setWeights(w1);
        HiddenLayers[0][1].setWeights(w1);

        HiddenLayers[1][0].setWeights(v1);

        HiddenLayers[0][0].setBias(-0.5);
        HiddenLayers[0][1].setBias(-1.5);

        HiddenLayers[1][0].setBias(-0.5);
    }
};

#endif