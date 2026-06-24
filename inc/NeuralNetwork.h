#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Perceptron.h"

class NeuralNetwork
{
private:
    /* data */
    std::vector<double> InputLayer;
    std::vector<std::vector<Perceptron>> HiddenLayers;
    std::vector<Perceptron> OutputLayer;

    ActivationFunction *act;

public:
    NeuralNetwork();
    NeuralNetwork(ActivationFunction *act);
    ~NeuralNetwork();

    void setInputsLayer(const std::vector<double> &Inp);
    void createHiddenLayers(int amount);

    void PopulateHiddenLayer(int index, int amount);
    void PopulateOutputLayer(int amount);

    void setActivation();
    void setOutAct();

    void setPercInputs(int index, std::vector<double> inp);
    void setOutputInps();

    std::vector<double> CalcLayerOutputs(int index);
    void PassThrough();

    std::vector<double> returnOutput();

    void CheckHiddenLayerOutputs()
    {
        for (int i = 0; i < this->HiddenLayers.size(); i++)
        {
            for (int j = 0; j < this->HiddenLayers[i].size(); j++)
            {
                std::cout << "OUTPUT OF HIDDEN LAYER \n";
                HiddenLayers[i][j].printInp(); 
                HiddenLayers[i][j].printOut();
                HiddenLayers[i][j].printY();
                std::cout << "END OF THIS PERCEPTON\n";
            }
        }
    }

    void checkOutputLayer()
    {
        std::cout << "CHECKING OUTPUT LAYER\n" ; 
        for ( int i = 0 ; i < OutputLayer.size() ; i++)
        {
            OutputLayer[i].printOut() ; 
            OutputLayer[i].printY() ;

            OutputLayer[i].printInp() ; 
        }
        std::cout << "CONCLUDS OUTPUT INFO\n" ; 
    }

    void KnownXORAnswer()
    {
        std::vector<double> w1 = {1, 1};
        std::vector<double> v1 = {1 , -2};

        HiddenLayers[0][0].setWeights(w1);
        HiddenLayers[0][1].setWeights(w1);
        OutputLayer[0].setWeights(v1);

        HiddenLayers[0][0].setBias(-0.5);
        HiddenLayers[0][1].setBias(-1.5);
        OutputLayer[0].setBias(-0.5);
    }
};

#endif