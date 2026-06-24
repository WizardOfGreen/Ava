#ifndef NEURALNETWORK_CPP
#define NEURALNETWORK_CPP

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::setInputsLayer(const std::vector<double> &Inp)
{
    this->InputLayer.clear();
    for (double x : Inp)
    {
        this->InputLayer.push_back(x);
    }
}

void NeuralNetwork::addLayer(int amount, ActivationFunction *act)
{
    this->HiddenLayers.push_back(std::vector<Perceptron>());

    int last = HiddenLayers.size() - 1;

    for (int i = 0; i < amount; i++)
    {
        HiddenLayers[last].push_back(Perceptron(act));
    }
}

void NeuralNetwork::setPercInputs(int index, std::vector<double> inp)
{
    for (int i = 0; i < this->HiddenLayers[index].size(); i++)
    {
        this->HiddenLayers[index][i].setInp(inp);
    }
}

std::vector<double> NeuralNetwork::CalcLayerOutputs(int index)
{
    std::vector<double> ret;
    if (index >= this->HiddenLayers.size())
    {
        std::cout << "Index is larger as Hidden layer\n";
        return ret;
    }

    for (int i = 0; i < HiddenLayers[index].size(); i++)
    {
        ret.push_back(HiddenLayers[index][i].CalcY());
    }

    return ret;
}

void NeuralNetwork::PassThrough()
{
    if (this->HiddenLayers.size() <= 0)
    {
        std::cout << "No Hidden Layers Detected\n";
        return;
    }

    setPercInputs(0, InputLayer);

    for (int i = 1; i < this->HiddenLayers.size(); i++)
    {
        setPercInputs(i, CalcLayerOutputs(i - 1));
    }
}

void NeuralNetwork::TrainNN(std::vector<std::vector<double>> inp, std::vector<std::vector<double>> out, int epoch)
{
    double error;
    int last ; 
    double learningRate = 0.05 ;  
    std::vector<double> outp ; 

    int VAL = 0 ; // CHANGE THIS LATER
    double TEST ; 
    double percOut ; 
    for (int i = 0; i < epoch; i++)
    {
        for (int j = 0; j < out.size(); j++)
        {
            setInputsLayer(inp[j]) ; 
            PassThrough();
            outp = getOutputs() ; 
            error = out[j][VAL] - outp[VAL]  ;

            last = HiddenLayers.size() - 1 ;
            TEST = outp[VAL] * ( 1 - outp[VAL]) * (error ) ;
            

            for ( int i = HiddenLayers.size() - 2 ; i >= 0 ; i-- )
            {
                for ( int j = 0 ; j < HiddenLayers[i].size() ; j++)
                {
                    for ( int k = 0 ; k < HiddenLayers[i][j].returnWeights().size() ; k++ )
                    {

                    }
                }
            }

        }
    }
}

std::vector<double> NeuralNetwork::getOutputs()
{
    std::vector<double> out;
    int last = this->HiddenLayers.size() - 1;
    for (int i = 0; i < this->HiddenLayers[last].size(); i++)
    {
        out.push_back(this->HiddenLayers[last][i].CalcY());
    }
    return out;
}

#endif