#ifndef NEURALNETWORK_CPP
#define NEURALNETWORK_CPP

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::NeuralNetwork(ActivationFunction *act)
{
    this->act = act;
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::setInputsLayer(const std::vector<double> &Inp)
{
    this->InputLayer.clear() ; 
    for (double x : Inp)
    {
        this->InputLayer.push_back(x);
    }
}

void NeuralNetwork::createHiddenLayers(int amount)
{
    for (int i = 0; i < amount; i++)
    {
        this->HiddenLayers.push_back(std::vector<Perceptron>());
    }
}

void NeuralNetwork::PopulateHiddenLayer(int index, int amount)
{
    if (index >= this->HiddenLayers.size())
    {
        std::cout << "Index is larger as Hidden layer\n";
        return;
    }

    for (int i = 0; i < amount; i++)
    {
        this->HiddenLayers[index].push_back(Perceptron(this->act));
    }
}

void NeuralNetwork::PopulateOutputLayer(int amount)
{
    for (int i = 0; i < amount; i++)
    {
        this->OutputLayer.push_back(Perceptron(this->act));
    }
}

void NeuralNetwork::setActivation()
{
    for (int i = 0; i < this->HiddenLayers.size(); i++)
    {
        for (int j = 0; j < this->HiddenLayers[j].size(); j++)
        {
            this->HiddenLayers[i][j].setActFunc(act);
        }
    }
}

void NeuralNetwork::setOutAct()
{
    for (int i = 0; i < this->OutputLayer.size(); i++)
    {
        this->OutputLayer[i].setActFunc(act);
    }
}

void NeuralNetwork::setPercInputs(int index, std::vector<double> inp)
{
    if (index >= this->HiddenLayers.size())
    {
        std::cout << "Index is larger as Hidden layer\n";
        return;
    }

    for (int i = 0; i < this->HiddenLayers[index].size(); i++)
    {
        this->HiddenLayers[index][i].setInp(inp);
    }
}

void NeuralNetwork::setOutputInps()
{
    int last = HiddenLayers.size() - 1;

    for (int i = 0; i < this->OutputLayer.size(); i++)
    {
        this->OutputLayer[i].setInp(CalcLayerOutputs(last));
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
    setOutputInps();
}

std::vector<double> NeuralNetwork::returnOutput()
{
    std::vector<double> out;
    for (int i = 0; i < this->OutputLayer.size(); i++)
    {
        out.push_back(OutputLayer[i].CalcY());
    }

    return out;
}

#endif