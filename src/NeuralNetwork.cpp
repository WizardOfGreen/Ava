#ifndef NEURALNETWORK_CPP
#define NEURALNETWORK_CPP

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::AssignLossFunction(LossFunction *lossFunc)
{
    this->lossFunc = lossFunc;
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
    this->Layer_Activations.push_back(act);

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

void NeuralNetwork::setLayerWeights(int indexOfLayer, int indexOfPerceptron, std::vector<double> newWeights)
{
    HiddenLayers[indexOfLayer][indexOfPerceptron].setWeights(newWeights);
}

void NeuralNetwork::setLayerBias(int indexOfLayer, int indexOfPerceptron, double newBias)
{
    HiddenLayers[indexOfLayer][indexOfPerceptron].setBias(newBias);
}

std::vector<double> NeuralNetwork::CalcLayerOutputs(int index)
{
    std::vector<double> ret;
    if (index >= this->HiddenLayers.size())
    {
        return ret;
    }

    for (int i = 0; i < HiddenLayers[index].size(); i++)
    {
        ret.push_back(HiddenLayers[index][i].retAct());
    }

    return ret;
}

void NeuralNetwork::FeedForward()
{
    if (this->HiddenLayers.size() <= 0)
    {
        return;
    }

    setPercInputs(0, InputLayer);

    for (int i = 1; i < this->HiddenLayers.size(); i++)
    {
        setPercInputs(i, CalcLayerOutputs(i - 1));
    }
}

void NeuralNetwork::TrainNN(const std::vector<std::vector<double>> &inp, const std::vector<std::vector<double>> &out, int epoch, double learning_rate)
{
    int indexOfLastLayer = HiddenLayers.size() - 1;
    std::vector<double> outputDeltas;
    std::vector<std::vector<double>> hiddenLayerDeltas;
    std::vector<std::vector<std::vector<double>>> weightGradients;
    std::vector<double> outps;

    for (int i = 0; i < epoch; i++)
    {
        for (int outInx = 0; outInx < out.size(); outInx++)
        {
            outps.clear();
            hiddenLayerDeltas.clear() ; 
            weightGradients.clear() ; 
            outputDeltas.clear() ; 

            setInputsLayer(inp[outInx]);
            FeedForward();
            outps = getOutputs();

            double delta = 0;

            // Check This One Later
            for (int idx = 0; idx < out[outInx].size(); idx++)
            {
                delta = (outps[idx] - out[outInx][idx]) * (Layer_Activations[indexOfLastLayer])->Derivative(outps[idx]);
                outputDeltas.push_back(delta);
            }

            for (int Layeridx = 0; Layeridx < HiddenLayers.size(); Layeridx++)
            {
                hiddenLayerDeltas.push_back(std::vector<double>());
                weightGradients.push_back(std::vector<std::vector<double>>());
                for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
                {
                    hiddenLayerDeltas[Layeridx].push_back(double());
                    weightGradients[Layeridx].push_back(std::vector<double>());
                    for (int weightIdx = 0; weightIdx < HiddenLayers[Layeridx][PercIdx].returnWeights().size(); weightIdx++)
                        weightGradients[Layeridx][PercIdx].push_back(double());
                }
            }

            double grad = 0;
            double resPast = 0;

            for (int Layeridx = indexOfLastLayer; Layeridx >= 0; Layeridx--)
            {
                for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
                {
                    if (Layeridx == indexOfLastLayer && indexOfLastLayer != 0) // Only triggers on last Layer and NN is not a single layer
                    {
                        hiddenLayerDeltas[Layeridx][PercIdx] = outputDeltas[PercIdx];
                        for (int WeightIdx = 0; WeightIdx < HiddenLayers[Layeridx][PercIdx].returnWeights().size(); WeightIdx++)
                        {
                            resPast = HiddenLayers[Layeridx - 1][WeightIdx].retAct();
                            weightGradients[Layeridx][PercIdx][WeightIdx] = resPast * outputDeltas[PercIdx];
                        }
                    }
                    else if (indexOfLastLayer == 0) // Triggers if NN is a single Layer
                    {
                        hiddenLayerDeltas[Layeridx][PercIdx] = outputDeltas[PercIdx];

                        for (int WeightIdx = 0; WeightIdx < HiddenLayers[Layeridx][PercIdx].returnWeights().size(); WeightIdx++)
                        {
                            resPast = this->InputLayer[WeightIdx];
                            weightGradients[Layeridx][PercIdx][WeightIdx] = resPast * outputDeltas[PercIdx];
                        }
                    }
                    else
                    {
                        double sum = 0;
                        for (int i = 0; i < HiddenLayers[Layeridx + 1].size(); i++)
                            sum += HiddenLayers[Layeridx + 1][i].returnWeights()[PercIdx] * hiddenLayerDeltas[Layeridx + 1][i];

                        double derivative = this->Layer_Activations[Layeridx]->Derivative(HiddenLayers[Layeridx][PercIdx].retAct());
                        delta = derivative * sum;
                        hiddenLayerDeltas[Layeridx][PercIdx] = delta;
                        int size = ((Layeridx == 0) ? this->InputLayer.size() : HiddenLayers[Layeridx - 1].size());
                        for (int i = 0; i < size; i++)
                        {
                            resPast = ((Layeridx == 0) ? this->InputLayer[i] : HiddenLayers[Layeridx - 1][i].retAct());
                            weightGradients[Layeridx][PercIdx][i] = resPast * delta;
                        }
                    } // If Statements to use specific code blocks for certain Layer Indexes
                } // Perceptron Loop
            } // Layer Loop

            for (int Layeridx = 0; Layeridx <= indexOfLastLayer; Layeridx++)
                for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
                {
                    double oldBias = HiddenLayers[Layeridx][PercIdx].returnBias();
                    double newBias = oldBias - learning_rate * hiddenLayerDeltas[Layeridx][PercIdx];
                    HiddenLayers[Layeridx][PercIdx].setBias(newBias);

                    std::vector<double> newWeights = HiddenLayers[Layeridx][PercIdx].returnWeights();

                    for (int WeightIdx = 0; WeightIdx < newWeights.size(); WeightIdx++)
                    {
                        double oldW = newWeights[WeightIdx];
                        double newWeight = oldW - learning_rate * weightGradients[Layeridx][PercIdx][WeightIdx];
                        newWeights[WeightIdx] = newWeight;
                    }

                    HiddenLayers[Layeridx][PercIdx].setWeights(newWeights);
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
        out.push_back(this->HiddenLayers[last][i].retAct());
    }
    return out;
}

#endif