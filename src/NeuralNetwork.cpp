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

void NeuralNetwork::setLayerWeights(int indexOfLayer, int indexOfPerceptron, std::vector<double> newWeights)
{
    HiddenLayers[indexOfLayer][indexOfPerceptron].setWeights(newWeights);
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
    int lastLayerIndex = this->HiddenLayers.size() - 1;
    std::vector<double> errOfOutpLayer;
    std::vector<std::vector<std::vector<double>>> errOfHiddenLayers; // ErrOfHidden Layer for Layer -> Perceptron -> Weight
    std::vector<double> newWeights;
    std::vector<std::vector<double>> prevErr;

    int mod = 0;

    for (int i = 0; i < epoch; i++)
    {
        if (mod == 4)
            mod = 0;

        this->setInputsLayer(inp[mod]);
        prevErr.clear();
        newWeights.clear();
        errOfHiddenLayers.clear();
        errOfOutpLayer.clear();

        PassThrough();

        int index = 0;
        for (double d : getOutputs())
        {
            d = 0.5 * pow((out[mod][0] - d), 2); // Different LOSS Function
            // d = d * (1 - d) * (out[mod][index] - d);
            index++;
            errOfOutpLayer.push_back(d);
        }
        double Err = errOfOutpLayer[0];
        double currErr;

        for (int i = 0; i < HiddenLayers.size(); i++)
        {
            prevErr.push_back(std::vector<double>());
        }

        for (int indexLayer = lastLayerIndex; indexLayer >= 0; indexLayer--)
        {
            for (int percIdx = 0; percIdx < HiddenLayers[indexLayer].size(); percIdx++) // For now , just one
            {
                newWeights.clear();
                newWeights = HiddenLayers[indexLayer][percIdx].returnWeights(); // new Weights Storage

                for (int weightIdx = 0; weightIdx < HiddenLayers[indexLayer][percIdx].returnWeights().size(); weightIdx++)
                {
                    if (indexLayer == lastLayerIndex) // Only Triggers Once
                    {
                        currErr = Err;
                        double perRes = HiddenLayers[indexLayer - 1][weightIdx].CalcY();
                        double newBias = HiddenLayers[indexLayer][percIdx].returnBias() - 1 * currErr;
                        newWeights[weightIdx] = 1 * currErr * perRes + newWeights[weightIdx];
                        HiddenLayers[indexLayer][percIdx].setBias(newBias);

                        double storeVal = perRes * (1 - perRes) * (newWeights[weightIdx] * currErr);
                        prevErr[indexLayer].push_back(storeVal);
                    }
                    else // Only Triggers Once
                    {
                        currErr = prevErr[indexLayer + 1][percIdx];
                        double perRes = HiddenLayers[indexLayer][weightIdx].CalcY();
                        newWeights[weightIdx] = 1 * currErr * HiddenLayers[indexLayer][percIdx].returnInputs()[weightIdx] + (newWeights[weightIdx]);

                        double storeVal = perRes * (1 - perRes) * (newWeights[weightIdx] * currErr);
                        prevErr[indexLayer].push_back(storeVal);
                        double newBias = HiddenLayers[indexLayer][percIdx].returnBias() - 1 * currErr;
                        HiddenLayers[indexLayer][percIdx].setBias(newBias);
                    }
                }
                HiddenLayers[indexLayer][percIdx].setWeights(newWeights);
            }
        }
        mod++;
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