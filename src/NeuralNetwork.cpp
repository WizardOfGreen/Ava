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
        std::cout << "Index is larger as Hidden layer\n";
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
        std::cout << "No Hidden Layers Detected\n";
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
    for (int i = 0; i < epoch; i++)
    {
        std::vector<double> outps;
        for (int outInx = 0; outInx < out.size(); outInx++)
        {
            setInputsLayer(inp[outInx]);
            FeedForward();
            outps = getOutputs(); // Returns the Activated Output i.e Predicted y.

            std::vector<double> outputDeltas;
            std::vector<std::vector<double>> hiddenLayerDeltas;
            std::vector<std::vector<std::vector<double>>> weightGradients;

            double delta = 0;

            // std::cout << "LOSS RESULTS : " << this->lossFunc->calculateLoss2(outps[0], out[outInx][0]) << std::endl;

            for (int idx = 0; idx < out[outInx].size(); idx++)
            {
                delta = (outps[idx] - out[outInx][idx]) * (Layer_Activations[indexOfLastLayer])->Derivative(outps[idx]); // Works
                outputDeltas.push_back(delta);                                                                           // Delta for Output Perceptron 0 and 1
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
                    {
                        weightGradients[Layeridx][PercIdx].push_back(double());
                    }
                }
            }

            double Err = 0;
            double grad;
            double resPast;
            double sum = 0;

            for (int Layeridx = indexOfLastLayer; Layeridx >= 0; Layeridx--)
            {
                for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
                {
                    for (int WeightIdx = 0; WeightIdx < HiddenLayers[Layeridx][PercIdx].returnWeights().size(); WeightIdx++)
                    {
                        if (indexOfLastLayer == 0) // Works in One Tested Case
                        {
                            double Inp = this->InputLayer[WeightIdx];
                            double w = HiddenLayers[Layeridx][PercIdx].returnWeights()[WeightIdx];
                            double Err = Inp * outputDeltas[PercIdx];

                            hiddenLayerDeltas[Layeridx][PercIdx] = outputDeltas[PercIdx];
                            weightGradients[Layeridx][PercIdx][WeightIdx] = Err;
                        }
                        else if (Layeridx == indexOfLastLayer) // EDGE CASE : Hidden layer is Output Layer
                        {
                            double PerRes = HiddenLayers[Layeridx - 1][WeightIdx].retAct(); // Return Activated Value
                            double Error = outputDeltas[PercIdx];
                            hiddenLayerDeltas[Layeridx][PercIdx] = Error;
                            double Err = PerRes * Error; // This is the Gradient
                            weightGradients[Layeridx][PercIdx][WeightIdx] = Err;
                        }
                        else if (Layeridx != 0) // EDGE CASE : Hidden layer is in the middle of Output and First Input Layer
                        {

                            double act = HiddenLayers[Layeridx][PercIdx].returnNet();
                            double d = this->Layer_Activations[Layeridx]->activate(act);
                            double derivative = this->Layer_Activations[Layeridx]->Derivative(d);

                            for (int i = 0; i < HiddenLayers[Layeridx + 1].size(); i++)
                            {
                                sum = sum + HiddenLayers[Layeridx + 1][i].returnWeights()[PercIdx] * hiddenLayerDeltas[Layeridx + 1][i];
                            }
                            delta = derivative * sum;
                            hiddenLayerDeltas[Layeridx][PercIdx] = delta;

                            for (int i = 0; i < HiddenLayers[Layeridx - 1].size(); i++)
                            {
                                resPast = HiddenLayers[Layeridx - 1][i].retAct();
                                grad = resPast * delta;
                                weightGradients[Layeridx][PercIdx][i] = grad;
                            }
                        }
                        else if (Layeridx == 0) // EDGE CASE : Hidden layer is after Input Layer
                        {
                            double act = HiddenLayers[Layeridx][PercIdx].returnNet();
                            double d = this->Layer_Activations[Layeridx]->activate(act);
                            double derivative = this->Layer_Activations[Layeridx]->Derivative(d);

                            for (int i = 0; i < HiddenLayers[Layeridx + 1].size(); i++)
                            {
                                sum = sum + HiddenLayers[Layeridx + 1][i].returnWeights()[PercIdx] * hiddenLayerDeltas[Layeridx + 1][i];
                            }
                            delta = derivative * sum;
                            hiddenLayerDeltas[Layeridx][PercIdx] = delta;

                            for (int i = 0; i < this->InputLayer.size(); i++)
                            {
                                resPast = this->InputLayer[i];
                                grad = resPast * delta;
                                weightGradients[Layeridx][PercIdx][i] = grad;
                            }
                        }
                    }
                }
            } // End of ERROR COMPUTING


            for (int Layeridx = 0; Layeridx <= indexOfLastLayer; Layeridx++)
            {
                for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
                {
                    double oldBias = HiddenLayers[Layeridx][PercIdx].returnBias();
                    double newBias = oldBias - learning_rate * hiddenLayerDeltas[Layeridx][PercIdx];

                    HiddenLayers[Layeridx][PercIdx].setBias(newBias);
                    std::vector<double> newWeights = HiddenLayers[Layeridx][PercIdx].returnWeights();

                    for (int WeightIdx = 0; WeightIdx < newWeights.size(); WeightIdx++)
                    {
                        int weightCount = newWeights.size();
                        double oldW = newWeights[WeightIdx];
                        double newWeight = oldW - learning_rate * weightGradients[Layeridx][PercIdx][WeightIdx];
                        newWeights[WeightIdx] = newWeight;
                    }

                    HiddenLayers[Layeridx][PercIdx].setWeights(newWeights);
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
        out.push_back(this->HiddenLayers[last][i].retAct());
    }
    return out;
}

#endif