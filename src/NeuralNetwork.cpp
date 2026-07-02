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
            outps = getOutputs();

            std::vector<double> outputDeltas;
            std::vector<std::vector<double>> hiddenLayerDeltas;
            std::vector<std::vector<double>> hiddenLayerGradients;

            double delta = 0;

            std::cout << "LOSS RESULTS : " << this->lossFunc->calculateLoss2(outps[0], out[outInx][0]) << std::endl;

            for (int idx = 0; idx < out[outInx].size(); idx++)
            {
                delta = (outps[idx] - out[outInx][idx]) * (Layer_Activations[(HiddenLayers.size() - 1)])->Derivative(outps[idx]); // Works
                outputDeltas.push_back(delta);
            }

            for (int Layeridx = 0; Layeridx < HiddenLayers.size(); Layeridx++)
            {
                hiddenLayerGradients.push_back(std::vector<double>());
                hiddenLayerDeltas.push_back(std::vector<double>());
                for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
                {
                    hiddenLayerDeltas[Layeridx].push_back(double());
                }
            }

            for (int Layeridx = indexOfLastLayer; Layeridx >= 0; Layeridx--)
            {
                for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
                {
                    for (int WeightIdx = 0; WeightIdx < HiddenLayers[Layeridx][PercIdx].returnWeights().size(); WeightIdx++)
                    {
                        if (indexOfLastLayer == 0) // Works in One Case
                        {
                            double Inp = this->InputLayer[WeightIdx];
                            double Err = 0;

                            double w = HiddenLayers[Layeridx][PercIdx].returnWeights()[WeightIdx];
                            Err = Inp * outputDeltas[PercIdx];
                            hiddenLayerDeltas[Layeridx][PercIdx] = outputDeltas[PercIdx];

                            hiddenLayerGradients[Layeridx].push_back(Err);
                        }
                        else if (Layeridx == indexOfLastLayer) // EDGE CASE : Hidden layer is Output Layer
                        {
                            double PerRes = HiddenLayers[Layeridx - 1][WeightIdx].retAct(); // Return Activated Value
                            double Error = outputDeltas[PercIdx];
                            hiddenLayerDeltas[Layeridx][PercIdx] = Error;
                            double Err = PerRes * Error; // This is the Gradient
                            hiddenLayerGradients[Layeridx].push_back(Err);
                        }
                        else if (Layeridx != 0) // EDGE CASE : Hidden layer is in the middle of Output and First Input Layer
                        {
                            double Err = 0;

                            for (int i = 0; i < hiddenLayerDeltas[Layeridx + 1].size(); i++)
                            {
                                double w = HiddenLayers[Layeridx + 1][i].returnWeights()[PercIdx];
                                Err += w * hiddenLayerDeltas[Layeridx + 1][i];
                            }
                            double PerRes = HiddenLayers[Layeridx][PercIdx].retAct(); // CalcY Return Activated Value

                            double der = (Layer_Activations[Layeridx])->activateDerivative(PerRes);
                            Err = Err * der;
                            hiddenLayerDeltas[Layeridx][PercIdx] = Err;
                            PerRes = HiddenLayers[Layeridx - 1][WeightIdx].retAct();
                            Err = Err * PerRes;
                            hiddenLayerGradients[Layeridx].push_back(Err);
                        }
                        else if (Layeridx == 0) // EDGE CASE : Hidden layer is after Input Layer
                        {
                            double PerRes = HiddenLayers[Layeridx][PercIdx].retAct();
                            double Err = 0;

                            for (int i = 0; i < hiddenLayerDeltas[Layeridx + 1].size(); i++)
                            {
                                double w = HiddenLayers[Layeridx + 1][i].returnWeights()[PercIdx];
                                Err += w * hiddenLayerDeltas[Layeridx + 1][i];
                            }

                            double der = (Layer_Activations[Layeridx])->activateDerivative(PerRes);
                            Err = Err * der;
                            hiddenLayerDeltas[Layeridx][PercIdx] = Err;
                            PerRes = this->InputLayer[WeightIdx];
                            Err = Err * PerRes;
                            hiddenLayerGradients[Layeridx].push_back(Err);
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
                        double newWeight = oldW - learning_rate * hiddenLayerGradients[Layeridx][PercIdx * weightCount + WeightIdx];
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