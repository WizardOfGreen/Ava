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
    int indexOfLastLayer = HiddenLayers.size() - 1;
    for (int i = 0; i < epoch; i++)
    {
        std::vector<std::vector<double>> Errors;
        std::vector<std::vector<double>> Ouputs;
        for (int outInx = 0; outInx < out.size(); outInx++)
        {
            Errors.push_back(std::vector<double>());
            Ouputs.push_back(std::vector<double>());

            for (int j = 0; j < inp.size(); j++)
            {
                setInputsLayer(inp[j]);
                PassThrough();
                double outp = getOutputs()[0]; // Change later , should be size of Output Neurons
                Ouputs[outInx].push_back(outp);
            }

            Errors[outInx].push_back(lossFunc->calculateLoss(Ouputs[outInx], out[outInx]));
            std::cout << "Error : " << Errors[outInx][0] << std::endl; // Needs to be size of Output Neurons

            std::vector<double> outpNeuronsError; // Errors of all the Output Neurons
            for (int outNeurErr = 0; outNeurErr < out.size(); outNeurErr++)
            {
                double err = Ouputs[outInx][outNeurErr] * (1 - Ouputs[outInx][outNeurErr]) * Errors[outInx][outInx];
                outpNeuronsError.push_back(err);
            }

            std::vector<std::vector<double>> hiddenLayerErrors;
            for (int Layeridx = 0; Layeridx < HiddenLayers.size(); Layeridx++)
            {
                hiddenLayerErrors.push_back(std::vector<double>());
                for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
                {
                    hiddenLayerErrors[Layeridx].push_back(double());
                }
            }

            for (int Layeridx = indexOfLastLayer - 1; Layeridx >= 0; Layeridx--)
            {
                for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
                {
                    for (int WeightIdx = 0; WeightIdx < HiddenLayers[Layeridx][PercIdx].returnWeights().size(); WeightIdx++)
                    {
                        std::vector<double> newWeights = HiddenLayers[Layeridx][PercIdx].returnWeights();
                        if (Layeridx == indexOfLastLayer - 1)
                        {
                            double PerRes = HiddenLayers[Layeridx][PercIdx].CalcY();
                            double weightVal = newWeights[WeightIdx];

                            // Cycles through all the Output Neurons
                            for (int outpNeurErr = 0; outpNeurErr < outpNeuronsError.size(); outpNeurErr++)
                            {
                                double Err = PerRes * (1 - PerRes) * (weightVal * outpNeuronsError[outpNeurErr]);
                                hiddenLayerErrors[Layeridx][PercIdx] = Err;
                                newWeights[WeightIdx] = 1 * Err * weightVal;
                            }
                        }
                        else // This should be tested later
                        {
                            double PerRes = HiddenLayers[Layeridx][PercIdx].CalcY();
                            double weightVal = HiddenLayers[Layeridx][PercIdx].returnWeights()[WeightIdx];

                            double hiddenLayerErr = hiddenLayerErrors[Layeridx + 1][WeightIdx]; // Scared of this part here !

                            double Err = PerRes * (1 - PerRes) * (weightVal * hiddenLayerErr);
                            hiddenLayerErrors[Layeridx][PercIdx] = Err;
                        }
                    }
                }
            }

            // After this point , All Errors have been Calculated
            // Proceeding with Weigth Updates

            for (int Layeridx = indexOfLastLayer; Layeridx >= 0; Layeridx--)
            {
                for (int PercIdx = 0; PercIdx < HiddenLayers[Layeridx].size(); PercIdx++)
                {
                    std::vector<double> newWeights = HiddenLayers[Layeridx][PercIdx].returnWeights();
                    for (int WeightIdx = 0; WeightIdx < newWeights.size(); WeightIdx++)
                    {

                        std::vector<double> Inputs;
                        if (Layeridx == 0)
                        {
                            Inputs = this->InputLayer;
                        }
                        else
                        {
                            Inputs = CalcLayerOutputs(Layeridx - 1);
                        }

                        if (Layeridx == indexOfLastLayer)
                        {
                            double Err = -1 * outpNeuronsError[PercIdx];
                            newWeights[WeightIdx] = 1 * Err * Inputs[WeightIdx] + newWeights[WeightIdx];
                            std::cout << "OutpLayer : " << newWeights[WeightIdx] << std::endl;
                        }
                        else
                        {
                            double Err = hiddenLayerErrors[Layeridx][ PercIdx] ; 
                            newWeights[WeightIdx] = 1 * Err * Inputs[WeightIdx] + newWeights[WeightIdx];
                            std::cout << "Error : " << Err << std::endl;
                            std::cout << "HiddeLayer : " << newWeights[WeightIdx] << std::endl;
                        }
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
        out.push_back(this->HiddenLayers[last][i].CalcY());
    }
    return out;
}

#endif