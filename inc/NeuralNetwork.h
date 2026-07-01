#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Perceptron.h"
#include "LossFunctions.h"

class NeuralNetwork
{
private:
    /* data */
    std::vector<double> InputLayer;
    std::vector<std::vector<Perceptron>> HiddenLayers;
    std::vector<ActivationFunction *> Layer_Activations;

    ActivationFunction *act;
    LossFunction *lossFunc;

public:
    NeuralNetwork();
    ~NeuralNetwork();

    void AssignLossFunction(LossFunction *lossFunc);

    void setInputsLayer(const std::vector<double> &Inp);
    void addLayer(int amount, ActivationFunction *act);
    void setPercInputs(int index, std::vector<double> inp);

    void setLayerWeights(int indexOfLayer, int indexOfPerceptron, std::vector<double> newWeights);

    std::vector<double> CalcLayerOutputs(int index);
    void PassThrough();

    void TrainNN(const std::vector<std::vector<double>>& inp, const std::vector<std::vector<double>>& out, int epoch);

    std::vector<double> getOutputs();

    void CheckHiddenLayerOutputs()
    {
        for (int i = 0; i < this->HiddenLayers.size(); i++)
        {
            for (int j = 0; j < this->HiddenLayers[i].size(); j++)
            {
                std::cout << "OUTPUT OF HIDDEN LAYER " << i << " AND PERCEPTRON " << j << std::endl;
                HiddenLayers[i][j].printInp();
                std::cout << HiddenLayers[i][j].CalcY() << std::endl;
                std::cout << "END OF THIS PERCEPTON\n";
            }
        }
    }
    void CheckInternalWeights(int layeridx, int percidx)
    {
        std::cout << "Weights for Layer " << layeridx + 1 << " Percepton : " << percidx << std::endl;
        for (double d : HiddenLayers[layeridx][percidx].returnWeights())
        {
            std::cout << d << " ";
        }
        std::cout << std::endl;
    }

    void CheckHiddenLayerInp(int idx)
    {
        for (int i = 0; i < HiddenLayers[idx].size(); i++)
        {
            HiddenLayers[idx][i].printInp();
        }
    }

    void checkWeights()
    {
        std::cout << std::endl;
        int lastLayerIndex = HiddenLayers.size() - 1;
        for (int indexLayer = 0; indexLayer < lastLayerIndex + 1; indexLayer++)
        {
            for (int percIdx = 0; percIdx < HiddenLayers[indexLayer].size(); percIdx++) // For now , just one
            {
                for (int weightIdx = 0; weightIdx < HiddenLayers[indexLayer][percIdx].returnWeights().size(); weightIdx++)
                {
                    std::cout << indexLayer << " " << percIdx << " " << weightIdx << " ";
                    std::cout << "W: " << HiddenLayers[indexLayer][percIdx].returnWeights()[weightIdx];
                    std::cout << std::endl;
                }
            }
        }
    }

    void KnownXORAnswer()
    {
        std::vector<double> w1 = {1, 1};
        std::vector<double> v1 = {1, -2};

        HiddenLayers[0][0].setWeights(w1);
        HiddenLayers[0][1].setWeights(w1);

        HiddenLayers[1][0].setWeights(v1);

        HiddenLayers[0][0].setBias(-0.5);
        HiddenLayers[0][1].setBias(-1.5);

        HiddenLayers[1][0].setBias(-0.5);
    }
};

#endif