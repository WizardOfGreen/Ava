#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Perceptron.h"
#include "LossFunctions.h"

class NeuralNetwork
{
private:
    std::vector<double> InputLayer;
    std::vector<std::vector<Perceptron>> HiddenLayers;
    std::vector<ActivationFunction *> Layer_Activations;

    ActivationFunction *act;
    LossFunction *lossFunc;

public:
    /**
     * Creates A Neural Network With no Inputs , Layers or Loss Functions
     */
    NeuralNetwork();
    /**
     * Basic Destructor
     */
    ~NeuralNetwork();

    /**
     * Assigns a Loss Function for error testing
     */
    void AssignLossFunction(LossFunction *lossFunc);
    /**
     * Returns the Loss Functions
     */
    LossFunction *ReturnLossFunction();

    /**
     * Adds the input layer i.e. all the inputs the NN will use
     */
    void setInputsLayer(const std::vector<double> &Inp);
    /**
     * Creates One layer at the "front" of the NN .
     * @param amount Adds this amount of Perceptrons to layer.
     * @param act This is the decided activation function that the Perceptron will have , aswell as the entire layer
     */
    void addLayer(int amount, ActivationFunction *act);
    /**
     * This function is used when a perceptron from one layer needs its input from the layer "behind it"
     * @param index . Index of the Perceptron in the layer.
     * @param inp Vector of the input of the Perceptronn
     */
    void setPercInputs(int index, std::vector<double> inp);

    /**
     * Sets the weight of the Perceptron in the chosen Layer.
     * @param indexOfLayer Index of the Layer to access . First layer is 0.
     * @param indexOfPerceptron Index of the Perceptron which weights will be changed.
     * @param newWeights The New weights the Perceptron will have.
     */
    void setLayerWeights(int indexOfLayer, int indexOfPerceptron, std::vector<double> newWeights);

    /**
     * Sets the Bias of the Perceptron in the chosen Layer.
     * @param indexOfLayer Index of the Layer to access . First layer is 0.
     * @param indexOfPerceptron Index of the Perceptron which Bias will be changed.
     * @param newBias The New Bias the Perceptron will have.
     */
    void setLayerBias(int indexOfLayer, int indexOfPerceptron, double newBias);

    /**
     * Calculates the outputs of the layer
     * @param index Which layer will be used to calculate the outputs
     */
    std::vector<double> CalcLayerOutputs(int index);

    /**
     * Uses all the functions above , starts at Layer 0 using the Inputs and Traverses through the NN using 
     * the previous layers output for the current layers input until the end is reached.
     */
    void FeedForward();
    /**
     * In essence , This is backpropagation.
     * This trains the Neural Network to achieve the desired outputs for inputs or the epoch was reached.
     * @param inp List of all the possible inputs , corresponds to the inp i.e NeuralNetworkResult(inp[2]) == out[2]
     * @param out List of all the possible outputs , corresponds to the inp i.e out[2] == NeuralNetworkResult(inp[2])
     * @param epoch Amount of times training will take place
     * @param learning_rate Learning Rate .How rapid or slow the changes to the Neural Network will take place.
     */
    void TrainNN(const std::vector<std::vector<double>> &inp, const std::vector<std::vector<double>> &out, int epoch, double learning_rate);

    /**
     * Returns the outputs of the current settings of weights and inputs assigned to the Neural Network.
     */
    std::vector<double> getOutputs();
};

#endif