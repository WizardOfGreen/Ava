#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include "ActFunctions.h"

class Perceptron
{
private:
    ActivationFunction *act;

    int numInputs = 0;
    double bias = 0.5;

    std::vector<double> Inputs;
    std::vector<double> Weights;

    double Outputs;

    double rate = 0.07;

public:
/**
 * Perceptron basic constructor
 */
    Perceptron();
    /**
     * Perceptron costructor with an Activation Function parameter.
     */
    Perceptron(ActivationFunction *act);

    /**
     * Assigning a activation function to the Perceptron
     */
    void setActFunc(ActivationFunction *a);
    /**
     * Getter function for the Activation Function
     */
    ActivationFunction *getAct();

    /**
     * Returns the Output of the Neuron After the value has been activated by the Perceptons Activation function.
     */
    double retAct();
    /**
     * Returns the output of the Neuron before the value has been activated by the Perceptons activation function.
     */
    double returnNet();

    /**
     * Sets the input for the Percepton.
     */
    void setInp(const std::vector<double> &inp);
    /**
     * Sets each weight of the inputs for the perceptron
     */
    void setWeights(std::vector<double> w);
    /**
     * Sets the bias for the perceptron
     */
    void setBias(const double b);

    /**
     * Trains the Perceptron using the Perceptron Learning Rule
     * @param out The Activated value the perceptron should try to reach.
     * @param epoch The amount of time the Perceptron will run the Learning Rule.
     */
    void TrainPerceptron(double out, int epoch);

    /**
     * Returns a copy of the weights of the perceptron for either error checking or saving the perceptrons data.
     */
    std::vector<double> & returnWeights();
    /**
     * Returns a copu of the Inputs for the Perceptron for error checking.
     */
    std::vector<double> & returnInputs();
    /**
     * Returns the bias for error checking or saving the perceptrons data.
     */
    double returnBias();

    ~Perceptron();
};

#endif
