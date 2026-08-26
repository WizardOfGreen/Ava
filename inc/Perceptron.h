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
    Perceptron();
    Perceptron(ActivationFunction *act);

    void setActFunc(ActivationFunction *a);
    ActivationFunction *getAct();

    double retAct();
    double returnNet();

    void setInp(const std::vector<double> &inp);
    void setWeights(std::vector<double> w);
    void setBias(const double b);

    void TrainPerceptron(double out , int epoch);

    std::vector<double> returnWeights();
    std::vector<double> returnInputs();
    double returnBias();

    ~Perceptron();
};

#endif
