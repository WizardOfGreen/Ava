#ifndef PERCEPTRON_CPP
#define PERCEPTRON_CPP

#include "Perceptron.h"

Perceptron::Perceptron()
{
}

Perceptron::~Perceptron()
{
}

void Perceptron::setActFunc(ActivationFunction *a)
{
    this->act = a;
}

double Perceptron::CalcY()
{
    double res = 0;
    res = res + this->bias;
    for (int i = 0; i < this->numInputs; i++)
    {
        res = res + this->Inputs[i] * this->Weights[i];
    }
    this->Outputs = res;
    return res;
}

double Perceptron::activatePerc()
{
    this->CalcY();
    this->Outputs = this->act->activate(this->Outputs);
    return this->Outputs;
}

void Perceptron::learnPerc(double trueVal)
{
    if (Outputs == 0)
        return;

    double err = trueVal - this->Outputs;
    for (int i = 0; i < this->Inputs.size(); i++)
    {
        this->Weights[i] = this->Weights[i] + rate * err * this->Inputs[i];
    }
    this->bias = this->bias + err * rate * 1;
}

void Perceptron::setInp(const std::vector<double> &inp)
{
    this->Inputs.clear();
    this->Weights.clear();

    this->numInputs = inp.capacity();
    for (double x : inp)
    {
        this->Inputs.push_back(x);
        this->Weights.push_back(1);
    }
}

void Perceptron::setBias(const double w)
{
    this->bias = w;
}

#endif
