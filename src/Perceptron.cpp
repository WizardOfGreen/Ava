#ifndef PERCEPTRON_CPP
#define PERCEPTRON_CPP

#include "Perceptron.h"

Perceptron::Perceptron()
{
}

Perceptron::Perceptron(ActivationFunction *act)
{
    this->act = act;
}

Perceptron::~Perceptron()
{
}

void Perceptron::setActFunc(ActivationFunction *a)
{
    this->act = a;
}

ActivationFunction *Perceptron::getAct()
{
    return this->act;
}

double Perceptron::returnNet()
{
    double res = 0;
    res = res + (this->bias * 1);
    for (int i = 0; i < this->numInputs; i++)
    {
        res = res + this->Inputs[i] * this->Weights[i];
    }
    this->Outputs = res;

    return res;
}

double Perceptron::retAct()
{
    double res = returnNet();
    return this->act->activate(res);
}

void Perceptron::setInp(const std::vector<double> &inp)
{
    this->Inputs.clear();
    this->numInputs = inp.capacity();

    for (double x : inp)
    {
        this->Inputs.push_back(x);
    }
}

void Perceptron::setWeights(std::vector<double> w)
{
    this->Weights.clear();
    for (double x : w)
    {
        this->Weights.push_back(x);
    }
}

void Perceptron::setBias(const double b)
{
    this->bias = b;
}

void Perceptron::TrainPerceptron(double out, int epoch)
{
    for (int i = 0; i < epoch; i++)
    {
        double res = this->retAct();
        double err = out - res;

        if (err == 0)
        {
            return;
        }

        for (int i = 0; i < this->Inputs.size(); i++)
        {
            this->Weights[i] = this->Weights[i] + rate * err * this->Inputs[i];
        }

        this->bias = this->bias + err * rate * 1;
    }
}

std::vector<double> &Perceptron::returnWeights()
{
    std::vector<double> copy;
    for (double x : this->Weights)
    {
        copy.push_back(x);
    }
    return copy;
}

std::vector<double> &Perceptron::returnInputs()
{
    std::vector<double> copy;
    for (double x : this->Inputs)
    {
        copy.push_back(x);
    }
    return copy;
}

double Perceptron::returnBias()
{
    return this->bias;
}

#endif
