#ifndef PERCEPTRON_CPP
#define PERCEPTRON_CPP

#include "Perceptron.h"

Perceptron::Perceptron()
{
}

Perceptron::Perceptron(ActivationFunction *act)
{
    this->act = act ; 
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
    return this->act    ;
}

double Perceptron::CalcY()
{
    double res = 0;
    res = res + this->bias;
    for (int i = 0; i < this->numInputs; i++)
    {
        res = res + this->Inputs[i] * this->Weights[i];
    }
    this->Outputs = (res);

    return this->act->activate(res) ;
}

double Perceptron::returnNet()
{
        double res = 0;
    res = res + this->bias;
    for (int i = 0; i < this->numInputs; i++)
    {
        res = res + this->Inputs[i] * this->Weights[i];
    }
    this->Outputs = (res);

    return this->Outputs ;

}

void Perceptron::setInp(const std::vector<double> &inp)
{
    this->Inputs.clear();

    this->numInputs = inp.capacity();
    for (double x : inp)
    {
        this->Inputs.push_back(x);
    }

    if (this->Inputs.size() > this->Weights.size())
        this->setWeights();
}

void Perceptron::setWeights()
{
    this->Weights.clear();
    while (this->Weights.size() < this->Inputs.size())
    {
        this->Weights.push_back(1);
    }
}

void Perceptron::setWeights(std::vector<double> w)
{
    this->Weights.clear() ; 
    for ( double x: w)
    {
        this->Weights.push_back(x) ; 
    }
}

void Perceptron::setBias(const double b)
{
    this->bias = b;
}

void Perceptron::TrainPerceptron(std::vector<double> inp, double out)
{
    this->Inputs.clear();
    this->numInputs = inp.capacity();

    for (double x : inp)
    {
        this->Inputs.push_back(x);
    }

    if (this->Inputs.size() > this->Weights.size())
        this->setWeights(); // Sets weights to a basic amount with size the same as input.

    double res = 0;
    res = res + this->bias;
    
    for (int i = 0; i < this->numInputs; i++)
    {
        res = res + this->Inputs[i] * this->Weights[i];
    }

    res = this->act->activate(res); // If res is >= 0 than return 1 , else return 0 ;
    double err = out - res ;

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

std::vector<double> Perceptron::returnWeights()
{
    return this->Weights ;
}

std::vector<double> Perceptron::returnInputs()
{
    return this->Inputs ; 
}

double Perceptron::returnBias()
{
    return this->bias ; 
}

#endif
