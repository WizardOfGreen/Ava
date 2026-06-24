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
    this->ActAmount = this->act->activate(this->CalcY());
    return this->ActAmount;
}

void Perceptron::learnPerc(double trueVal)
{
    double err = trueVal - (this->activatePerc()) ;

    if (err == 0)
        return;

    for (int i = 0; i < this->Inputs.size(); i++)
    {
        this->Weights[i] = this->Weights[i] + rate * err * this->Inputs[i];
    }
    this->bias = this->bias + err * rate * 1;
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
    this->Weights.clear() ;
    while (this->Weights.size() < this->Inputs.size())
    {
        this->Weights.push_back(1);
    }

    std::cout << "This should only run once\n" ; 
    std::cout << "Weights Amount : " << this->Weights.size() << std::endl ; 
}

void Perceptron::setBias(const double w)
{
    this->bias = w;
}

#endif
