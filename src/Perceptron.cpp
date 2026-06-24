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
    this->Outputs = this->act->activate(res);

    return res;
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

    std::cout << "\nThis should only run once\n";
    std::cout << "Weights Amount : " << this->Weights.size() << std::endl;
    std::cout << std::endl;
}

void Perceptron::setBias(const double w)
{
    this->bias = w;
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

#endif
