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
    this->act = a ; 
}

double Perceptron::CalcY()
{
    this->Outputs = this->biasInput * this->biasWeight ; 
    for ( int i = 0 ; i < this->numInputs ; i++ ) {
        this->Outputs += this->Inputs[i] * this->Weights[i] ;
    }
    return 0.0;
}

double Perceptron::activatePerc()
{
    this->CalcY() ; 
    this->ActRes = this->act->activate(this->Outputs) ; 
    return this->ActRes ;
}

void Perceptron::learnPerc(double trueVal)
{
    if ( Outputs == trueVal)
        return ;

    double err = trueVal - this->Outputs  ;
    err += rate ; 
    for ( int i = 0 ; i < this->Inputs.size() ; i++)
    {
        this->Weights[i] += err ; 
    }
    this->biasWeight += err ; 
}

void Perceptron::setInpAW(const std::vector<double>& inp)
{
    this->Inputs.clear() ;
    this->Weights.clear() ; 
    
    this->numInputs = inp.capacity() ;
    for ( double x : inp )
    {
        this->Inputs.push_back(x) ; 
        this->Weights.push_back(0) ; 
    }
}

void Perceptron::setBias(const double w)
{
    this->biasInput = 1 ;
    this->biasWeight = w ;
}

#endif

