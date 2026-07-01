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
    double bias = 0;

    std::vector<double> Inputs;
    std::vector<double> Weights;

    double Outputs;

    double rate = 0.05;

public:
    Perceptron();
    Perceptron(ActivationFunction *act);
    ~Perceptron();

    void setActFunc(ActivationFunction *a);
    ActivationFunction * getAct() ; 
    double CalcY();
    double returnNet() ; 

    void printY()
    {
        std::cout << act->activate(this->Outputs) << std::endl;
    }

    void printInp() 
    {
         std::cout << "INPUTS\n" ;
        for ( double d : this->Inputs )
        {
            std::cout << d << " " ; 
        }
        std::cout << std::endl ; 
    }

    void setInp(const std::vector<double> &inp);
    void setWeights();
    void setWeights(std::vector<double> w);
    void setBias(const double b);

    void TrainPerceptron(std::vector<double> inp, double out) ; 
    std::vector<double> returnWeights() ; 
    std::vector<double> returnInputs() ; 
    double returnBias() ; 


};

#endif
