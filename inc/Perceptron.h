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
    double bias = 1;

    std::vector<double> Inputs;
    std::vector<double> Weights;

    double Outputs;

    double rate = 0.05;

public:
    Perceptron();
    Perceptron(ActivationFunction *act);
    ~Perceptron();

    void setActFunc(ActivationFunction *a);
    double CalcY();

    void printY()
    {
        std::cout << this->Outputs << std::endl;
    }
    void printAct()
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

    void TrainPerceptron(std::vector<double> inp, double out);

    void testFunc()
    {
        std::cout << act->activate(3) << std::endl;
    }

    void printWeights()
    {
        for (double x : Weights)
        {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        std::cout << this->bias << std::endl;
    }

    void printOut()
    {
        std::cout << this->Outputs << std::endl;
    }
};

#endif
