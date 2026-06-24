#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include "ActFunctions.h"

class Perceptron
{
private:
    ActivationFunction * act ;

    int numInputs = 0 ;
    double bias = 1 ;

    std::vector<double> Inputs ; 
    std::vector<double> Weights ; 

    double Outputs ;
    double ActAmount ; 

    double rate = 0.05 ; 

public:
    Perceptron();
    ~Perceptron();

    void setActFunc(ActivationFunction * a ) ;
    double CalcY() ;
    double activatePerc() ;
    void learnPerc(double trueVal) ; 

    void setInp(const std::vector<double>& inp) ; 
    void setWeights() ; 
    void setBias(const double w ) ; 

    void testFunc() {
        std::cout << act->activate(3) << std::endl  ;  
    }

    void printWeights() {
        for ( double x : Weights)
        {
            std::cout << x << " " ; 
        }
        std::cout << std::endl ; 
        std::cout << this->bias << std::endl ; 
    }

    void printOut() {
        std::cout << this->Outputs << std::endl ; 
    }
};

#endif
