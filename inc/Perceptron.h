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

    double biasWeight = 0 ;
    double biasInput = 1 ; 

    std::vector<double> Inputs ; 
    std::vector<double> Weights ; 

    double Outputs ;
    double ActRes ; 

    double rate = 0.01 ; 

public:
    Perceptron();
    ~Perceptron();

    void setActFunc(ActivationFunction * a ) ;
    double CalcY() ;
    double activatePerc() ;
    void learnPerc(double trueVal) ; 

    void setInpAW(const std::vector<double>& inp , const std::vector<double>& w) ; 
    void setBias(const double w ) ; 

    void testFunc() {
        std::cout << act->activate(3) << std::endl  ;  
    }
};

#endif
