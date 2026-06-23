#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <cmath>

const double e = std::exp(1.0);

class ActivationFunction
{
private:
    /* data */
public:
    virtual double activate(double x ) = 0 ; 
};

#endif