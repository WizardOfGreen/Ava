#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <cmath>

const double e = std::exp(1.0);

class ActivationFunction
{
private:
    /* data */
public:
    virtual double activate(double x) = 0;
    
    // Active Derivative does not expect an activated value
    virtual double activateDerivative(double a) { return 0.0; };
    // Derivative expects an activated value
    virtual double Derivative(double a) { return 0.0; };
};

#endif