#ifndef ACTFUNCTIONS_H
#define ACTFUNCTIONS_H

#include "ActivationFunction.h"

class Sigmoid : public ActivationFunction
{
public:
    double activate(double x)
    {
        return 1 / (1 + pow(e, (x * -1)));
    }

    double activateDerivative(double a)
    {
        double x = this->activate(a);
        double res = x * (1 - x);
        return res;
    }
};

class ReLU : public ActivationFunction
{
public:
    double activate(double x)
    {
        return ((x >= 0) ? x : 0);
    }

    double activateDerivative(double a)
    {
        double x = this->activate(a);
        return x > 0.0 ? 1.0 : 0.0;
    }
};

class SoftPlus : public ActivationFunction
{
public:
    double activate(double x)
    {
        return std::log(1 + pow(e, x));
    }

    double activateDerivative(double a)
    {
        double x = this->activate(a);
        return x > 0.0 ? 1.0 : 0.0;
    }
};

class TanH : public ActivationFunction
{
public:
    double activate(double x)
    {
        double r1, r2;
        r1 = pow(e, 2 * x) - 1;
        r2 = pow(e, 2 * x) + 1;
        double res = r1 / r2;
        return res;
    }

    // Error is within 6 decimal points 
    double activateDerivative(double a)
    {
        double x = this->activate(a);
        double res = 1 -  pow (x , 2) ; 
        return res;
    }
};

class StepFunc : public ActivationFunction
{
public:
    double activate(double x)
    {
        return ((x >= 0) ? 1 : 0);
    }
};

#endif
