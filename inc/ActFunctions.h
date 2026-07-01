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
        double res = a * (1 - a);
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
};

class SoftPlus : public ActivationFunction
{
public:
    double activate(double x)
    {
        return std::log(1 + pow(e, x));
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
