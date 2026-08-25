#ifndef ACTFUNCTIONS_H
#define ACTFUNCTIONS_H

#include "ActivationFunction.h"

class Sigmoid : public ActivationFunction
{
public:
    // σ(x) = 1 / (1 + e^-x)
    double activate(double x)
    {
        return 1 / (1 + pow(e, (x * -1)));
    }

    // σ'(x) = σ(x) * ( 1 - σ(x) ) ;  
    double activateDerivative(double a)
    {
        double x = activate(a) ;
        double res = x * (1 - x);
        return res;
    }
    // activation not required for derivative
    double pre_activation_derivative(double a)
    {
        double x = (a) ;
        double res = x * (1 - x);
        return res;
    }
};

class ReLU : public ActivationFunction
{
public:
    double activate(double x)
    {
        return ((x >= 0.0) ? x : 0.0 );
    }

    double activateDerivative(double a)
    {
        double x = a;
        return x > 0.0 ? 1.0 : 0.0;
    }

    double pre_activation_derivative(double x) 
    {
        return (x > 0.0) ? 1.0 : 0.0 ; 
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
        double res = 1 / (1 + pow(e, (-1 * a)));
        return res;
    }

    double pre_activation_derivative(double x )
    {
        double res = 1 - ( pow ( e , -1 * x )) ; 
        return res ; 
    }
};

class TanH : public ActivationFunction
{
public:
    double activate(double x)
    {
        double r1, r2;

        double x1 = pow(e, x);
        double x2 = pow(e, (x * -1));

        r1 = x1 - x2;
        r2 = x1 + x2;
        double res = r1 / r2;
        return res;
    }

    double activateDerivative(double a)
    {
        double x = a;
        double res = 1 - pow(x, 2);
        res = 4 / pow ( (pow(e , x) + pow( e , -1 * x ) ) , 2 ) ; 
        return res;
    }

    double pre_activation_derivative(double x)
    {
        double res = 1 - pow ( x , 2) ; 
        return res  ; 
    }
};

class StepFunc : public ActivationFunction
{
public:
    double activate(double x)
    {
        return ((x >= 0) ? 1 : 0);
    }
    double activateDerivative(double a)
    {
        return ((a >= 0) ? 1 : 0);
    }

};

#endif
