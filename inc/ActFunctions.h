#ifndef ACTFUNCTIONS_H
#define ACTFUNCTIONS_H

#include "ActivationFunction.h"

/**
 * Class that contains all functions for Sigmoid Activation function.
 */
class Sigmoid : public ActivationFunction
{
public:
    // σ(x) = 1 / (1 + e^-x)
    /**
     * Standard Sigmoid function
     * σ(x) = 1 / (1 + e^-x)
     * @param x Value to be activated
     */
    double activate(double x)
    {
        return 1 / (1 + pow(e, (x * -1)));
    }

    // σ'(x) = σ(x) * ( 1 - σ(x) ) ;
    /**
     * Activated Derivative of Sigmoid
     * @param a Will be converted using sigmoid and used in the derivative.
     * σ'(x) = σ(x) * ( 1 - σ(x) )
     */
    double activateDerivative(double a)
    {
        double x = activate(a);
        double res = x * (1 - x);
        return res;
    }
    // activation not required for derivative
    /**
     * Derivative of the sigmoid function where parameter a is not
     * activated by the sigmoid
     *
     */
    double pre_activation_derivative(double a)
    {
        double x = (a);
        double res = x * (1 - x);
        return res;
    }
};
/**
 * Class that contains all functions for ReLu Activation function.
 */
class ReLU : public ActivationFunction
{
public:
    /**
     * Standard Relu formula
     */
    double activate(double x)
    {
        return ((x > 0.0) ? x : 0.0);
    }

    /**
     * Derivative of Relu , Activates between 1 and 0
     */
    double activateDerivative(double a)
    {
        double x = a;
        return x > 0.0 ? 1.0 : 0.0;
    }

    /**
     * Same as activateDerivative
     */
    double pre_activation_derivative(double x)
    {
        return (x > 0.0) ? 1.0 : 0.0;
    }
};
/**
 * Class that contains all functions for Softplus Activation function.
 */
class SoftPlus : public ActivationFunction
{
public:
    /**
     * Standard Softplus function
     * f(x) = ln(1 + e^x)
     */
    double activate(double x)
    {
        return std::log(1 + pow(e, x));
    }

    /**
     * Derivative of Softplus
     * f'(x) = 1 / (1 + e^-x) ;
     */
    double activateDerivative(double a)
    {
        double res = 1 / (1 + pow(e, (-1 * a)));
        return res;
    }

    /**
     * Same as activate Derivative
     */
    double pre_activation_derivative(double x)
    {
        double res = 1 - (pow(e, -1 * x));
        return res;
    }
};

/**
 * Class that contains all functions for Tanh Activation function.
 */
class TanH : public ActivationFunction
{
public:
    /**
     * Standard TanH formula
     * f(x) = (e^x - e^-x ) / (e^x + e^-x)
     */
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

    /**
     * Activated derivative of TanH
     * f'(x) = 4 / (e^x + e^-x)^2
     */
    double activateDerivative(double a)
    {
        double x = a;
        double res = 1 - pow(x, 2);
        res = 4 / pow((pow(e, x) + pow(e, -1 * x)), 2);
        return res;
    }

    double pre_activation_derivative(double x)
    {
        double res = 1 - pow(x, 2);
        return res;
    }
};
/**
 * Simple step function
 */
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
