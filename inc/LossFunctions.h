#ifndef LOSSFUNCTIONS_H
#define LOSSFUNCTIONS_H

#include "LossFunction.h"

// Requires all n: Number of data points ,
class MSE : public LossFunction
{
public:
    double calculateLoss(std::vector<double> predicted, std::vector<double> trueval)
    {
        int size = predicted.size();
        double res = 0;
        for (int i = 0; i < size; i++)
        {
            res += pow((trueval[i] - predicted[i]), 2);
        }
        res /= size;
        return res;
    }
    
    double calculateLoss2(double predicted, double trueval)
    {
        double res = 0;
        res = 0.5 * pow((trueval - predicted), 2);
        return res;
    }
};

class MSE2 : public LossFunction
{
public:
    double calculateLoss(std::vector<double> predicted, std::vector<double> trueval)
    {
        int size = predicted.size();
        double res = 0;
        res = (predicted[0] - trueval[0]) * (predicted[0]) * (1 - predicted[0]);
        return res;
    }

    double calculateLoss2(double predicted, double trueval)
    {
        double res = 0;
        res = 0.5 * pow((trueval - predicted), 2);
        return res;
    }
};

class RootMSE : public LossFunction
{
public:
    double calculateLoss(std::vector<double> predicted, std::vector<double> trueval)
    {
        return sqrt((new MSE())->calculateLoss(predicted, trueval));
    }

    double calculateLoss2(double predicted, double trueval)
    {
        double res = 0;
        res = (predicted - trueval) * (predicted) * (1 - predicted);
        return res;
    }
};

class AbsMSE : public LossFunction
{
public:
    double calculateLoss(std::vector<double> predicted, std::vector<double> trueval)
    {
        int size = predicted.size();
        double res = 0;
        for (int i = 0; i < size; i++)
        {
            res += abs((trueval[i] - predicted[i]));
        }
        res /= size;
        return res;
    }
};

class BookLoss : public LossFunction
{
public:
    double calculateLoss(std::vector<double> predicted, std::vector<double> trueval)
    {
        return 0.00;
    }
    double calculateLoss2(double predicted, double trueval)
    {
        double res = pow((trueval - predicted), 2);
        return res;
    }
};

#endif