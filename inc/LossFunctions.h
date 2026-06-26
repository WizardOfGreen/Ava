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
};

class RootMSE : public LossFunction
{
public:
    double calculateLoss(std::vector<double> predicted, std::vector<double> trueval)
    {
        return sqrt((new MSE())->calculateLoss(predicted, trueval));
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

#endif