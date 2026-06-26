#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include <cmath>
#include <vector>

class LossFunction
{
public:
    virtual double calculateLoss(std::vector<double> predicted, std::vector<double> trueval) = 0;
    virtual double calculateLoss2(double predicted, double trueval) { return 0.0; };
};

#endif