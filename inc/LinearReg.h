#ifndef LINEARREG_H
#define LINEARREG_H

#include <cmath>
#include <string>
#include <vector>
#include <iostream>


class LinearReg
{
    std::vector<float> PreVal ; 

    std::vector<std::vector <std::string>> Data ; 

    std::vector<float> YValues ; 
    std::vector<std::vector <float>> XValues ; 
    std::vector<float> Slopes ; 

    float Intercept ; 
    float xmean , ymean ;
    double costFunction ;  



public:
    LinearReg();
    ~LinearReg();

    void LoadData(const std::vector<std::vector<std::string>>& inp ) ; 
    void CalculateMeans() ; 
    void CalcPredictedValues() ; 

    void CostFunction() ; 
    void PrintValues()  ;

    const std::vector<std::vector <float>>& ReturnXValues() ; 
};

#endif