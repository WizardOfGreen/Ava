#ifndef LINEARREG_CPP
#define LINEARREG_CPP

#include "inc/LinearReg.h"
#include <iomanip>

LinearReg::LinearReg()
{
}

LinearReg::~LinearReg()
{
}

void LinearReg::LoadData(const std::vector<std::vector<std::string>> &inp)
{
    int n = inp.size();
    int m;
    XValues.resize(inp[1].size() - 1);

    for (int i = 1; i < n; i++)
    {
        m = inp[i].size();
        YValues.push_back(std::stof(inp[i][0]));
        for (int j = 1; j < m; j++)
        {
            int x_index = j - 1 ; 
            if (inp[i][j].compare("yes") == 0)
                XValues[x_index].push_back(1);
            else if (inp[i][j].compare("no") == 0)
                XValues[x_index].push_back(0);
            else if (inp[i][j].compare("unfurnished") == 0)
                XValues[x_index].push_back(0);
            else if (inp[i][j].compare("semi-furnished") == 0)
                XValues[x_index].push_back(1);
            else if (inp[i][j].compare("furnished") == 0)
                XValues[x_index].push_back(2);
            else
                XValues[x_index].push_back(std::stof(inp[i][j]));
        }
    }
}

void LinearReg::CalculateMeans()
{
    int sizen = this->XValues.size() ; 
    int sizem = this->XValues[0].size() ; 

    this->xmean = 0 ;
    this->ymean = 0 ;

    for ( int i = 0 ; i < sizen ; i++)
    {
        this->xmean = 0 ; 
        ymean += this->YValues[i] ; 
        for ( int j = 0 ; j < sizem ; j++)
        {

        }
    }

    ymean = ymean / sizen ; 
}

void LinearReg::CalcPredictedValues()
{
    int m = this->XValues.size() ; 
    int n = this->XValues[0].size() ; 

    for ( int i = 0 ; i < m ; i++)
    {
        this->PreVal.push_back(0) ; 

        for ( int j = 0 ; j < n ; j++)
        {
            this->PreVal[i] += this->XValues[i][j] * this->Slopes[j] ;  
        }
    }
}

void LinearReg::CostFunction()
{
    int n = this->YValues.size();

    int Sum = 0;
    for (int i = 0; i < n; i++)
    {
        Sum += pow(this->PreVal[i] - this->YValues[i], 2);
    }

    this->costFunction = Sum * (1 / n);
}

void LinearReg::PrintValues()
{
    for ( float Y : this->YValues )
    {
        std::cout << "Y Val : " << std::fixed  << std::setprecision(2) << Y << std::endl ; 
    }

    for ( std::vector< float > x1 : this->XValues )
    {
        std::cout << "XValues : " ; 
        for ( float x2 : x1 )
        {
            std::cout << x2 << " " ; 
        }
        std::cout << std::endl ; 
    }
}

const std::vector<std::vector <float>> &LinearReg::ReturnXValues()
{
    return (this->XValues) ; 
}

#endif
