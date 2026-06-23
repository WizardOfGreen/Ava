#include <iostream>
#include <string>

#include "Matrix.h"
#include "ActFunctions.h"
#include "Perceptron.h"

Perceptron P;

void runEpoch(std::vector<double> inp , double out)
{
    P.setInpAW(inp) ; 
    P.activatePerc() ; 
    P.learnPerc(out) ; 
}

int main()
{
    P.setActFunc(new StepFunc());

    int epoch = 100;

    std::vector<std::vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}};

    std::vector<double> outp = {0, 0, 0, 1};

    for ( int i = 0 ; i < epoch ; i++)
    {
        for ( int j = 0 ; j < 4 ; j++)
        {
            runEpoch(inputs[j] , outp[j]) ; 
        }
    }

    P.printWeights() ; 
    std::cout << std::endl ; 

    std::vector<double> inp = {0 ,0 } ; 
    P.setInpAW(inp) ; 
    P.activatePerc() ; 
    P.printOut() ; 

    std::cout
        << "Program Ran to Completion\n";
    system("pause");
    system("cls");
    return 0;
}