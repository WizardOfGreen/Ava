#include <iostream>
#include <string>

#include "Matrix.h"
#include "ActFunctions.h"
#include "Perceptron.h"

Perceptron P;

void runEpoch(std::vector<double> inp, double out)
{
    P.TrainPerceptron(inp, out);
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

    std::vector<double> gand = {0, 0, 0, 1};
    std::vector<double> gor = {0, 1, 1, 1};
    std::vector<double> gnand = {1, 1, 1, 0};
    std::vector<double> gnor = {1, 0, 0, 0};

    std::vector<double> gxor = {0, 1, 1, 0};

    for (int i = 0; i < epoch; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            runEpoch(inputs[j], gxor[j]);
        }
    }

    P.printWeights();
    std::cout << std::endl;

    std::vector<double> inp = {0, 0};
    P.setInp(inp);
    P.CalcY();
    P.printOut();

    inp = {0, 1};
    P.setInp(inp);
    P.CalcY();
    P.printOut();

    inp = {1, 0};
    P.setInp(inp);
    P.CalcY();
    P.printOut();

    inp = {1, 1};
    P.setInp(inp);
    P.CalcY();
    P.printOut();

    std::cout << "Program Ran to Completion\n";
    system("pause");
    system("cls");
    return 0;
}