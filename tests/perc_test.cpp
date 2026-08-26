#include <iostream>
#include <vector>

#include "Perceptron.h"
#include "Helper.h"

bool testInp(std::vector<double> exp, std::vector<double> res)
{
    if (exp.size() != res.size())
        return false;

    for (int i = 0; i < exp.size(); i++)
    {
        if (exp[i] != res[i])
            return false;
    }

    return true;
}

int main()
{
    bool failure = false;
    Perceptron P;
    ActivationFunction *sigmoid = new Sigmoid();
    Perceptron P2(sigmoid);
    ActivationFunction *compare_sigmoid = new Sigmoid();

    ActivationFunction *testingCase;
    ActivationFunction *testingCase2;

    std::cout << "Testing Activation Functions\n";
    P.setActFunc(sigmoid);
    testingCase = P.getAct();
    testingCase2 = P2.getAct();

    if ((typeid(testingCase) == typeid(compare_sigmoid)) && (typeid(testingCase) == typeid(compare_sigmoid)))
    {
        std::cout << "Activation Getter and Setter Passed!\n";
    }
    else
    {
        std::cout << "Activation Getter and Setter Failed!\n";
        return 1;
    }

    std::vector<double> inp = {1.0, 2.0};
    std::vector<double> weight = {0.2, 0.1};

    double bias = 0.1055;
    double out;
    double act;
    double expOut = 0.5055;
    double expAct = 0.623750978;

    P.setInp(inp);
    P.setWeights(weight);
    P.setBias(bias);

    std::cout << "Testing Getter and Setters for Weigth , Input , Bias!\n";

    std::vector<double> retInp = P.returnInputs();
    std::vector<double> retWeight = P.returnWeights();
    double retBias = P.returnBias();

    if (testInp(inp, retInp))
        std::cout << "Getter And Setter For Input Passed!\n";
    else
    {
        std::cout << "Getter And Setter For Input Passed!\n";
        return 1;
    }

    if (testInp(weight, retWeight))
        std::cout << "Getter And Setter For Weight Passed!\n";
    else
    {
        std::cout << "Getter And Setter For Weight Passed!\n";
        return 1;
    }

    if (retBias == bias)
        std::cout << "Getter And Setter For Bias Passed!\n";
    else
    {
        std::cout << "Getter And Setter For Bias Failed!\n";
        return 1;
    }
    std::cout << "Testing Net Return & Activation for Neuron\n";

    out = P.returnNet();
    act = P.retAct();

    if (compare(out, expOut))
    {
        std::cout << "Output Failed!\n";
        return 1;
    }
    else
        std::cout << "Output Passed!\n";

    if (compare(act, expAct))
    {
        std::cout << "Activation Failed!\n";
        std::cout << "Res : " << act << " Exp : " << expAct << std::endl;
        return 1;
    }
    else
        std::cout << "Activation Passed!\n";

    std::cout << "Testing Training of Perceptron for 300 Epochs!\n";

    P.TrainPerceptron(0.8, 300);
    double trainAct = P.retAct();

    if (compare(0.8, trainAct))
    {
        std::cout << "Training Failed !\n";
        return 1;
    }
    else
        std::cout << "Training Successfull!\n";

    std::cout << "Perceptron Passed All Tests!\n" ; 

    return 0;
}