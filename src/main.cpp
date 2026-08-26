#include <iostream>
#include <string>

#include "ActFunctions.h"
#include "LossFunctions.h"
#include "Perceptron.h"

#include "NeuralNetwork.h"

void PrintFeedForwardResults(NeuralNetwork &N, std::vector<double> &inp1, std::vector<double> &out1)
{
    std::cout << "INPUT : ";
    for (double d : inp1)
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    std::cout << "OUTPUT: ";
    for (double d : N.getOutputs())
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    std::cout << "Expected Output : ";
    for (double d : out1)
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}

int main()
{
    // =======================================
    // Neural Networks :
    // =======================================
    // XORTraining();                           // 1 Layers , 2 Inputs ,1 Output.
    // XORSolved() ;                            // Solved Problem To Test Feed Forwarding.
    // FeedForwardTest();                       // Tests Feed Forward Pass
    // =======================================
    // Function Tests :
    // =======================================
    // LossFunctionTests();
    // ActivationTests();
    // =======================================
    // Perceptron Tests :
    // Perceptron P ;
    // P.setInp( { 1 , 0 } ) ;
    // std::cout << "Perceptron Test\n" ;

    // =======================================
    // singlePerceptronTest() ;
    // =======================================

    std::cout << "Program Ran Successfully\n";

    return 0;
}