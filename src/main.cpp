#include <iostream>
#include <string>

#include "ActFunctions.h"
#include "LossFunctions.h"
#include "Perceptron.h"

#include "NeuralNetwork.h"
#include "ImageExtractor.h"

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
    ImageExtractor img ; 
    std::vector<std::vector<int>> x = img.ConvertImageToMatrix("dogs/GoldenRetriever/0EEGHVQ8Z263.jpg");

    std::cout << "Program Ran Successfully\n";

    return 0;
}