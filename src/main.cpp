#include <iostream>
#include <string>

#include "inc/Matrix.h"

int main()
{
    Matrix<double> M;
    std::vector<std::vector<std::string>> data;

    std::vector<std::vector<double>> a = {
        {4, 3, 8},
        {6, 2, 5},
        {1, 5, 9}};

    std::vector<std::vector<double>> c ; 
    std::vector<std::vector<double>> b = M.Inverse(a) ;
    
    std::cout << "Program Ran to Completion\n";
    system("pause");
    system("cls");
    return 0;
}