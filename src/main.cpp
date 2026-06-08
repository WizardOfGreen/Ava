#include <iostream>
#include <string>

#include "Matrix.h"

int main()
{
    Matrix<double> M ;
    std::vector<std::vector<std::string>> data;

    std::vector<std::vector<double>> a = {
        {4, 3, 8},
        {6, 2, 5},
        {1, 5, 8}}; 

    std::vector<std::vector<double>> c ; 
    std::vector<std::vector<double>> b = M.Inverse(a) ;

    M.printMatrixes(a) ;
    M.printMatrixes(b) ; 

    M.printMatrixes(M.MultiplyMatrix(a,b)) ; 

    std::cout << "Program Ran to Completion\n";
    system("pause");
    return 0;
}