#include <iostream>
#include <string>

#include "Matrix.h"
#include "ActFunctions.h"
#include "Perceptron.h"

int main()
{

    Perceptron P  ;
    P.setActFunc( new StepFunc() ) ;
    P.testFunc() ;

    int epoch = 10 ;  

    std::vector<double> case1 = { 0 , 0 } ;  
    std::vector<double> case2 = { 0 , 1 } ;  
    std::vector<double> case3 = { 1 , 0 } ;  
    std::vector<double> case4 = { 1 , 1 } ;  



    std::cout << "Program Ran to Completion\n";
    system("pause");
    system("cls") ; 
    return 0;
} 