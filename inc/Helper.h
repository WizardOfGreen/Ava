#include <iostream>
#include <cmath>

double epsilon = 1e-9;

bool compare(double x, double y)
{
    double res = std::abs(x - y);
    if (res > epsilon)
        return true;

    return false;
}
