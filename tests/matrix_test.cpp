#include <iostream>
#include <cassert>
// #include "your_header.h" // Include your headers here if needed

// A simple function to test (or you can call functions from your src/inc files)
int add(int a, int b) {
    return a + b;
}

int main() {
    std::cout << "Running basic sanity check tests...\n";

    // Test 1: Basic assertion that will pass
    assert(add(2, 3) == 5);

    // Test 2: Another passing check
    assert(add(-1, 1) == 0);

    std::cout << "All tests passed successfully!\n";
    return 0;
}