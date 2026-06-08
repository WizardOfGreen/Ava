#ifndef READER_H
#define READER_H

#include <iostream>
#include <string>
#include <vector>

#include <fstream>

class Reader
{
private:
    std::string path;
    std::vector<std::vector<std::string>> data ; 
public:
    Reader(/* args */);
    ~Reader();

    void ReadFile(const std::string &path);
    void ReadingData(std::string &line) ; 
    const std::vector<std::vector<std::string>>& ReturnData();

    void printData() ; 
};

#endif