#ifndef READER_CPP
#define READER_CPP

#include "inc/Reader.h"

Reader::Reader()
{
}

Reader::~Reader()
{
}

void Reader::ReadFile(const std::string &path)
{
    if (path.compare("") == 0)
    {
        std::cout << "Path is Empty\n";
        return;
    }

    std::fstream Data;
    std::string line;

    try
    {
        Data.open(path);

        while (getline(Data, line))
        {
            ReadingData(line) ; 
        }

        std::cout << "Data Has been Loaded\n" ;
    }
    catch (...)
    {
        std::cerr << "An Error Occurred\n";
    }
}

void Reader::ReadingData(std::string &line)
{
    int size , pos ; 
    size = data.size() ;  
    data.resize( size + 1 ) ; 
    std::string val;

    while (line.compare("") != 0 )
    {
        pos = line.find(",");
        if (pos != -1)
        {
            val = line.substr(0, pos);
            line.erase(0, pos + 1);

            data[size].push_back(val) ; 
        }
        else 
        {
            val = line ;
            line = ""  ;
            data[size].push_back(val) ;  
        }
    }

}

const std::vector<std::vector<std::string>> &Reader::ReturnData()
{
    return this->data ; 
}

void Reader::printData()
{
        for ( std::vector<std::string> v : data)
    {
        for ( std::string y : v )
        {
            std::cout << y << " " ;
        }
        std::cout << std::endl ; 
    }
}

#endif
