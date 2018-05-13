#include <iostream>
#include <fstream>
#include <random>

#define TYPE "float"

int main()
{
    std::ofstream fout{"mat.hh"};
    fout << R"(#pragma once
#include <array>

)";

    std::vector<int> size{4, 6, 100, 200, 500, 1000, 2000, 5000, 10000};
    // for (auto i = 0u; i < size.size(); i++) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 1);
    for (auto i = 0u; i < 2; i++) {
        fout << "std::array<std::array<" TYPE ", " << size[i] << ">, "
            << size[i] << "> mat_" << size[i] << "{{\n";
        for (auto row = 0; row < size[i]; row++) {
            fout << "\t{";
            for (auto col = 0; col < size[i]; col++)
                fout << dist(gen) << ", ";
            fout << "},\n";
        }
        fout << "}};\n\n";
    }
}

