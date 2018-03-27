// ml:run = time -p $bin > output
// ml:ccf += -fno-vectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include "timer.hh"

using value_type = double;
// auto constexpr size = 100'000'000u;
auto size = 0u;

int main()
{
    {
        std::fstream fin{"config"};
        fin >> size;
    }

    std::vector<value_type> a(size);
    std::vector<value_type> b(size);
    std::vector<value_type> c(size);
    std::iota(std::begin(a), std::end(a), 0);
    std::iota(std::begin(b), std::end(b), 0);

    utils::timer t;
    t.start();

    for (auto i = 0u; i < size; i++)
        c[i] = a[i] + b[i];

    t.stop();
    // std::cerr << "time: " << t.elapsed_seconds() << "\n";
    std::cout << t.elapsed_seconds() << "\n";

    // for (auto i = 0u; i < size; i++)
    //     std::cout << c[i] << " ";
    // std::cout << "\n";
}

