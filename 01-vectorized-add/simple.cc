// ml:run = time -p $bin > output
// ml:ccf += -fno-vectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
#include <iostream>
#include <numeric>
#include <vector>

using value_type = double;
auto constexpr size = 10'000'000u;

int main()
{
    std::vector<value_type> a(size);
    std::vector<value_type> b(size);
    std::vector<value_type> c(size);
    std::iota(std::begin(a), std::end(a), 0);
    std::iota(std::begin(b), std::end(b), 0);


    for (auto i = 0u; i < size; i++)
        c[i] = a[i] + b[i];

    for (auto i = 0u; i < size; i++)
        std::cout << c[i] << " ";
    std::cout << "\n";
}

