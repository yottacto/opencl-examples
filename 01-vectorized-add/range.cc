// ml:run = time -p $bin > output
// ml:ccf += -fno-vectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
#include <iostream>
#include <functional>
#include <range/v3/all.hpp>

int main()
{
    using namespace ranges;
    using value_type = double;
    auto constexpr size = 10'000'000u;
    // auto constexpr size = 10;

    // std::vector<value_type> a = view::ints(0) | view::take(size);
    // std::vector<value_type> b = view::ints(0) | view::take(size);
    auto c = view::zip_with(
            std::plus<value_type>{},
            view::ints(0),
            view::ints(0)
        ) | view::take(size);

    for (auto i : c)
        std::cout << i << " ";
    std::cout << "\n";
}

