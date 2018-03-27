// ml:ldf += -lOpenCL
#include <iostream>

#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/operator.hpp>

namespace compute = boost::compute;

int main()
{
    float a[] = { 1, 2, 3, 4 };
    float b[] = { 5, 6, 7, 8 };
    float c[] = { 0, 0, 0, 0 };

    compute::vector<float> vector_a(a, a + 4);
    compute::vector<float> vector_b(b, b + 4);
    compute::vector<float> vector_c(4);

    compute::transform(
        vector_a.begin(),
        vector_a.end(),
        vector_b.begin(),
        vector_c.begin(),
        compute::plus<float>()
    );

    compute::copy(vector_c.begin(), vector_c.end(), c);

    std::cout << "c: [" << c[0] << ", "
                        << c[1] << ", "
                        << c[2] << ", "
                        << c[3] << "]" << std::endl;
}

