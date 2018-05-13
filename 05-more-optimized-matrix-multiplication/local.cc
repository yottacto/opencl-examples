// ml:run = time -p $bin 256
// ml:ldf += -lOpenCL
#include <iostream>
#include <iterator>
#include <fstream>
#include <iterator>
#include <numeric>
#include <random>
#include <CL/cl.hpp>
#include "timer.hh"
#include "constant.hh"

using value_type = float;
auto constexpr ts = TS;
auto constexpr rep_times = REP_TIMES;
auto size = 0u;
auto bsize = 0u;
std::string program_name{"mul1"};

#define PRINT_MATRIX 0

template <class Vec>
void print_matrix(Vec const& v, std::size_t n)
{
#if PRINT_MATRIX
    for (auto i = 0u; i < n; i++) {
        for (auto j = 0u; j < n; j++)
            std::cout << v[i * n + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
#else
#endif
}

template <class Vec>
void generate_real(Vec& a)
{
    std::random_device rd;
    std::uniform_real_distribution<value_type> dist(0., 100.);
    for (auto& i : a)
        i = dist(rd);
}

int main(int argc, char** argv)
{
    size = std::stoi(std::string{argv[1]});
    bsize = size * size * sizeof(value_type);

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        return 1;
    }

    cl::Platform default_platform{all_platforms[0]};
    // std::cout << "Using platform: "
    //     << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        return 1;
    }

    cl::Device default_device{all_devices[0]};
    // std::cout << "Using device: "
    //     << default_device.getInfo<CL_DEVICE_NAME>() << "\n\n";

    cl::Context context{default_device};

    cl::Program::Sources sources;

    std::fstream fin{"kernels.cl"};
    std::string kernel_code {
        std::istreambuf_iterator<char>{fin},
        std::istreambuf_iterator<char>{}
    };

    sources.emplace_back(kernel_code.c_str(), kernel_code.size());

    cl::Program program{context, sources};
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cerr << "Error building: "
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        return 1;
    }

    cl::Buffer buffer_a{context, CL_MEM_READ_WRITE, bsize};
    cl::Buffer buffer_b{context, CL_MEM_READ_WRITE, bsize};
    cl::Buffer buffer_c{context, CL_MEM_READ_WRITE, bsize};

    std::vector<value_type> a(size * size);
    std::vector<value_type> b(size * size);

    // generate numbers
    generate_real(a);
    generate_real(b);


    print_matrix(a, size);

    // std::iota(std::begin(a), std::end(a), 0);

    cl::CommandQueue queue{context, default_device};

    queue.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, bsize, a.data());
    queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, bsize, b.data());

    auto mul = cl::make_kernel<
        int,
        int,
        int,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&
    >(program, program_name);

    cl::EnqueueArgs eargs{queue, cl::NDRange(size, size), cl::NDRange(ts, ts)};

    auto sum_time = 0.;
    for (auto rep = 0; rep < rep_times; rep++) {
        utils::timer t;
        t.start();

        mul(eargs, size, size, size, buffer_a, buffer_b, buffer_c).wait();

        t.stop();

        sum_time += t.elapsed_seconds();
        // std::cout << t.elapsed_seconds() << "\n";
    }
    std::vector<value_type> c(size * size);

    queue.enqueueReadBuffer(buffer_c, CL_TRUE, 0, bsize, c.data());

    print_matrix(c, size);

    std::cout << sum_time / rep_times << "\n";
}

