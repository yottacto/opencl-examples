// ml:run = time -p $bin > output
// ml:ccf += -fno-vectorize
// ml:ldf += -lOpenCL
#include <iostream>
#include <iterator>
#include <numeric>
#include <CL/cl.hpp>

int main()
{
    std::cout << "\n";

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        return 1;
    }

    cl::Platform default_platform{all_platforms[0]};
    std::cout << "Using platform: "
        << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        return 1;
    }

    cl::Device default_device{all_devices[0]};
    std::cout << "Using device: "
        << default_device.getInfo<CL_DEVICE_NAME>() << "\n\n";

    cl::Context context{default_device};
    cl::Program::Sources sources;

    std::string kernel_code{R"(
        void kernel add(global int const* A, global int const* B, global int* C)
        {
            C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
        }
    )"};

    sources.emplace_back(kernel_code.c_str(), kernel_code.size());

    cl::Program program{context, sources};
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cout << "Error building: "
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        return 1;
    }

    using value_type = double;
    auto constexpr size = 10'000'000u;
    auto constexpr bsize = size * sizeof(int);

    cl::Buffer buffer_a{context, CL_MEM_READ_WRITE, bsize};
    cl::Buffer buffer_b{context, CL_MEM_READ_WRITE, bsize};
    cl::Buffer buffer_c{context, CL_MEM_READ_WRITE, bsize};

    std::vector<value_type> a(size);
    std::vector<value_type> b(size);

    std::iota(std::begin(a), std::end(a), 0);
    std::iota(std::begin(b), std::end(b), 0);

    cl::CommandQueue queue{context, default_device};

    queue.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, bsize, a.data());
    queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, bsize, b.data());

    auto add = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&>(program,"add");
    cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(size), cl::NullRange};
    add(eargs, buffer_a,buffer_b,buffer_c).wait();

    std::vector<value_type> c(size);

    queue.enqueueReadBuffer(buffer_c, CL_TRUE, 0, bsize, c.data());

    std::cout << "  result: \n";
    for (auto i : c)
        std::cout << i << " ";
    std::cout << "\n";
}

