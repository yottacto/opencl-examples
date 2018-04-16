// ml:run = time -p $bin
// ml:ccf += -fno-vectorize
// ml:ldf += -lOpenCL
#include <iostream>
#include <fstream>
#include <iterator>
#include <numeric>
#include <CL/cl.hpp>
#include "timer.hh"

#define HARDWARE_INFO 0
#define OUTPUT_RESULTS 0
#define TRACE_STATUS 1

using value_type = int;
// auto constexpr KB = 1024;
// auto constexpr MB = 1024*KB;

// auto constexpr bsize = 64*MB;
// auto constexpr size  = bsize / sizeof(value_type);
auto constexpr size  = 100000;
auto constexpr bsize = size * sizeof(value_type);

void test_event(cl::Event& e, std::string const& name = "")
{
#if TRACE_STATUS
    std::string output{"status for event \e[33m" + name + " \e[0m:\n"};
    for (auto i = 0; ; i++) {
        cl_int status;
        e.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status);
        std::string str;
        switch (status) {
            case CL_QUEUED:    str = "CL_QUEUED";    break;
            case CL_SUBMITTED: str = "CL_SUBMITTED"; break;
            case CL_RUNNING:   str = "CL_RUNNING";   break;
            case CL_COMPLETE:  str = "CL_COMPLETE";  break;
        }
        output += "round [" + std::to_string(i) + "],\tstatus = \e[0;36m[" + str + "]\e[0m\n";
        if (status == CL_COMPLETE)
            break;
    }
    std::cout << output << "\n";
#endif
}

int main()
{

    // {
    //     std::fstream fin{"config"};
    //     fin >> size;
    //     bsize = size * sizeof(value_type);
    // }

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cerr << " No platforms found. Check OpenCL installation!\n";
        return 1;
    }

    cl::Platform default_platform{all_platforms[0]};
#if HARDWARE_INFO
    std::cout << "Using platform: "
        << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
#endif

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cerr << " No devices found. Check OpenCL installation!\n";
        return 1;
    }

    cl::Device default_device{all_devices[0]};
#if HARDWARE_INFO
    std::cout << "Using device: "
        << default_device.getInfo<CL_DEVICE_NAME>() << "\n\n";
#endif

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
        std::cout << "Error building: "
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        return 1;
    }

    cl::CommandQueue queue{context, default_device};

    cl::Buffer buffer_a{context, CL_MEM_READ_WRITE, bsize};
    cl::Buffer buffer_b{context, CL_MEM_READ_WRITE, bsize};
    cl::Buffer buffer_c{context, CL_MEM_READ_WRITE, bsize};

    std::vector<value_type> a(size);
    std::iota(std::begin(a), std::end(a), 0);

    std::vector<cl::Event> event1(1);
    std::vector<cl::Event> event2(1);

    // utils::timer t1;
    // t1.start();
    queue.enqueueWriteBuffer(
        buffer_a,
        CL_FALSE,
        0,
        bsize,
        a.data(),
        nullptr,
        &event1[0]
    );

    test_event(event1[0], "copy host data to buffer_a");

    // t1.stop();
    // std::cout << t1.elapsed_microseconds() << "us\n";

    // utils::timer t2;
    // t2.start();
    queue.enqueueWriteBuffer(
        buffer_b,
        CL_FALSE,
        0,
        bsize,
        a.data(),
        // &event1,
        nullptr,
        &event2[0]
    );

    test_event(event2[0], "copy host data to buffer_b");

    // t2.stop();
    // std::cout << t2.elapsed_microseconds() << "us\n";

    // event2[0].wait();

    auto add = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&>(program, "add");
    cl::EnqueueArgs eargs1{queue, cl::NDRange(size)};
    event1[0] = add(eargs1, buffer_a, buffer_b, buffer_c);

    test_event(event1[0], "excuting kernel [add]");

    auto mul_minus = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&>(program, "mul_minus");
    cl::EnqueueArgs eargs2{queue, event1[0], cl::NDRange(size)};
    event2[0] = mul_minus(eargs2, buffer_a, buffer_b, buffer_c);

    test_event(event2[0], "excuting kernel [mul_minus]");

    std::vector<value_type> c(size);
    queue.enqueueReadBuffer(buffer_c, CL_TRUE, 0, bsize, c.data(), &event2);

#if OUTPUT_RESULTS
    std::cout << "buffer_a: ";
    for (auto i : a)
        std::cout << i << "\t";
    std::cout << "\n";

    std::cout << "buffer_b: ";
    for (auto i : a)
        std::cout << i << "\t";
    std::cout << "\n";

    std::cout << "buffer_c: ";
    for (auto i : c)
        std::cout << i << "\t";
    std::cout << "\n";
#endif

    for (auto i = 0u; i < size; i++)
        if (c[i] != (a[i] + a[i]) * a[i] - a[i]) {
            std::cerr << "\e[0;31m""\nError: ""\e[0m""result is not correct at i = " << i << "\n\n";
            return 1;
        }

    std::cerr << "\n\e[0;32m""Success!\n\n""\e[0m";
}

