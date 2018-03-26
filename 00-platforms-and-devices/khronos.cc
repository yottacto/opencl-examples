// ml:run = $bin | ccze -A
// ml:ldf += -lOpenCL
#include <iostream>
#include <iomanip>
#include <sstream>
#include <CL/cl2.hpp>

template <class T>
std::string memory_GB(T const& byte)
{
    std::string size;
    {
        std::stringstream buf;
        buf << byte;
        buf >> size;
    }

    std::string size_GB;
    {
        std::stringstream buf;
        double dsize = std::stod(size) / (1<<30);
        buf << std::fixed << std::setprecision(2) << dsize;
        buf >> size_GB;
    }

    return size + " (" + size_GB + " GiB)";
}

int main()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (auto const& p : platforms) {
        auto constexpr width = 40;
        std::cout << "\n";
        std::cout << "  " << std::left << std::setw(width) << "Platform Name" << p.getInfo<CL_PLATFORM_NAME>() << "\n";

        std::vector<cl::Device> devices;

        // CL_DEVICE_TYPE_DEFAULT
        // CL_DEVICE_TYPE_CPU
        // CL_DEVICE_TYPE_GPU
        // CL_DEVICE_TYPE_ACCELERATOR
        // CL_DEVICE_TYPE_ALL
        p.getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

        for (auto const& d : devices) {
            std::cout << std::left << std::setw(width) << "Device Name"         << d.getInfo<CL_DEVICE_NAME>()                << "\n";
            std::cout << std::left << std::setw(width) << "Device Version"      << d.getInfo<CL_DEVICE_VERSION>()             << "\n";
            std::cout << std::left << std::setw(width) << "Max compute units"   << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()   << "\n";
            std::cout << std::left << std::setw(width) << "Max clock frequency" << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "MHz\n";
            std::cout << std::left << std::setw(width) << "Global memory size"  << memory_GB(d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()) << "\n";
        }
    }

    std::cout << "\n";
}

