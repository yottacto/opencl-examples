// ml:run = time -p $bin
// ml:ldf += -lOpenCL -I/usr/include/opencv -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_photo -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_xobjdetect -lopencv_imgcodecs -lopencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core
#include <iostream>
#include <iterator>
#include <fstream>
#include <iterator>
#include <cmath>
#include <CL/cl.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "timer.hh"

using value_type = unsigned char;
auto constexpr rep_times = 1;
float const pi = std::acos(-1.);
float const angle = pi / 3.;
auto size = 0u;
auto bsize = 0u;

int main()
{
    // std::cout << "\n";

    {
        std::fstream fin{"config"};
    }

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

    cv::Mat in_img = cv::imread("test.jpg");
    // std::cerr << in_img.type() << "\n";
    // std::cerr << in_img.elemSize() << "\n";
    // std::cerr << in_img.elemSize1() << "\n";
    // cv::Mat gray_img;
    // cv::cvtColor(in_img, gray_img, cv::COLOR_RGB2GRAY);
    // std::cerr << gray_img.elemSize() << "\n";
    // std::cerr << gray_img.elemSize1() << "\n";

    std::size_t width  = in_img.cols;
    std::size_t height = in_img.rows;

    std::cerr << "width = " << width << ", height = " << height << "\n";

    // cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Display window", gray_img);
    // cv::waitKey(0);

    // cl::ImageFormat format{CL_LUMINANCE, CL_UNORM_INT8};
    cl::ImageFormat format{CL_RGB, CL_UNORM_INT8};
    // cl::Image2D input{context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, format, width, height, 0, gray_img.data};
    cl::Image2D input{context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, format, width, height, 0, in_img.data};
    cl::Image2D output{context, CL_MEM_READ_WRITE, format, width, height};

    cl::CommandQueue queue{context, default_device};

    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;

    cl::size_t<3> region;
    region[0] = width;
    region[1] = height;
    region[2] = 1;

    // queue.enqueueWriteImage(
    //     input,
    //     CL_TRUE,
    //     origin,
    //     region,
    //     0, // row pitch
    //     0, // slice pitch
    //     in_img.data
    // );
    //     // return 0;

    auto rotate = cl::make_kernel<
        cl::Image2D&,
        cl::Image2D&,
        float
    >(program, "image_rotate");

    cl::EnqueueArgs eargs{queue, cl::NDRange(width, height)};

    auto sum_time = 0.;
    for (auto rep = 0; rep < rep_times; rep++) {
        utils::timer t;
        t.start();

        rotate(eargs, input, output, angle).wait();

        t.stop();

        sum_time += t.elapsed_seconds();
        // std::cout << t.elapsed_seconds() << "\n";
    }

    queue.enqueueReadImage(
        output,
        CL_TRUE,
        origin,
        region,
        0, // row pitch
        0, // slice pitch
        in_img.data
    );

    // cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Display window", in_img);
    // while (cv::waitKey(1000) != 27) {
    // }

    std::cout << sum_time / rep_times << "\n";
}

