#include "libclklt.h"

ClKLT::ClKLT()
{
  cl_int error = 0;   // Used to handle error codes
  cl_platform_id platform;
  cl_command_queue queue;

  // Platform
  error = oclGetPlatformID(&platform);
  if (error != CL_SUCCESS) {
      cout << "Error getting platform id: " << errorMessage(error) << endl;
      exit(error);
    }
  // Device
  error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) {
      cout << "Error getting device ids: " << errorMessage(error) << endl;
      exit(error);
    }
  // Context
  context = clCreateContext(0, 1, &device, NULL, NULL, &error);
  if (error != CL_SUCCESS) {
      cout << "Error creating context: " << errorMessage(error) << endl;
      exit(error);
    }
  // Command-queue
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (error != CL_SUCCESS) {
      cout << "Error creating command queue: " << errorMessage(error) << endl;
      exit(error);
    }

}

void ClKLT::AllocateMemory() {
    // Allocate memory on host :

    CHECK_OPENCL_ERROR()

 /*
    // Allocate memory on device :
    cudaMalloc((void**)&gpu_img_prev1_RGB,  sizeof(char) * w * h * 3);
    cudaMalloc((void**)&gpu_img_prev2_RGB,  sizeof(char) * w * h * 3);
    cudaMalloc((void**)&gpu_img_cur1_RGB,   sizeof(char) * w * h * 3);
    cudaMalloc((void**)&gpu_img_cur2_RGB,   sizeof(char) * w * h * 3);

    // Allocate Pyramids
    cudaMalloc((void**)&gpu_img_pyramid_prev1[0], sizeof(float) * w * h);
    cudaMalloc((void**)&gpu_img_pyramid_prev2[0], sizeof(float) * w * h);
    cudaMalloc((void**)&gpu_img_pyramid_cur1[0],  sizeof(float) * w * h);
    cudaMalloc((void**)&gpu_img_pyramid_cur2[0],  sizeof(float) * w * h);

    // Allocate smoothed pictures (for pyramid building)
    cudaMalloc((void**)&gpu_smoothed_prev1_x, sizeof(float) * w * h);
    cudaMalloc((void**)&gpu_smoothed_prev2_x, sizeof(float) * w * h);
    cudaMalloc((void**)&gpu_smoothed_cur1_x,  sizeof(float) * w * h);
    cudaMalloc((void**)&gpu_smoothed_cur2_x,  sizeof(float) * w * h);

    cudaMalloc((void**)&gpu_smoothed_prev1, sizeof(float) * w * h);
    cudaMalloc((void**)&gpu_smoothed_prev2, sizeof(float) * w * h);
    cudaMalloc((void**)&gpu_smoothed_cur1 , sizeof(float) * w * h);
    cudaMalloc((void**)&gpu_smoothed_cur2 , sizeof(float) * w * h);

    // Allocate spatial derivatives & pict buffer
    cudaMalloc((void**)&gpu_deriv_x,   sizeof(float) * w * h);
    cudaMalloc((void**)&gpu_deriv_y,   sizeof(float) * w * h);

    // Allocate LK compute intermediates :
    cudaMalloc((void **) &gpu_neighbourhood_det, sizeof(float) * MAX_POINTS);
    cudaMalloc((void **) &gpu_neighbourhood_Iyy, sizeof(float) * MAX_POINTS);
    cudaMalloc((void **) &gpu_neighbourhood_Ixy, sizeof(float) * MAX_POINTS);
    cudaMalloc((void **) &gpu_neighbourhood_Ixx, sizeof(float) * MAX_POINTS);

    // Indexes
    cudaMalloc ((void**) &gpu_pt_indexes , 2 * MAX_POINTS * sizeof(float));

    // Textures - Keep all the gradients in texture space, everytime !
    cudaMallocArray(&gpu_array_pict_0, &gpu_textr_pict_0.channelDesc, 2 * w, h);
    cudaMallocArray(&gpu_array_pict_1, &gpu_textr_pict_0.channelDesc, 2 * w, h);
    cudaMallocArray(&gpu_array_pict_2, &gpu_textr_pict_0.channelDesc, 2 * w, h);
    cudaMallocArray(&gpu_array_pict_3, &gpu_textr_pict_0.channelDesc, 2 * w, h);

    cudaMallocArray(&gpu_array_deriv_x_0, &gpu_textr_deriv_x.channelDesc, 2 * w,h); // the array will receive all the pyramid levels !
    cudaMallocArray(&gpu_array_deriv_y_0, &gpu_textr_deriv_y.channelDesc, 2 * w,h);
    cudaMallocArray(&gpu_array_deriv_x_1, &gpu_textr_deriv_x.channelDesc, 2 * w,h);
    cudaMallocArray(&gpu_array_deriv_y_1, &gpu_textr_deriv_y.channelDesc, 2 * w,h);
    cudaMallocArray(&gpu_array_deriv_x_2, &gpu_textr_deriv_x.channelDesc, 2 * w,h);
    cudaMallocArray(&gpu_array_deriv_y_2, &gpu_textr_deriv_y.channelDesc, 2 * w,h);
    cudaMallocArray(&gpu_array_deriv_x_3, &gpu_textr_deriv_x.channelDesc, 2 * w,h);
    cudaMallocArray(&gpu_array_deriv_y_3, &gpu_textr_deriv_y.channelDesc, 2 * w,h);

    setupTextures();

    // Displacements
    cudaMalloc((void**)&gpu_dx, sizeof(float) * MAX_POINTS);
    cudaMalloc((void**)&gpu_dy, sizeof(float) * MAX_POINTS);

    cudaMalloc((void**)&gpu_dx1, sizeof(float) * MAX_POINTS);
    cudaMalloc((void**)&gpu_dy1, sizeof(float) * MAX_POINTS);

    cudaMalloc((void**)&gpu_dx2, sizeof(float) * MAX_POINTS);
    cudaMalloc((void**)&gpu_dy2, sizeof(float) * MAX_POINTS);

    cudaMalloc((void**)&gpu_dx3, sizeof(float) * MAX_POINTS);
    cudaMalloc((void**)&gpu_dy3, sizeof(float) * MAX_POINTS);


    // Check GPU status
    cudaMalloc((void**)&gpu_status, sizeof(char) * MAX_POINTS);

    // Pyramids
    int _w = w;
    int _h = h;

    // Allocate pinned memory on host
    cudaHostAlloc((void**)&dx1, MAX_POINTS * sizeof(float), 0);
    cudaHostAlloc((void**)&dy1, MAX_POINTS * sizeof(float), 0);

    cudaHostAlloc((void**)&dx2, MAX_POINTS * sizeof(float), 0);
    cudaHostAlloc((void**)&dy2, MAX_POINTS * sizeof(float), 0);

    cudaHostAlloc((void**)&dx3, MAX_POINTS * sizeof(float), 0);
    cudaHostAlloc((void**)&dy3, MAX_POINTS * sizeof(float), 0);

    cudaHostAlloc((void**)&dx4, MAX_POINTS * sizeof(float), 0);
    cudaHostAlloc((void**)&dy4, MAX_POINTS * sizeof(float), 0);

    cudaHostAlloc((void**)&status, MAX_POINTS * sizeof(char), 0);

    checkCUDAError ("Memory Allocation");

    pyr_w[0] = w;
    pyr_h[0] = h;

    for(int i=1; i < _n_pyramids; ++i) {
      _w /= 2;
      _h /= 2;
      pyr_w[i] = _w;  // Pyramid size
      pyr_h[i] = _h;

      cudaMalloc((void**)&gpu_img_pyramid_prev1[i], sizeof(float)*_w*_h);
      cudaMalloc((void**)&gpu_img_pyramid_prev2[i], sizeof(float)*_w*_h);
      cudaMalloc((void**)&gpu_img_pyramid_cur1[i] , sizeof(float)*_w*_h);
      cudaMalloc((void**)&gpu_img_pyramid_cur2[i] , sizeof(float)*_w*_h);
    }

    // That's all, folks
    this->b_mem4_allocated = true;
*/


    cl_mem clCreateImage (	cl_context context,
      cl_mem_flags flags,
      const cl_image_format *image_format,
      const cl_image_desc *image_desc,
      void *host_ptr,
      cl_int *errcode_ret)




  }
