#include "clklt.h"

ClKLT::ClKLT()  {
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


  _max_points = MAX_POINTS;


  // TODO : Create CL context


  // TODO :

  this->b_mem_allocated = false;
}

void ClKLT::allocateMemory() {

  if (this->b_mem_allocated)
      return;


  // Let's suppose that we know the picture size..


  // Allocate memory on host :

  // CHECK_OPENCL_ERROR()

  /*
  // Allocate memory on device :
  cudaMalloc((void**)&gpu_img_prev_RGB,  sizeof(char) * w * h * 3);
  cudaMalloc((void**)&gpu_img_cur_RGB,   sizeof(char) * w * h * 3);

  // Allocate Pyramids
  cudaMalloc((void**)&gpu_img_pyramid_prev[0], sizeof(float) * w * h);
  cudaMalloc((void**)&gpu_img_pyramid_cur[0],  sizeof(float) * w * h);

  // Allocate smoothed pictures (for pyramid building)
  cudaMalloc((void**)&gpu_smoothed_prev_x, sizeof(float) * w * h);
  cudaMalloc((void**)&gpu_smoothed_cur_x,  sizeof(float) * w * h);

  cudaMalloc((void**)&gpu_smoothed_prev, sizeof(float) * w * h);
  cudaMalloc((void**)&gpu_smoothed_cur , sizeof(float) * w * h);

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

  cudaMallocArray(&gpu_array_deriv_x, &gpu_textr_deriv_x.channelDesc, 2 * w,h); // the array will receive all the pyramid levels !
  cudaMallocArray(&gpu_array_deriv_y, &gpu_textr_deriv_y.channelDesc, 2 * w,h);

  setupTextures();

  // Displacements
  cudaMalloc((void**)&gpu_dx, sizeof(float) * MAX_POINTS);
  cudaMalloc((void**)&gpu_dy, sizeof(float) * MAX_POINTS);

  cudaMalloc((void**)&gpu_dx1, sizeof(float) * MAX_POINTS);
  cudaMalloc((void**)&gpu_dy1, sizeof(float) * MAX_POINTS);


  // Check GPU status
  cudaMalloc((void**)&gpu_status, sizeof(char) * MAX_POINTS);

  // Pyramids
  int _w = w;
  int _h = h;

  // Allocate pinned memory on host
  cudaHostAlloc((void**)&dx1, MAX_POINTS * sizeof(float), 0);
  cudaHostAlloc((void**)&dy1, MAX_POINTS * sizeof(float), 0);

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
  */



  // Define the picture format
  cl_image_format pict_format;
  pict_format.image_channel_data_type = CL_UNSIGNED_INT8;
  pict_format.image_channel_order = CL_RGB;

  cl_image_format pict_float_format;
  pict_format.image_channel_data_type = CL_FLOAT;
  pict_format.image_channel_order = CL_LUMINANCE;

  cl_int err_code;

  // Allocate all the picture buffers as cl_mem
  this->gpu_img_prev_RGB = clCreateImage (this->context,
    CL_MEM_READ_WRITE,
    &pict_format,
    &pict_params,
    NULL,
    &err_code);

  this->gpu_deriv_x = clCreateImage (this->context,
    CL_MEM_READ_WRITE,
    &pict_float_format,
    &pict_params,
    NULL,
    &err_code);

  this->gpu_deriv_y = clCreateImage (this->context,
    CL_MEM_READ_WRITE,
    &pict_float_format,
    &pict_params,
    NULL,
    &err_code);

  // Allocate the point buffers (not in the texture units there)
  initial_pose  = gcl_malloc(_max_points * 2 * sizeof(float), NULL, NULL);
  next_pose     = gcl_malloc(_max_points * 2 * sizeof(float), NULL, NULL);

  status        = gcl_malloc(_max_points * sizeof(char), NULL, NULL);

  this->b_mem_allocated = true;
}


void ClKLT::newFrame(unsigned char *data,
                     int size_x,
                     int size_y) {

  // Allocate memory if that's not already been done
  if (!b_mem_allocated)
    // Define the picture parameters
    pict_params.image_type  = CL_MEM_OBJECT_IMAGE2D;
    pict_params.image_width   = size_x;
    pict_params.image_height  = size_y;

    allocateMemory();

  // Compute the tracking from the previous position to the next one
  // TODO, if there was a picture already..

  // Output the results ?

}

void ClKLT::releaseMemory() {

  // clReleaseMemObject(device_mem);

}
