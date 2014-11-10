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

  // Allocate Pyramids
  this->gpu_img_pyramid_cur = clCreateImage(this->context,
                                            CL_MEM_READ_WRITE,
                                            &pict_float_format,
                                            &pict_params,
                                            NULL,
                                            &err_code);

  this->gpu_img_pyramid_prev = clCreateImage(this->context,
                                             CL_MEM_READ_WRITE,
                                             &pict_float_format,
                                             &pict_params,
                                             NULL,
                                             &err_code);

  // Define the picture parameters
  _cl_image_desc pyr_params;
  pyr_params.image_type   = CL_MEM_OBJECT_IMAGE2D;
  pyr_params.image_width  = pict_params.image_width;
  pyr_params.image_height = pict_params.image_height;

  for(int i=1; i < _n_pyramids; ++i) {
    pyr_params.image_width  /= 2;
    pyr_params.image_width  /= 2;

    this->gpu_img_pyramid_prev[i] = clCreateImage(this->context,
                                                  CL_MEM_READ_WRITE,
                                                  &pict_float_format,
                                                  &pict_params,
                                                  NULL,
                                                  &err_code);

    this->gpu_img_pyramid_cur[i] = clCreateImage(this->context,
                                                 CL_MEM_READ_WRITE,
                                                 &pict_float_format,
                                                 &pict_params,
                                                 NULL,
                                                 &err_code);
  }

  // Allocate smoothed pictures (for pyramid building)
  this->gpu_smoothed_prev_x = clCreateImage(this->context,
                                             CL_MEM_READ_WRITE,
                                             &pict_float_format,
                                             &pict_params,
                                             NULL,
                                             &err_code);

  this->gpu_smoothed_cur_x = clCreateImage(this->context,
                                             CL_MEM_READ_WRITE,
                                             &pict_float_format,
                                             &pict_params,
                                             NULL,
                                             &err_code);

  this->gpu_smoothed_prev = clCreateImage(this->context,
                                             CL_MEM_READ_WRITE,
                                             &pict_float_format,
                                             &pict_params,
                                             NULL,
                                             &err_code);

  this->gpu_smoothed_cur = clCreateImage(this->context,
                                             CL_MEM_READ_WRITE,
                                             &pict_float_format,
                                             &pict_params,
                                             NULL,
                                             &err_code);

  // Allocate the LK Tracker intermediates
  size_t buffer_size = _max_points * sizeof(float);

  gpu_neighbourhood_det = clCreateBuffer(this->context, CL_MEM_READ_WRITE, buffer_size, NULL, err_code);
  gpu_neighbourhood_Iyy = clCreateBuffer(this->context, CL_MEM_READ_WRITE, buffer_size, NULL, err_code);
  gpu_neighbourhood_Ixy = clCreateBuffer(this->context, CL_MEM_READ_WRITE, buffer_size, NULL, err_code);
  gpu_neighbourhood_Ixx = clCreateBuffer(this->context, CL_MEM_READ_WRITE, buffer_size, NULL, err_code);

  // Allocate the point buffers (not in the texture units there)
  gpu_initial_pose  = clCreateBuffer(this->context, NULL, 2 * buffer_size, NULL, err_code);
  gpu_next_pose     = clCreateBuffer(this->context, NULL, 2 * buffer_size, NULL, err_code);

  // Allocate pined memory on host to get the results back
  next_pose     = malloc(2 * buffer_size);
  initial_pose  = malloc(2 * buffer_size);

  pinned_initial_pose = clCreateBuffer(this->context, CL_MEM_READ_ONLY,
                                       2 * buffer_size, initial_pose, err_code);

  pinned_next_pose    = clCreateBuffer(this->context, CL_MEM_WRITE_ONLY ,
                                       2 * buffer_size, next_pose, err_code);

  status        = clCreateBuffer(this->context, CL_MEM_READ_WRITE, _max_points * sizeof(char), NULL, err_code);

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
