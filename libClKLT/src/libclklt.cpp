#include "libclklt.h"


LibClKLT::LibClKLT()
{
  cl_int error = 0;   // Used to handle error codes
  cl_platform_id platform;
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;

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

// TODO : Instanciate an OpenCL context, build a proper constructor to get the new pictures, ...


