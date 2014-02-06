#ifndef LIBCLKLT_H
#define LIBCLKLT_H

// Define constants
#define MAX_POINTS 10000   // max number of points to track
#define LEVELS      5      // number of pyramid levels


#include <iostream>
#include <CL/cl.hpp>

class ClKLT  {
  private:
    bool AllocateMemory();
    bool ReleaseMemory();

    cl_context context;                 /**< CL context */
    cl_device_id *devices;              /**< CL device list */

    cl_mem gpu_img_prev_RGB;
    cl_mem gpu_img_cur_RGB;

    // 4-Frame tracking
    cl_uchar4 *gpu_img_prev1_RGB;
    cl_uchar4 *gpu_img_prev2_RGB;

    cl_uchar4 *gpu_img_cur1_RGB;
    cl_uchar4 *gpu_img_cur2_RGB;

    // Picture buffers
    cl_mem *gpu_img_pyramid_prev1[LEVELS];
    cl_mem *gpu_img_pyramid_prev2[LEVELS];
    cl_mem *gpu_img_pyramid_cur1[LEVELS];
    cl_mem *gpu_img_pyramid_cur2[LEVELS];

    cl_mem *gpu_sobel_prev1;
    cl_mem *gpu_sobel_prev2;
    cl_mem *gpu_sobel_cur1;
    cl_mem *gpu_sobel_cur2;

    cl_mem *buff1;
    cl_mem *buff2;

    cl_mem *gpu_smoothed_prev1_x;
    cl_mem *gpu_smoothed_prev2_x;
    cl_mem *gpu_smoothed_cur1_x;
    cl_mem *gpu_smoothed_cur2_x;

    cl_mem *gpu_smoothed_prev1;
    cl_mem *gpu_smoothed_prev2;
    cl_mem *gpu_smoothed_cur1;
    cl_mem *gpu_smoothed_cur2;

    cl_mem *gpu_deriv_x;
    cl_mem *gpu_deriv_y;

    cl_mem *gpu_neighbourhood_det;
    cl_mem *gpu_neighbourhood_Iyy;
    cl_mem *gpu_neighbourhood_Ixy;
    cl_mem *gpu_neighbourhood_Ixx;

  public:
    ClKLT();
    ~ClKLT();

};

#endif // LIBCLKLT_H
