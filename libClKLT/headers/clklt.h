#ifndef LIBCLKLT_H
#define LIBCLKLT_H

// Define constants
#define MAX_POINTS 10000   // max number of points to track
#define LEVELS      5      // number of pyramid levels


#include <iostream>
#include <CL/cl.hpp>

class ClKLT  {
  private:
    bool allocateMemory();
    bool releaseMemory();

    bool b_mem_allocated;

    unsigned int _size_x, _size_y, _max_points;

    cl_context context;                 /**< CL context */
    cl_device_id *devices;              /**< CL device list */

    cl_mem gpu_img_prev_RGB;
    cl_mem gpu_img_cur_RGB;

    // Features buffers
    void *gpu_initial_pose, *gpu_next_pose, *status;

    cl_mem pinned_initial_pose;
    cl_mem pinned_next_pose;

    float *next_pose, *initial_pose;

    // Picture buffers
    cl_image_desc pict_params;

    cl_uchar4 *gpu_img_prev_RGB;
    cl_uchar4 *gpu_img_cur_RGB;

    cl_mem gpu_img_pyramid_prev[LEVELS];
    cl_mem gpu_img_pyramid_cur[LEVELS];

    cl_mem gpu_sobel_prev;
    cl_mem gpu_sobel_cur;

    cl_mem gpu_smoothed_prev_x;
    cl_mem gpu_smoothed_cur_x;

    cl_mem gpu_smoothed_prev;
    cl_mem gpu_smoothed_cur;

    cl_mem gpu_deriv_x;
    cl_mem gpu_deriv_y;

    cl_mem gpu_neighbourhood_det;
    cl_mem gpu_neighbourhood_Iyy;
    cl_mem gpu_neighbourhood_Ixy;
    cl_mem gpu_neighbourhood_Ixx;

  public:
    ClKLT();
    ~ClKLT();

    /*!
     * \brief newFrame
     * \param data
     * \param size_x
     * \param size_y
     */
    void newFrame(unsigned char *data,
                  int _size_x,
                  int _size_y) ;

};

#endif // LIBCLKLT_H
