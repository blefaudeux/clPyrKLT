/*
@author: Benjamin Lefaudeux (blefaudeux at github)

This script uses OpenCV to calibrate a batch of cameras in one run.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef CUDALK_H
#define CUDALK_H

#include <sys/time.h>
#include <math.h> // For sqrt and round functions..

#include <cuda.h>
#include <cuda_runtime.h>

// OpenCV 2.x includes
#undef __SSE2__
#include <opencv2/core/core.hpp>
#include <opencv2/legacy/compat.hpp>

// Define constants
#define MAX_POINTS 10000        // max number of points to track
#define LEVELS      5           // number of pyramid levels
#define PATCH_R     6           // default patch radius, patch size is (2*PATCH_R+1)*(2*PATCH_R+1)
#define NTHREAD_X   14
#define NTHREAD_Y   14
#define MAX_ITERATIONS 20
#define MV_THRESHOLD   .3F
#define MIN_DET FLT_EPSILON

// MIN/MAX macros
#ifndef MIN
#define MIN(a,b) ((a)<(b)) ? (a) : (b)
#endif

#ifndef MAX
#define MAX(a,b) ((a)>(b)) ? (a) : (b)
#endif

class cudaLK
{
  private:
    /*!
       * \brief initMem
       */
    void initMem();

    /*!
       * \brief initMem4Frame
       */
    void initMem4Frame();

    /*!
     * \brief fillDerivatives
     * \param pict_pyramid
     * \param gpu_array_deriv_x
     * \param gpu_array_deriv_y
     */
    void fillDerivatives(float **pict_pyramid,
                         cudaArray *gpu_array_deriv_x,
                         cudaArray *gpu_array_deriv_y);

    void bindTextureUnits(cudaArray *pict0,
                          cudaArray *pict1,
                          cudaArray *deriv_x,
                          cudaArray *deriv_y);

    /*!
       * \brief releaseMem
       */
    void releaseMem();

    /*!
       * \brief swapPyramids
       */
    void swapPyramids();

    /*!
       * \brief setupTextures
       */
    void setupTextures();

    /*!
     * \brief setSymbols
     * \param pyr_level
     */
    void setSymbols(int pyr_level);

    /*!
     * \brief computeDerivatives
     * \param in
     * \param _w
     * \param _h
     * \param deriv_buff_x
     * \param deriv_buff_y
     * \param gpu_array_deriv_x
     * \param gpu_array_deriv_y
     */
    void computeDerivatives(const float *in,
                            float       *deriv_buff_x,
                            float       *deriv_buff_y,
                            int         pyr_level,
                            cudaArray   *gpu_array_deriv_x,
                            cudaArray   *gpu_array_deriv_y);

    /*!
       * \brief checkCUDAError
       * \param msg
       */
    void checkCUDAError(const char *msg);

    // Flag
    bool b_use_weighted_norm;

    int w, h;
    int _n_pyramids, _patch_radius, _max_points;
    int pyr_w[LEVELS], pyr_h[LEVELS];
    int _n_threads_x, _n_threads_y;

    // TODO : cleanup using structures !
    //
    // |
    // V
    unsigned char *gpu_img_prev_RGB;
    unsigned char *gpu_img_cur_RGB;

    // 4-Frame tracking
    unsigned char *gpu_img_prev1_RGB;
    unsigned char *gpu_img_prev2_RGB;

    unsigned char *gpu_img_cur1_RGB;
    unsigned char *gpu_img_cur2_RGB;

    // Picture buffers
    float *gpu_img_pyramid_prev1[LEVELS];
    float *gpu_img_pyramid_prev2[LEVELS];
    float *gpu_img_pyramid_cur1[LEVELS];
    float *gpu_img_pyramid_cur2[LEVELS];

    float *gpu_sobel_prev1;
    float *gpu_sobel_prev2;
    float *gpu_sobel_cur1;
    float *gpu_sobel_cur2;

    float *buff1;
    float *buff2;

    float *gpu_smoothed_prev1_x;
    float *gpu_smoothed_prev2_x;
    float *gpu_smoothed_cur1_x;
    float *gpu_smoothed_cur2_x;

    float *gpu_smoothed_prev1;
    float *gpu_smoothed_prev2;
    float *gpu_smoothed_cur1;
    float *gpu_smoothed_cur2;

    float *gpu_deriv_x;
    float *gpu_deriv_y;

    float *gpu_neighbourhood_det;
    float *gpu_neighbourhood_Iyy;
    float *gpu_neighbourhood_Ixy;
    float *gpu_neighbourhood_Ixx;



    // Texture buffers
    cudaArray *gpu_array_pict_0;
    cudaArray *gpu_array_pict_1;
    cudaArray *gpu_array_pict_2;
    cudaArray *gpu_array_pict_3;

    cudaArray *gpu_array_deriv_x_0;
    cudaArray *gpu_array_deriv_y_0;

    cudaArray *gpu_array_deriv_x_1;
    cudaArray *gpu_array_deriv_y_1;

    cudaArray *gpu_array_deriv_x_2;
    cudaArray *gpu_array_deriv_y_2;

    cudaArray *gpu_array_deriv_x_3;
    cudaArray *gpu_array_deriv_y_3;
    // \4-Frame tracking

    bool   b_mem_allocated;
    bool   b_mem4_allocated;
    bool   b_first_time;

    float *gpu_smoothed_prev_x;
    float *gpu_smoothed_cur_x;
    float *gpu_smoothed_prev;
    float *gpu_smoothed_cur;

    float *gpu_pt_indexes;  // Indexes of the points to follow


    cudaArray *gpu_array_pyramid_prev;
    cudaArray *gpu_array_pyramid_prev_Ix;
    cudaArray *gpu_array_pyramid_prev_Iy;
    cudaArray *gpu_array_pyramid_cur;

    float   *gpu_dx, *gpu_dy;
    float   *gpu_dx1, *gpu_dy1;
    float   *gpu_dx2, *gpu_dy2;
    float   *gpu_dx3, *gpu_dy3;

    char    *gpu_status;

    timespec diffTime(timespec start, timespec end);



  public:
    /*!
     * \brief cudaLK
     */
    cudaLK();

    /*!
     * \brief cudaLK
     * \param levels
     * \param patch_radius
     */
    cudaLK(int n_pyramids, int patch_radius, int n_max_points, bool weighted_norm);

    ~cudaLK();

    /*!
     * \brief resetDisplacements
     * Resets the displacement arrays to 0
     */
    void resetDisplacements();


    /*!
     * \brief run4Frames
     * \param cur1
     * \param cur2
     * \param _w
     * \param _h
     * \param pt_to_track
     * \param n_pts
     * \param patch_R
     * \param b_CvtToGrey
     */
    void run4Frames(IplImage *cur1, IplImage *cur2, float *pt_to_track, int  n_pts, bool b_CvtToGrey) ;

    /*!
     * \brief loadBackPictures
     * \param prev1
     * \param prev2
     * \param _w
     * \param _h
     * \param b_CvtToGrey
     */
    void loadBackPictures(const IplImage *prev1, const IplImage *prev2, bool b_CvtToGrey);

    /*!
     * \brief loadCurPictures
     * \param cur1
     * \param cur2
     * \param _w
     * \param _h
     * \param b_CvtToGrey
     */
    void loadCurPictures(const IplImage *cur1, const IplImage *cur2, bool b_CvtToGrey);

    /*!
     * \brief exportDebug
     * \param outPict
     */
    void exportDebug(IplImage *outPict);

    /*!
     * \brief sobelFiltering : Compute the Sobel gradient picture using kernels
     * \param pict_in
     * \param pict_out
     */
    void sobelFiltering(const float *pict_in,
                        const int w,
                        const int h,
                        float *pict_out);

    /*!
     * \brief sobelFiltering : Compute the Sobel gradient picture using kernels
     * \param pict_in
     * \param pict_out
     */
    void sobelFilteringX(const float *pict_in,
                         const int w,
                         const int h,
                         float *pict_out);

    /*!
     * \brief sobelFiltering : Compute the Sobel gradient picture using kernels
     * \param pict_in
     * \param pict_out
     */
    void sobelFilteringY(const float *pict_in,
                         const int w,
                         const int h,
                         float *pict_out);

    /*!
     * \brief dummyCall
     */
    void dummyCall();

    /*!
     * Coordinates of the tracked points
     */
    float *dx1, *dx2, *dx3, *dx4;
    float *dy1, *dy2, *dy3, *dy4;

    char *status;
};

#endif
