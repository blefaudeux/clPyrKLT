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

// MIN/MAX macros
#ifndef MIN
#define MIN(a,b) ((a)<(b)) ? (a) : (b)
#endif

#ifndef MAX
#define MAX(a,b) ((a)>(b)) ? (a) : (b)
#endif

struct PictWorkspace
{
        void allocate( int const width, int const height, int const nPyramids )
        {
            m_width = width;
            m_height = height;
            size_t  pictSize = width * height;

            // Allocate picture buffers
            cudaMalloc((void**)& d_img, pictSize * 3 * sizeof(char) );

            // Allocate pyramids
            cudaMalloc((void**)&d_pyramid[0], pictSize * sizeof(float) );

            int pyrW[LEVELS], pyrH[LEVELS];

            pyrW[0] = width;
            pyrH[0] = height;
            int _w = m_width;
            int _h = m_height;

            for(int i=1; i < nPyramids; ++i) {
                _w /= 2;
                _h /= 2;
                pyrW[i] = _w;  // Pyramid size
                pyrH[i] = _h;

                cudaMalloc((void**)&d_pyramid[i], sizeof(float)*_w*_h);
            }

            // Allocate smoothed pictures (for pyramid building)
            cudaMalloc((void**)&d_smoothX, pictSize * sizeof(float));
            cudaMalloc((void**)&d_smoothY, pictSize * sizeof(float));

            // Allocate spatial derivatives
            cudaMalloc((void**)&d_derivX, pictSize * sizeof(float));
            cudaMalloc((void**)&d_derivY, pictSize * sizeof(float));
        }

        void free()
        {
            cudaFree( d_img );
            for (int i=0; i<LEVELS; ++i)
            {
                cudaFree( d_pyramid[i] );
            }

            cudaFree( d_smoothX );
            cudaFree( d_smoothY );

            cudaFree( d_derivX );
            cudaFree( d_derivY );
        }

    int m_width, m_height;

    unsigned char * d_img;
    float * d_pyramid[LEVELS];
    float * d_sobel;
    float * d_smoothX;
    float * d_smoothY;

    float * d_derivX;
    float * d_derivY;
};

class cudaLK
{
    public:
        cudaLK();

        cudaLK( int n_pyramids,
                int patch_radius,
                int n_max_points,
                bool weighted_norm);

        ~cudaLK();

        void resetDisplacements();

        void run4Frames(IplImage *cur1,
                        IplImage *cur2,
                        float *pt_to_track,
                        int  nPoints,
                        bool cvtToGrey);

        void loadPictures(const IplImage *img1,
                          const IplImage *img2,
                          bool b_CvtToGrey );

        void exportDebug(IplImage *outPict);

        void sobelFiltering(const float *pict_in,
                            const int m_width,
                            const int m_height,
                            float *pict_out);

        void sobelFilteringX(const float *pict_in,
                             const int m_width,
                             const int m_height,
                             float *pict_out);

        void sobelFilteringY(const float *pict_in,
                             const int m_width,
                             const int m_height,
                             float *pict_out);

        void dummyCall();

    private:
        void initMem();

        void initMem4Frame();

        void bindTextureUnits(cudaArray *pict0,
                              cudaArray *pict1,
                              cudaArray *deriv_x,
                              cudaArray *deriv_y);
        void buildPyramids();

        void cvtPicture(bool useCurrent, bool cvtToGrey);

        void fillDerivatives(float **pict_pyramid,
                             cudaArray *gpu_array_deriv_x,
                             cudaArray *gpu_array_deriv_y);

        void processTracking(int nPoints);

        void releaseMem();

        void swapPyramids();

        void setupTextures();

        void setSymbols(int pyr_level);

        void computeDerivatives(const float *in,
                                float       *deriv_buff_x,
                                float       *deriv_buff_y,
                                int         pyr_level,
                                cudaArray   *gpu_array_deriv_x,
                                cudaArray   *gpu_array_deriv_y);

        void checkCUDAError(const char *msg);

    private:
        bool b_use_weighted_norm;

        int m_width, m_height;
        int m_nPyramids, m_patchRadius, m_maxPoints;
        int m_pyrW[LEVELS], m_pyrH[LEVELS];
        int m_nThX, m_nThY;

        // TODO : cleanup using structures !
        //
        // |
        // V
        PictWorkspace m_imgCur1;
        PictWorkspace m_imgCur2;
        PictWorkspace m_imgPrev1;
        PictWorkspace m_imgPrev2;

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

        float *gpu_smoothed_prev1_x;
        float *gpu_smoothed_prev2_x;
        float *gpu_smoothed_cur1_x;
        float *gpu_smoothed_cur2_x;

        float *gpu_smoothed_prev1;
        float *gpu_smoothed_prev2;
        float *gpu_smoothed_cur1;
        float *gpu_smoothed_cur2;

        float *buff1;
        float *buff2;

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

        float *dx1, *dx2, *dx3, *dx4;
        float *dy1, *dy2, *dy3, *dy4;

        char *status;
};

#endif
