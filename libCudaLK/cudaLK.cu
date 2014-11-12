/*
@author: Benjamin Lefaudeux (blefaudeux at github)

This program computes optical flow using the nVidia CUDA API

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

DISCLAIMER (Benjamin Lefaudeux)

Some parts of the code below are from Nghia Ho (http://nghiaho.com/)
See nghiaho12 @ yahoo.com (available without specific licence)
*/


#include "cudaLK.h"
#include <stdio.h>

#include "derivativeKernels.cu"
#include "trackingKernels.cu"

const float scaling[] = {1, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f};

// TODO :
// - constant velocity model : keep previous velocity if the point was already tracked
// - adaptative gain  ?

// ----------------------------------------------------------------------



// Possible weight coefficients for tracking cost evaluation :
// Gaussian discretisation
/*
 *       1  4  6  4  1
 *       4 16 24 16  4
 *       6 24 36 24  6
 *       4 16 24 16  4
 *       1  4  6  4  1
 */

// Compute clock time
timespec cudaLK::diffTime(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}

// Compute divUp / Useful for kernel job repartition
int iDivUp( int a,  int b )
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}




cudaLK::cudaLK()
{
    // Use default values for pyramid levels & LK search radius
    _n_pyramids     = LEVELS;
    _patch_radius   = PATCH_R;
    _max_points     = MAX_POINTS;
    _n_threads_x    = NTHREAD_X;
    _n_threads_y    = NTHREAD_Y;

    cudaMemcpyToSymbol (LK_patch, &_patch_radius, sizeof(int));   // in device constant memory
    cudaMemcpyToSymbol (LK_points, &_max_points, sizeof(int));

    // Init flags
    b_mem_allocated   = false;
    b_mem4_allocated  = false;
    b_first_time      = true;
    b_use_weighted_norm = false;
}

cudaLK::cudaLK( int n_pyramids,
                int patch_radius ,
                int n_max_points,
                bool weighted_norm )
{
    //  /!\ Memory is not allocated at this point
    // Call initMem() or initMem4Frame()

    // Specify pyramid levels and LK search radius
    _n_pyramids     = n_pyramids;
    _patch_radius   = patch_radius;
    _max_points     = n_max_points;
    _n_threads_x    = NTHREAD_X;
    _n_threads_y    = NTHREAD_Y;

    cudaMemcpyToSymbol (LK_patch, &_patch_radius, sizeof(int));   // in device constant memory
    cudaMemcpyToSymbol (LK_points, &_max_points, sizeof(int));


    // Init flags
    b_mem_allocated   = false;
    b_mem4_allocated  = false;
    b_first_time      = true;
    b_use_weighted_norm = weighted_norm;
}

cudaLK::~cudaLK()
{
    releaseMem ();
}

void cudaLK::bindTextureUnits( cudaArray *pict0,
                               cudaArray *pict1,
                               cudaArray *deriv_x,
                               cudaArray *deriv_y )
{

    cudaUnbindTexture (gpu_textr_pict_0);
    cudaUnbindTexture (gpu_textr_pict_1);
    cudaUnbindTexture (gpu_textr_deriv_x);
    cudaUnbindTexture (gpu_textr_deriv_y);

    cudaBindTextureToArray(gpu_textr_pict_0,  pict0, gpu_textr_pict_0.channelDesc);
    cudaBindTextureToArray(gpu_textr_pict_1,  pict1, gpu_textr_pict_1.channelDesc);

    cudaBindTextureToArray (gpu_textr_deriv_x,   deriv_x, gpu_textr_deriv_x.channelDesc);
    cudaBindTextureToArray (gpu_textr_deriv_y,   deriv_y, gpu_textr_deriv_y.channelDesc);
}

void cudaLK::buildPyramids()
{
    // 1D & 2D-indexing of kernels
    int blocksW = w/_n_threads_x + ((w % _n_threads_x)?1:0);
    int blocksH = h/_n_threads_y + ((h % _n_threads_y )?1:0);
    dim3 blocks(blocksW, blocksH);
    dim3 threads(_n_threads_x, _n_threads_y);

    // Build pyramids
    for(int i=0; i < _n_pyramids-1; i++) {
        kernelSmoothX<<<blocks, threads>>>(gpu_img_pyramid_prev1[i], pyr_w[i], pyr_h[i], gpu_smoothed_prev1_x);
        kernelSmoothX<<<blocks, threads>>>(gpu_img_pyramid_prev2[i], pyr_w[i], pyr_h[i], gpu_smoothed_prev2_x);

        kernelSmoothY<<<blocks, threads>>>(gpu_smoothed_prev1_x, pyr_w[i], pyr_h[i], gpu_smoothed_prev1);
        kernelSmoothY<<<blocks, threads>>>(gpu_smoothed_prev2_x, pyr_w[i], pyr_h[i], gpu_smoothed_prev2);

        pyrDownsample<<<blocks, threads>>>(gpu_smoothed_prev1, pyr_w[i], pyr_h[i], gpu_img_pyramid_prev1[i+1], pyr_w[i+1], pyr_h[i+1]);
        pyrDownsample<<<blocks, threads>>>(gpu_smoothed_prev2, pyr_w[i], pyr_h[i], gpu_img_pyramid_prev2[i+1], pyr_w[i+1], pyr_h[i+1]);
    }
}

void cudaLK::checkCUDAError(const char *msg) {
    // Check GPU status to catch errors
    // "msg" is printed in case of an exception

    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}


void cudaLK::computeDerivatives(float const *in,
                                float       *deriv_buff_x,
                                float       *deriv_buff_y,
                                int         pyr_level,
                                cudaArray   *gpu_array_deriv_x,
                                cudaArray   *gpu_array_deriv_y) {
    //
    // Compute spatial derivatives using Scharr or Sobel operator
    //

    // 2D-indexing for kernel managements
    int _n_threads_x = NTHREAD_X;
    int _n_threads_y = NTHREAD_Y;
    int _w = pyr_w[pyr_level], _h=pyr_h[pyr_level];

    int blocksW = _w/_n_threads_x + ((_w % _n_threads_x)?1:0);
    int blocksH = _h/_n_threads_y + ((_h % _n_threads_y)?1:0);

    dim3 blocks(blocksW, blocksH);
    dim3 threads(_n_threads_x, _n_threads_y);

    // Compute via kernels
    kernelScharrX <<<blocks, threads>>>(in, _w, _h, deriv_buff_x);
    kernelScharrY <<<blocks, threads>>>(in, _w, _h, deriv_buff_y);

    checkCUDAError("ComputingSpatialDerivatives");

    // Copy result to texture buffers
    int offset = 0;

    for (int i=0; i<pyr_level; ++i)
    {
        offset += pyr_w[i];
    }

    cudaMemcpy2DToArrayAsync(gpu_array_deriv_x,
                             offset * sizeof(float),
                             0,
                             deriv_buff_x,
                             sizeof(float)*_w,
                             sizeof(float)*_w,
                             _h,
                             cudaMemcpyDeviceToDevice);

    cudaMemcpy2DToArrayAsync(gpu_array_deriv_y,
                             offset * sizeof(float),
                             0,
                             deriv_buff_y,
                             sizeof(float)*_w,
                             sizeof(float)*_w,
                             _h,
                             cudaMemcpyDeviceToDevice);

    //  checkCUDAError("ComputingSpatialDerivatives-memdump");
}

void cudaLK::cvtPicture(bool useCurrent, bool cvtToGrey)
{
    int blocks1D = (w*h)/256 + (w*h % 256?1:0); // for greyscale

    if (useCurrent)
    {
        if (cvtToGrey) {
            // RGB -> grey
            convertRGBToGrey<<<blocks1D, 256>>>(gpu_img_cur1_RGB, gpu_img_pyramid_cur1[0], w*h);
            convertRGBToGrey<<<blocks1D, 256>>>(gpu_img_cur2_RGB, gpu_img_pyramid_cur2[0], w*h);
        } else {
            convertGreyToFloat<<<blocks1D, 256>>>(gpu_img_cur1_RGB, gpu_img_pyramid_cur1[0], w*h);
            convertGreyToFloat<<<blocks1D, 256>>>(gpu_img_cur2_RGB, gpu_img_pyramid_cur2[0], w*h);
        }
    }
    else
    {
        if (cvtToGrey) {
            // RGB -> grey
            convertRGBToGrey<<<blocks1D, 256>>>(gpu_img_prev1_RGB, gpu_img_pyramid_prev1[0], w*h);
            convertRGBToGrey<<<blocks1D, 256>>>(gpu_img_prev2_RGB, gpu_img_pyramid_prev2[0], w*h);
            checkCUDAError("convertRGBToGrey");
        } else {
            convertGreyToFloat<<<blocks1D, 256>>>(gpu_img_prev1_RGB, gpu_img_pyramid_prev1[0], w*h);
            convertGreyToFloat<<<blocks1D, 256>>>(gpu_img_prev2_RGB, gpu_img_pyramid_prev2[0], w*h);
            checkCUDAError("convertToFloat");
        }
    }
}

void cudaLK::exportDebug(IplImage *outPict) {
    // Debug function to see what's going on in picture buffers
    // Not reliable for IPLImages because of widthStep --> TODO ?

    // Copy buffer back to host
    float pict_x_f[w*h];
    //  float pict_y_f[w*h];

    // SOBEL
    cudaMemcpy (pict_x_f, gpu_img_pyramid_prev1[0], w*h*sizeof(float), cudaMemcpyDeviceToHost);
    //  cudaMemcpy (pict_x_f, gpu_img_pyramid_cur1[0], w*h*sizeof(float), cudaMemcpyDeviceToHost);

    // Get picture max value
    float val = 0.f;
    float max_val = 0.f;
    for (int i = 0; i<w; ++i) {
        for (int j=0; j<h; ++j) {
            val = pict_x_f[i +j*w];
            if (val > max_val)
                max_val = val;
        }
    }

    // Convert to char
    for (int i = 0; i<w; ++i) {
        for (int j=0; j<h; ++j) {
            val = pict_x_f[i +j*w];
            outPict->imageData[i +j*outPict->widthStep] = (unsigned char) round(val/max_val*254);
        }
    }

    // Check derivatives

    //  cudaMemcpy(pict_x_f, gpu_deriv_x, w*h*sizeof(float), cudaMemcpyDeviceToHost);
    //  cudaMemcpy(pict_y_f, gpu_deriv_y, w*h*sizeof(float), cudaMemcpyDeviceToHost);
    //  checkCUDAError ("ExportDebugFunction");

    //  // Get picture max value
    //  float val = 0.f;
    //  float max_val = 0.f;
    //  for (int i = 0; i<w; ++i) {
    //    for (int j=0; j<h; ++j) {
    //      val = sqrt(pict_x_f[i +j*w]*pict_x_f[i +j*w] + pict_y_f[i +j*w]*pict_y_f[i +j*w]);
    ////      val = pict_y_f[i +j*w];
    //      if (val > max_val)
    //        max_val = val;
    //    }

    //  }

    //  printf("ExportDebug max value : %d x %d - %f\n", w, h, max_val);

    //  // Convert to char
    //  for (int i = 0; i<w; ++i) {
    //    for (int j=0; j<h; ++j) {
    //      val = sqrt(pict_x_f[i +j*w]*pict_x_f[i +j*w] + pict_y_f[i +j*w]*pict_y_f[i +j*w]);
    ////      val = pict_y_f[i +j*w];
    //      outPict->imageData[i +j*outPict->widthStep] = (unsigned char) round(val/max_val*254);
    //    }
    //  }

    checkCUDAError ("Debug exportation");
}

void cudaLK::initMem()
{
    // Picture buffers
    cudaMalloc((void**)&gpu_img_prev_RGB, sizeof(char)*w*h*3);
    cudaMalloc((void**)&gpu_img_cur_RGB, sizeof(char)*w*h*3);

    cudaMalloc((void**)&gpu_img_pyramid_prev1[0], sizeof(float)*w*h);
    cudaMalloc((void**)&gpu_img_pyramid_cur1[0], sizeof(float)*w*h);

    cudaMalloc((void**)&gpu_smoothed_prev_x, sizeof(float)*w*h);
    cudaMalloc((void**)&gpu_smoothed_cur_x, sizeof(float)*w*h);

    cudaMalloc((void**)&gpu_smoothed_prev, sizeof(float)*w*h);
    cudaMalloc((void**)&gpu_smoothed_cur, sizeof(float)*w*h);

    // Indexes
    cudaMalloc ((void**) &gpu_pt_indexes, 2*MAX_POINTS*sizeof(float));

    // Texture
    cudaMallocArray(&gpu_array_pyramid_prev, &texRef_pyramid_prev.channelDesc, w, h);
    cudaMallocArray(&gpu_array_pyramid_cur, &texRef_pyramid_cur.channelDesc, w, h);

    cudaBindTextureToArray(texRef_pyramid_prev, gpu_array_pyramid_prev, texRef_pyramid_prev.channelDesc);
    cudaBindTextureToArray(texRef_pyramid_cur,  gpu_array_pyramid_cur,  texRef_pyramid_cur.channelDesc);

    texRef_pyramid_prev.normalized = 0;
    texRef_pyramid_prev.filterMode = cudaFilterModeLinear;
    texRef_pyramid_prev.addressMode[0] = cudaAddressModeClamp;
    texRef_pyramid_prev.addressMode[1] = cudaAddressModeClamp;

    texRef_pyramid_cur.normalized = 0;
    texRef_pyramid_cur.filterMode = cudaFilterModeLinear;
    texRef_pyramid_cur.addressMode[0] = cudaAddressModeClamp;
    texRef_pyramid_cur.addressMode[1] = cudaAddressModeClamp;

    cudaMalloc((void**)&gpu_dx, sizeof(float)*w*h);
    cudaMalloc((void**)&gpu_dy, sizeof(float)*w*h);
    cudaMalloc((void**)&gpu_status, sizeof(char)*w*h);

    int _w = w;
    int _h = h;

    dx1 = new float[w*h];
    dy1 = new float[w*h];
    status = new char[w*h];

    pyr_w[0] = w;
    pyr_h[0] = h;

    for(int i=1; i < _n_pyramids; ++i)
    {
        _w /= 2;
        _h /= 2;
        pyr_w[i] = _w;
        pyr_h[i] = _h;

        cudaMalloc((void**)&gpu_img_pyramid_prev1[i], sizeof(float)*_w*_h);
        cudaMalloc((void**)&gpu_img_pyramid_cur1[i], sizeof(float)*_w*_h);
    }

    b_mem_allocated = true;
    printf("[CudaKLT]: Memory allocated\n");
}

void cudaLK::initMem4Frame()
{
    // Allocate picture buffers
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

    checkCUDAError ("Allocating 4Frames memory");
    printf("[CucaKLT] : 4Frames memory allocated\n");
}

void cudaLK::dummyCall()
{
    // Do something on the GPU to wake up the beast..
    int dummy = 1;
    cudaMemcpyToSymbol (LK_width, &dummy, sizeof(int));
}

void cudaLK::fillDerivatives(float **pict_pyramid,
                             cudaArray *gpu_array_deriv_x,
                             cudaArray *gpu_array_deriv_y)
{
    // Compute derivatives & load them into texture units
    for(int l = _n_pyramids-1; l >= 0; l--) {
        computeDerivatives(pict_pyramid[l],
                           gpu_deriv_x,       // Buffers
                           gpu_deriv_y,
                           l,
                           gpu_array_deriv_x, // Final texture recipients
                           gpu_array_deriv_y);
    }
}

void cudaLK::loadBackPictures(const IplImage *prev1,
                              const IplImage *prev2,
                              bool b_CvtToGrey) {

    // Load initial pictures to be used in backbuffers (and allocate memory if nedded)
    // Just called once

    // Allocate memory if needed
    if (!b_mem4_allocated) {
        w = prev1->width;
        h = prev1->height;

        // Initiate constant memory variables
        cudaMemcpyToSymbol (LK_width, &w, sizeof(w));
        cudaMemcpyToSymbol (LK_height, &h, sizeof(h));

        initMem4Frame ();

        int n_iterations  = MAX_ITERATIONS;
        float threshold   = MV_THRESHOLD;
        cudaMemcpyToSymbol (LK_iteration, &n_iterations, sizeof(int));
        cudaMemcpyToSymbol (LK_threshold, &threshold, sizeof(float));

        // Init weighting parameters, if needed :
        float temp_weight_array[MAX_WEIGHT_VALUES];

        if (w*h > MAX_WEIGHT_VALUES) {
            // Window is too big.. no weighting for now
            this->b_use_weighted_norm = false;
        }

        if (this->b_use_weighted_norm) {
            for (int i = -w; i<= w; ++i) {
                for (int j = -h; j<= h; ++j) {
                    temp_weight_array[i + j*w] = exp (-(i*j)/10.f); // TODO : handle std settings gracefully..
                }
            }

            cudaMemcpyToSymbol (LK_Weight, &temp_weight_array, w*h*sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    checkCUDAError("LoadBackPicture - set symbols");

    // Transfer from host memspace to gpu memspace
    if (b_CvtToGrey) {
        cudaMemcpy2D (gpu_img_prev1_RGB, w*sizeof(uchar), prev1->imageData, prev1->widthStep, 3 * prev1->width * sizeof(uchar), prev1->height, cudaMemcpyHostToDevice );
        cudaMemcpy2D (gpu_img_prev2_RGB, w*sizeof(uchar), prev2->imageData, prev2->widthStep, 3 * prev2->width * sizeof(uchar), prev2->height, cudaMemcpyHostToDevice );
    } else {
        cudaMemcpy2D (gpu_img_prev1_RGB, w*sizeof(uchar), prev1->imageData, prev1->widthStep, prev1->width * sizeof(uchar), prev1->height, cudaMemcpyHostToDevice );
        cudaMemcpy2D (gpu_img_prev2_RGB, w*sizeof(uchar), prev2->imageData, prev2->widthStep, prev2->width * sizeof(uchar), prev2->height, cudaMemcpyHostToDevice );
    }
    checkCUDAError("LoadBackPicture");


    // Convert picture to floats & grey
    cvtPicture(false, b_CvtToGrey);
    buildPyramids();
    checkCUDAError("pyrDownsample");

    // Load cudaArray buffer from pyramids
    int pyr_offset = 0;
    for (int l=0; l<_n_pyramids; ++l) {
        cudaMemcpy2DToArrayAsync (gpu_array_pict_0,
                                  pyr_offset * sizeof(float),
                                  0,
                                  gpu_img_pyramid_prev1[l],
                                  sizeof(float)*pyr_w[l],
                                  sizeof(float)*pyr_w[l],
                                  pyr_h[l],
                                  cudaMemcpyDeviceToDevice);

        cudaMemcpy2DToArrayAsync(gpu_array_pict_1,
                                 pyr_offset * sizeof(float),
                                 0,
                                 gpu_img_pyramid_prev2[l],
                                 sizeof(float)*pyr_w[l],
                                 sizeof(float)*pyr_w[l],
                                 pyr_h[l],
                                 cudaMemcpyDeviceToDevice);

        pyr_offset += pyr_w[l];
    }
    checkCUDAError("Fill in pict buffers");

    // Fill in derivatives, for the two pictures :
    fillDerivatives(gpu_img_pyramid_prev1,
                    gpu_array_deriv_x_0,
                    gpu_array_deriv_y_0);

    fillDerivatives(gpu_img_pyramid_prev2,
                    gpu_array_deriv_x_1,
                    gpu_array_deriv_y_1);

    checkCUDAError("Computing derivatives");

    cudaMemset(gpu_status, 0, sizeof(char) * MAX_POINTS); // Not ready to track
    printf("CUDA : back pictures loaded %d x %d \n", w, h);
}


// Load current pair of pictures
// Called every time
void cudaLK::loadCurPictures(const IplImage *cur1,
                             const IplImage *cur2,
                             bool b_CvtToGrey) {

    if (!this->b_mem4_allocated) {
        printf("CUDA : error - memory must be allocated before use\n");
        return;
    } else if ( (cur1->width != w) || (cur1->height !=h) ) {
        printf("CUDA : error - pictures must have the same size\n");
        return;
    }

    int blocksW = w/_n_threads_x + ((w % _n_threads_x)?1:0);
    int blocksH = h/_n_threads_y + ((h % _n_threads_y)?1:0);
    dim3 blocks(blocksW, blocksH);
    dim3 threads(_n_threads_x, _n_threads_y);

    // Transfer from host memspace to gpu memspace
    if (b_CvtToGrey) {
        cudaMemcpy2D (gpu_img_cur1_RGB, w*sizeof(uchar), cur1->imageData, cur1->widthStep, 3 * cur1->width * sizeof(uchar), cur1->height, cudaMemcpyHostToDevice );
        cudaMemcpy2D (gpu_img_cur2_RGB, w*sizeof(uchar), cur2->imageData, cur2->widthStep, 3 * cur2->width * sizeof(uchar), cur2->height, cudaMemcpyHostToDevice );
    } else {
        cudaMemcpy2D (gpu_img_cur1_RGB, w*sizeof(uchar), cur1->imageData, cur1->widthStep, cur1->width * sizeof(uchar), cur1->height, cudaMemcpyHostToDevice );
        cudaMemcpy2D (gpu_img_cur2_RGB, w*sizeof(uchar), cur2->imageData, cur2->widthStep, cur2->width * sizeof(uchar), cur2->height, cudaMemcpyHostToDevice );
    }

    checkCUDAError("pictCopyToGPU");

    // Convert picture to floats & grey
    cvtPicture(true, b_CvtToGrey);
    checkCUDAError("pictConversion");

    // Build pyramids
    for(int i=0; i < _n_pyramids-1; i++) {
        kernelSmoothX<<<blocks, threads>>>(gpu_img_pyramid_cur1[i], pyr_w[i], pyr_h[i], gpu_smoothed_cur1_x);
        kernelSmoothX<<<blocks, threads>>>(gpu_img_pyramid_cur2[i], pyr_w[i], pyr_h[i], gpu_smoothed_cur2_x);

        kernelSmoothY<<<blocks, threads>>>(gpu_smoothed_cur1_x, pyr_w[i], pyr_h[i], gpu_smoothed_cur1);
        kernelSmoothY<<<blocks, threads>>>(gpu_smoothed_cur2_x, pyr_w[i], pyr_h[i], gpu_smoothed_cur2);

        pyrDownsample<<<blocks, threads>>>(gpu_smoothed_cur1, pyr_w[i], pyr_h[i], gpu_img_pyramid_cur1[i+1], pyr_w[i+1], pyr_h[i+1]);
        pyrDownsample<<<blocks, threads>>>(gpu_smoothed_cur2, pyr_w[i], pyr_h[i], gpu_img_pyramid_cur2[i+1], pyr_w[i+1], pyr_h[i+1]);
    }

    // Load cudaArray buffer from pyramids
    int pyr_offset = 0;
    for (int l=0; l<_n_pyramids; ++l) {
        cudaMemcpy2DToArrayAsync (gpu_array_pict_2,
                                  pyr_offset * sizeof(float),
                                  0,
                                  gpu_img_pyramid_cur2[l],
                                  sizeof(float)*pyr_w[l],
                                  sizeof(float)*pyr_w[l],
                                  pyr_h[l],
                                  cudaMemcpyDeviceToDevice);

        cudaMemcpy2DToArrayAsync(gpu_array_pict_3,
                                 pyr_offset * sizeof(float),
                                 0,
                                 gpu_img_pyramid_cur1[l],
                                 sizeof(float)*pyr_w[l],
                                 sizeof(float)*pyr_w[l],
                                 pyr_h[l],
                                 cudaMemcpyDeviceToDevice);

        pyr_offset += pyr_w[l];
    }

    // Fill in derivatives, for the two pictures :
    fillDerivatives(gpu_img_pyramid_cur1,
                    gpu_array_deriv_x_3,
                    gpu_array_deriv_y_3);

    fillDerivatives(gpu_img_pyramid_cur2,
                    gpu_array_deriv_x_2,
                    gpu_array_deriv_y_2);

    cudaMemset(gpu_status, 1, sizeof(char) * MAX_POINTS); // Ready to track

    checkCUDAError("Pyramid building");
}

void cudaLK::processTracking( int nPoints )
{


    // 2D-indexing for kernels
    int n_pts_ceil = MIN(nPoints, MAX_POINTS);
    int n_pts_sq = (int) round( sqrt(n_pts_ceil)) + 1;

    int blocksW = n_pts_sq/_n_threads_x +
                  ((n_pts_sq % _n_threads_x)?1:0);

    int blocksH = n_pts_sq/_n_threads_y +
                  ((n_pts_sq % _n_threads_y )?1:0);

    dim3 blocks(blocksW, blocksH);
    dim3 threads(_n_threads_x, _n_threads_y);

    for( int l = _n_pyramids-1; l >= 0; l-- )
    {
        // Set constant parameters
        setSymbols(l);

        // Compute gradient descent parameters
        compute_spatial_grad <<<blocks, threads>>>(gpu_pt_indexes,
                                                   gpu_status,
                                                   gpu_neighbourhood_det,
                                                   gpu_neighbourhood_Iyy,
                                                   gpu_neighbourhood_Ixy,
                                                   gpu_neighbourhood_Ixx);

        // Compute the new position of the points
        track_pts_slim<<<blocks, threads>>>(gpu_pt_indexes,
                                            gpu_dx,
                                            gpu_dy,
                                            gpu_status,
                                            gpu_neighbourhood_det,
                                            gpu_neighbourhood_Iyy,
                                            gpu_neighbourhood_Ixy,
                                            gpu_neighbourhood_Ixx);
    }
}

void cudaLK::resetDisplacements() {
    cudaMemset(gpu_dx,0, sizeof(float) * MAX_POINTS);
    cudaMemset(gpu_dy,0, sizeof(float) * MAX_POINTS);
}

void cudaLK::releaseMem() {
    // Redundant tracking allocated
    if (this->b_mem4_allocated ) {
        printf("CudaLK : Releasing 4-Frames buffers\n");

        // Release pyramids
        for(int i=1; i < _n_pyramids; i++) {
            cudaFree(gpu_img_pyramid_prev1[i]);
            cudaFree(gpu_img_pyramid_prev2[i]);
            cudaFree(gpu_img_pyramid_cur1[i]);
            cudaFree(gpu_img_pyramid_cur2[i]);
        }

        // Release picture buffers
        cudaFree(gpu_img_prev1_RGB);
        cudaFree(gpu_img_prev2_RGB);
        cudaFree(gpu_img_cur1_RGB);
        cudaFree(gpu_img_cur2_RGB);

        // Release Pyramids
        cudaFree(gpu_img_pyramid_prev1[0]);
        cudaFree(gpu_img_pyramid_prev2[0]);
        cudaFree(gpu_img_pyramid_cur1[0]);
        cudaFree(gpu_img_pyramid_cur2[0]);

        // Release smoothed pictures (for pyramids)
        cudaFree(gpu_smoothed_prev1_x);
        cudaFree(gpu_smoothed_prev2_x);
        cudaFree(gpu_smoothed_cur1_x);
        cudaFree(gpu_smoothed_cur2_x);

        cudaFree(gpu_smoothed_prev1);
        cudaFree(gpu_smoothed_prev2);
        cudaFree(gpu_smoothed_cur1);
        cudaFree(gpu_smoothed_cur2);


        // Release spatial derivatives
        cudaFree(gpu_deriv_x);
        cudaFree(gpu_deriv_y);

        // Release compute intermediates
        cudaFree(gpu_neighbourhood_det);
        cudaFree(gpu_neighbourhood_Iyy);
        cudaFree(gpu_neighbourhood_Ixy);
        cudaFree(gpu_neighbourhood_Ixx);

        // Indexes
        cudaFree ((void**) &gpu_pt_indexes);

        // Unbind textures
        cudaUnbindTexture (gpu_textr_pict_0);
        cudaUnbindTexture (gpu_textr_pict_1);

        cudaUnbindTexture (gpu_textr_deriv_x);
        cudaUnbindTexture (gpu_textr_deriv_y);

        // Release Arrays behind textures
        cudaFreeArray (gpu_array_pict_0);
        cudaFreeArray (gpu_array_pict_1);
        cudaFreeArray (gpu_array_pict_2);
        cudaFreeArray (gpu_array_pict_3);

        cudaFreeArray (gpu_array_deriv_x_0);
        cudaFreeArray (gpu_array_deriv_y_0);
        cudaFreeArray (gpu_array_deriv_x_1);
        cudaFreeArray (gpu_array_deriv_y_1);
        cudaFreeArray (gpu_array_deriv_x_2);
        cudaFreeArray (gpu_array_deriv_y_2);
        cudaFreeArray (gpu_array_deriv_x_3);
        cudaFreeArray (gpu_array_deriv_y_3);

        // Release Displacements
        cudaFree(gpu_dx);
        cudaFree(gpu_dy);

        cudaFree(gpu_dx1);
        cudaFree(gpu_dy1);

        cudaFree(gpu_dx2);
        cudaFree(gpu_dy2);

        cudaFree(gpu_dx3);
        cudaFree(gpu_dy3);

        cudaFreeHost(dx1);
        cudaFreeHost(dy1);
        cudaFreeHost(dx2);
        cudaFreeHost(dy2);
        cudaFreeHost(dx3);
        cudaFreeHost(dy3);
        cudaFreeHost(dx4);
        cudaFreeHost(dy4);
        cudaFreeHost(status);

        // Check GPU status
        cudaFree(gpu_status);

        printf("CudaLK : buffers released\n");
    }

    // Simple tracking allocated
    if (this->b_mem_allocated) {
        // Free arrays
        for(int i=0; i < _n_pyramids; i++) {
            cudaFree(gpu_img_pyramid_prev1[i]);
            cudaFree(gpu_img_pyramid_cur1[i]);
        }

        cudaFree(gpu_smoothed_prev_x);
        cudaFree(gpu_smoothed_cur_x);
        cudaFree(gpu_smoothed_prev);
        cudaFree(gpu_smoothed_cur);
        cudaFree(gpu_pt_indexes);

        cudaFreeHost(dx1);
        cudaFreeHost(dy1);

        cudaFree(gpu_dx);
        cudaFree(gpu_dy);
        cudaFree(gpu_status);

        // Free textures
        cudaFreeArray(gpu_array_pyramid_prev);
        cudaFreeArray(gpu_array_pyramid_prev_Ix);
        cudaFreeArray(gpu_array_pyramid_prev_Iy);
        cudaFreeArray(gpu_array_pyramid_cur);

        delete [] status;
    }
}

// Coherent sparse tracking on stereo pair
//!\\ Previous set of pictures must be loaded prior to using this function
void cudaLK::run4Frames( IplImage  *cur1,
                         IplImage  *cur2,
                         float     *pt_to_track,
                         int       nPoints,
                         bool      cvtToGrey )
{
    int const & width  = cur1->width;
    int const & height = cur1->height;
    int const n_pts_ceil = MIN(nPoints, MAX_POINTS);

    // Check memory allocation before proceeding
    if (!b_mem4_allocated)
    {
        fprintf(stderr, "run4Frames : error - memory must be allocated and \n .. initial pictures loaded\n");
        exit(EXIT_FAILURE);
    }
    else if ((width != w) || (height != h))
    {
        fprintf(stderr, "run4Frames : error - Pictures must have the same size\n");
        exit(EXIT_FAILURE);
    }

    int win_size_full = _patch_radius;
    int win_size_short = 2;

    // Load current pictures & build pyramids
    loadCurPictures(cur1, cur2, cvtToGrey);

    // Load the coordinates of the points to track & define some settings
    cudaMemcpy(gpu_pt_indexes, pt_to_track, 2 * n_pts_ceil * sizeof(float), cudaMemcpyHostToDevice);

    checkCUDAError ("Loading pictures");

    // -----------------------------------------------------
    // "Loop" tracking
    // -----------------------------------------------------

    // -----------------------------------------------------
    // --- Step 1 -----

    // Bind textures and arrays...
    cudaMemcpyToSymbol (LK_win_size, &win_size_short, sizeof(int)); // win_size_short
    bindTextureUnits(gpu_array_pict_0,
                     gpu_array_pict_3,
                     gpu_array_deriv_x_0,
                     gpu_array_deriv_y_0);

    // Process
    resetDisplacements();
    processTracking(nPoints);
    checkCUDAError ("First step");

    // Copy back results
    cudaMemcpy(gpu_dx1, gpu_dx, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToDevice); // Handle "status" array
    cudaMemcpy(gpu_dy1, gpu_dy, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToDevice);

    cudaThreadSynchronize();

    cudaMemcpyAsync(dx1, gpu_dx1, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToHost); // Non-blocking memcpy
    cudaMemcpyAsync(dy1, gpu_dy1, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToHost);

    // -----------------------------------------------------
    // --- Step 2 -----

    // Change texture binding
    cudaMemcpyToSymbol (LK_win_size, &win_size_full, sizeof(int));
    bindTextureUnits(gpu_array_pict_3,
                     gpu_array_pict_2,
                     gpu_array_deriv_x_3,
                     gpu_array_deriv_y_3);

    // Process
    resetDisplacements();
    processTracking(nPoints);
    checkCUDAError ("Second step");

    // Copy back results
    cudaMemcpy(gpu_dx2, gpu_dx, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToDevice); // Handle "status" array
    cudaMemcpy(gpu_dy2, gpu_dy, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToDevice);

    cudaThreadSynchronize();

    cudaMemcpyAsync(dx2, gpu_dx2, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToHost); // Non-blocking memcpy
    cudaMemcpyAsync(dy2, gpu_dy2, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToHost);


    // -----------------------------------------------------
    // --- Step 3 -----
    cudaMemcpyToSymbol (LK_win_size, &win_size_short, sizeof(int)); // win_size_short

    // Change texture binding
    bindTextureUnits(gpu_array_pict_2,
                     gpu_array_pict_1,
                     gpu_array_deriv_x_2,
                     gpu_array_deriv_y_2);

    // Process
    resetDisplacements();
    processTracking(nPoints);
    checkCUDAError ("Third step");

    // Copy back results
    cudaMemcpy(gpu_dx3, gpu_dx, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_dy3, gpu_dy, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToDevice);

    cudaThreadSynchronize();

    cudaMemcpyAsync(dx3, gpu_dx3, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToHost,0);
    cudaMemcpyAsync(dy3, gpu_dy3, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToHost,0);


    // -----------------------------------------------------
    // --- Step 4 -----

    cudaMemcpyToSymbol (LK_win_size, &win_size_full, sizeof(int));

    // Change texture binding
    bindTextureUnits(gpu_array_pict_1,
                     gpu_array_pict_0,
                     gpu_array_deriv_x_1,
                     gpu_array_deriv_y_1);

    // Process
    resetDisplacements();
    processTracking(nPoints);

    cudaThreadSynchronize();
    checkCUDAError ("Last step");

    // Copy back results to Host (non blocking memcpy to pinned memory)
    cudaMemcpyAsync(dx4, gpu_dx, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(dy4, gpu_dy, sizeof(float) * n_pts_ceil, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(status, gpu_status, sizeof(char) * n_pts_ceil, cudaMemcpyDeviceToHost);

    // Cpy new point position :
    cudaMemcpy(pt_to_track, gpu_pt_indexes, sizeof(float) * 2 * n_pts_ceil, cudaMemcpyDeviceToHost);

    // Swap the pictures & pyramids (current -> back)
    swapPyramids();
}


void cudaLK::setSymbols(int pyr_level) {
    int pyr_deriv_offset = 0;
    char initGuess;

    for (int i=0; i<pyr_level; ++i) {
        pyr_deriv_offset += pyr_w[i];
    }
    cudaMemcpyToSymbol (LK_width_offset, &pyr_deriv_offset, sizeof(int));

    initGuess = (pyr_level == _n_pyramids-1);
    cudaMemcpyToSymbol (LK_pyr_w,     &pyr_w[pyr_level],    sizeof(int));
    cudaMemcpyToSymbol (LK_pyr_h,     &pyr_h[pyr_level],    sizeof(int));
    cudaMemcpyToSymbol (LK_pyr_level, &pyr_level,           sizeof(int));
    cudaMemcpyToSymbol (LK_scaling,   &scaling[pyr_level],  sizeof(float));
    cudaMemcpyToSymbol (LK_init_guess,&initGuess,           sizeof(char));
}


void cudaLK::setupTextures() {
    // Picture buffers
    gpu_textr_pict_0.normalized = 0;
    gpu_textr_pict_0.filterMode = cudaFilterModeLinear;
    gpu_textr_pict_0.addressMode[0] = cudaAddressModeClamp;  // Handle request outside boundaries
    gpu_textr_pict_0.addressMode[1] = cudaAddressModeClamp;

    gpu_textr_pict_1.normalized = 0;
    gpu_textr_pict_1.filterMode = cudaFilterModeLinear;
    gpu_textr_pict_1.addressMode[0] = cudaAddressModeClamp;  // Handle request outside boundaries
    gpu_textr_pict_1.addressMode[1] = cudaAddressModeClamp;

    // Spatial derivatives
    gpu_textr_deriv_x.normalized = 0;
    gpu_textr_deriv_x.filterMode = cudaFilterModeLinear;
    gpu_textr_deriv_x.addressMode[0] = cudaAddressModeClamp;  // Handle request outside boundaries
    gpu_textr_deriv_x.addressMode[1] = cudaAddressModeClamp;

    gpu_textr_deriv_y.normalized = 0;
    gpu_textr_deriv_y.filterMode = cudaFilterModeLinear;
    gpu_textr_deriv_y.addressMode[0] = cudaAddressModeClamp;  // Handle request outside boundaries
    gpu_textr_deriv_y.addressMode[1] = cudaAddressModeClamp;

    checkCUDAError ("Initializing textures");
}

void cudaLK::sobelFiltering(const float *pict_in,
                            const int w,
                            const int h,
                            float *pict_out) {
    // TODO

    // 2D-indexing for kernel managements
    int _n_threads_x = NTHREAD_X;
    int _n_threads_y = NTHREAD_Y;

    int blocksW = w/_n_threads_x + ((w % _n_threads_x)?1:0);
    int blocksH = h/_n_threads_y + ((h % _n_threads_y)?1:0);

    dim3 blocks(blocksW, blocksH);
    dim3 threads(_n_threads_x, _n_threads_y);

    // Compute via kernels
    kernelSobelX <<<blocks, threads>>>(pict_in, w, h, buff1);

    kernelSobelY <<<blocks, threads>>>(pict_in, w, h, buff2);

    // Mix sobel gradient into one picture
    kernelAdd <<<blocks, threads>>>(buff1, buff2, w, h, pict_out);

    cudaThreadSynchronize();
}

void cudaLK::sobelFilteringX(const float *pict_in,
                             const int w,
                             const int h,
                             float *pict_out) {
    // TODO

    // 2D-indexing for kernel managements
    int _n_threads_x = NTHREAD_X;
    int _n_threads_y = NTHREAD_Y;

    int blocksW = w/_n_threads_x + ((w % _n_threads_x)?1:0);
    int blocksH = h/_n_threads_y + ((h % _n_threads_y)?1:0);

    dim3 blocks(blocksW, blocksH);
    dim3 threads(_n_threads_x, _n_threads_y);

    // Compute via kernels
    kernelSobelX <<<blocks, threads>>>(pict_in, w, h, pict_out);
    cudaThreadSynchronize();
}

void cudaLK::sobelFilteringY(const float *pict_in,
                             const int w,
                             const int h,
                             float *pict_out) {
    // TODO

    // 2D-indexing for kernel managements
    int _n_threads_x = NTHREAD_X;
    int _n_threads_y = NTHREAD_Y;

    int blocksW = w/_n_threads_x + ((w % _n_threads_x)?1:0);
    int blocksH = h/_n_threads_y + ((h % _n_threads_y)?1:0);

    dim3 blocks(blocksW, blocksH);
    dim3 threads(_n_threads_x, _n_threads_y);

    // Compute via kernels
    kernelSobelY <<<blocks, threads>>>(pict_in, w, h, pict_out);
    cudaThreadSynchronize();
}

// Swap current/backbuffer pyramids
void cudaLK::swapPyramids () {
    // Swap pyramid arrays:
    cudaMemcpy2DArrayToArray  (gpu_array_pict_0,
                               0, 0,
                               gpu_array_pict_3,
                               0, 0,
                               2 * w * sizeof(float),
                               h,
                               cudaMemcpyDeviceToDevice);

    cudaMemcpy2DArrayToArray  (gpu_array_pict_1,
                               0, 0,
                               gpu_array_pict_2,
                               0, 0,
                               2 * w * sizeof(float),
                               h,
                               cudaMemcpyDeviceToDevice);


    // Swap derivatives pyramid :
    cudaMemcpy2DArrayToArray (gpu_array_deriv_x_0,
                              0, 0,
                              gpu_array_deriv_x_3,
                              0, 0,
                              2 * w * sizeof(float),
                              h,
                              cudaMemcpyDeviceToDevice);

    cudaMemcpy2DArrayToArray (gpu_array_deriv_y_0,
                              0, 0,
                              gpu_array_deriv_y_3,
                              0, 0,
                              2 * w * sizeof(float),
                              h,
                              cudaMemcpyDeviceToDevice);

    cudaMemcpy2DArrayToArray (gpu_array_deriv_x_1,
                              0, 0,
                              gpu_array_deriv_x_2,
                              0, 0,
                              2 * w * sizeof(float),
                              h,
                              cudaMemcpyDeviceToDevice);

    cudaMemcpy2DArrayToArray (gpu_array_deriv_y_1,
                              0, 0,
                              gpu_array_deriv_y_2,
                              0, 0,
                              2 * w * sizeof(float),
                              h,
                              cudaMemcpyDeviceToDevice);
}
/********************************
 * Previous work....
 *********************************/

/*
 *void cudaLK::runTracking(unsigned char *prev,     // Previous picture
                         unsigned char *cur,      // New picture
                         int _w,                  // Picture width
                         int _h,                  // Picture height
                         float *pt_to_track,      // 2D array of indexes : points to track (floats !)
                         int n_pts,               // Number of points to track
                         bool b_CvtToGrey)        // Do the RGB2GRAY conversion or not (single channel picture)
{
  //
  // Sparse optical field calculus : follow points specified in "pt_to_track" array
  //

  if (! b_mem_allocated) {
    w = _w;
    h = _h;
    initMem();
  }

  int _n_threads_x = NTHREAD_X;
  int _n_threads_y = NTHREAD_Y;

  int blocksW = w/_n_threads_x + ((w % _n_threads_x)?1:0);
  int blocksH = h/_n_threads_y + ((h % _n_threads_y )?1:0);
  dim3 blocks(blocksW, blocksH);
  dim3 threads(_n_threads_x, _n_threads_y);
  int blocks1D = (w*h)/256 + (w*h % 256?1:0); // for greyscale
  int blocks1D_tracking = n_pts/256 + (n_pts % 256 ? 1:0);

  // Copy image to GPU :
  if(b_CvtToGrey) {
    cudaMemcpy(gpu_img_prev_RGB, prev, w*h*3, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_img_cur_RGB, cur, w*h*3, cudaMemcpyHostToDevice);
    checkCUDAError("start");
  } else {
    cudaMemcpy(gpu_img_prev_RGB, prev, w*h, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_img_cur_RGB, cur, w*h, cudaMemcpyHostToDevice);
    checkCUDAError("start");
  }

  // Copy indexes to follow to GPU :
  cudaMemcpy(gpu_pt_indexes, pt_to_track, 2*n_pts*sizeof(float), cudaMemcpyHostToDevice);
  checkCUDAError ("Copying indexes to follow");


  // Convert pictures (float & grey)
  if (b_CvtToGrey) {
    // RGB -> grey
    convertRGBToGrey<<<blocks1D, 256>>>(gpu_img_prev_RGB, gpu_img_pyramid_prev1[0], w*h);
    convertRGBToGrey<<<blocks1D, 256>>>(gpu_img_cur_RGB, gpu_img_pyramid_cur1[0], w*h);
    cudaThreadSynchronize();
    checkCUDAError("convertRGBToGrey");
  } else {
    // Simply convert char to float in kernel
    convertGreyToFloat<<<blocks1D, 256>>>(gpu_img_prev_RGB, gpu_img_pyramid_prev1[0], w*h);
    convertGreyToFloat<<<blocks1D, 256>>>(gpu_img_cur_RGB, gpu_img_pyramid_cur1[0], w*h);
    cudaThreadSynchronize();
    checkCUDAError("convertToFloat");
  }

  // Build pyramids
  for(int i=0; i < _n_pyramids-1; i++) {
    kernelSmoothX<<<blocks, threads>>>(gpu_img_pyramid_prev1[i], pyr_w[i], pyr_h[i], gpu_smoothed_prev_x);
    kernelSmoothX<<<blocks, threads>>>(gpu_img_pyramid_cur1[i], pyr_w[i], pyr_h[i], gpu_smoothed_cur_x);
    cudaThreadSynchronize();
    kernelSmoothY<<<blocks, threads>>>(gpu_smoothed_prev_x, pyr_w[i], pyr_h[i], gpu_smoothed_prev);
    kernelSmoothY<<<blocks, threads>>>(gpu_smoothed_cur_x, pyr_w[i], pyr_h[i], gpu_smoothed_cur);
    cudaThreadSynchronize();

    pyrDownsample<<<blocks, threads>>>(gpu_smoothed_prev, pyr_w[i], pyr_h[i], gpu_img_pyramid_prev1[i+1], pyr_w[i+1], pyr_h[i+1]);
    pyrDownsample<<<blocks, threads>>>(gpu_smoothed_cur,  pyr_w[i], pyr_h[i], gpu_img_pyramid_cur1[i+1],  pyr_w[i+1], pyr_h[i+1]);
    cudaThreadSynchronize();

    checkCUDAError("pyrDownsample here");
  }

  cudaMemset(gpu_status, 1, sizeof(char) * MAX_POINTS);

  // Do the actual tracking
  for(int l=_n_pyramids-1; l >= 0; l--) {

    // Copy arrays to texture
    cudaMemcpy2DToArray(gpu_array_pyramid_prev, 0, 0, gpu_img_pyramid_prev1[l],
                        sizeof(float)*pyr_w[l], sizeof(float)*pyr_w[l], pyr_h[l], cudaMemcpyDeviceToDevice);

    cudaMemcpy2DToArray(gpu_array_pyramid_cur, 0, 0, gpu_img_pyramid_cur1[l],
                        sizeof(float)*pyr_w[l], sizeof(float)*pyr_w[l], pyr_h[l], cudaMemcpyDeviceToDevice);

    // Track
    //    trackPt<<<blocks1D_tracking, 256>>>(gpu_pt_indexes, n_pts, w, h, pyr_w[l], pyr_h[l], scaling[l], l, (l == levels-1), gpu_dx, gpu_dy, 10, .3f, gpu_status);
    trackPt<<<blocks1D_tracking, 256>>>(gpu_pt_indexes, n_pts, w, h, pyr_w[l], pyr_w[l], scaling[l], l, (l == _n_pyramids-1), gpu_dx, gpu_dy, 10, .3f, gpu_status);

    cudaThreadSynchronize();
  }

  // Copy back results
  cudaMemcpy(dx1, gpu_dx, sizeof(float)*n_pts, cudaMemcpyDeviceToHost);
  cudaMemcpy(dy1, gpu_dy, sizeof(float)*n_pts, cudaMemcpyDeviceToHost);
  cudaMemcpy(status, gpu_status, sizeof(char)*n_pts, cudaMemcpyDeviceToHost);
}
*/
