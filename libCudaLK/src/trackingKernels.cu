
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>

// Constant values on device
// /!\ undefined in host code, just in kernels /!\ __device__
#define MAX_WEIGHT_VALUES 50
#define MIN_DET FLT_EPSILON

__constant__ __device__ int   LK_iteration;
__constant__ __device__ int   LK_patch;
__constant__ __device__ int   LK_points;
__constant__ __device__ int   LK_height;
__constant__ __device__ int   LK_width;
__constant__ __device__ int   LK_pyr_w;
__constant__ __device__ int   LK_pyr_h;
__constant__ __device__ int   LK_pyr_level;
__constant__ __device__ int   LK_width_offset;
__constant__ __device__ char  LK_init_guess;
__constant__ __device__ float LK_scaling;
__constant__ __device__ float LK_threshold;
__constant__ __device__ float LK_Weight[MAX_WEIGHT_VALUES];
__constant__ __device__ int   LK_win_size;

// Texture buffer is used for each image for on-the-fly interpolation
texture <float, 2, cudaReadModeElementType> texRef_pyramid_prev;
texture <float, 2, cudaReadModeElementType> texRef_pyramid_cur;

// Image pyramids -> texture buffers
texture <float, 2, cudaReadModeElementType> gpu_textr_pict_0;   // pictures > texture space
texture <float, 2, cudaReadModeElementType> gpu_textr_pict_1;

texture <float, 2, cudaReadModeElementType> gpu_textr_deriv_x;  // gradients > texture space
texture <float, 2, cudaReadModeElementType> gpu_textr_deriv_y;

// Convert RGB Picture to grey/float
__global__ void convertRGBToGrey(unsigned char *d_in, float *d_out, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < N)
    {
        d_out[idx] = d_in[idx*3]*0.1144f
                     + d_in[idx*3+1]*0.5867f
                     + d_in[idx*3+2]*0.2989f;
    }
}

// Convert Grey uchar picture to float
__global__ void convertGreyToFloat(unsigned char const * const d_in, float *d_out, int const N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < N)
        d_out[idx] = __fdividef((float) d_in[idx], 254.0);
}

// Downsample picture to build pyramid lower level (naive implementation..)
__global__ void pyrDownsample(float const * const in, int w1, int h1, float * const out, int w2, int h2)
{
    // Input has to be greyscale
    int x2 = blockIdx.x*blockDim.x + threadIdx.x;
    int y2 = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x2 < w2) && (y2 < h2) ) {
        int x = x2*2;
        int y = y2*2;
        int x_1 = x-1;
        int y_1 = y-1;
        int x_2 = x+1;
        int y_2 = y+1;

        // Pad values
        if(x_1 < 0) x_1 = 0;
        if(y_1 < 0) y_1 = 0;
        if(x_2 >= w1) x_2 = w1 - 1;
        if(y_2 >= h1) y_2 = h1 - 1;

        //     Initial extrapolation pattern 1/4 1/8 1/16
        out[y2*w2 + x2] = 0.25f*in[y*w1+x] + 0.125f*(in[y*w1+x_1] + in[y*w1+x_2] + in[y_1*w1+x] + in[y_2*w1+x]) +
                0.0625f*(in[y_1*w1+x_1] + in[y_2*w1+x_1] + in[y_1*w1+x_2] + in[y_2*w1+x_2]);
        {
            // OpenCV extrapolation pattern :
            //    out[y2*w2 + x2] = 0.375f*in[y*w1+x] + 0.25f*(in[y*w1+x_1] + in[y*w1+x_2] + in[y_1*w1+x] + in[y_2*w1+x]) +
            //        0.0625f*(in[y_1*w1+x_1] + in[y_2*w1+x_1] + in[y_1*w1+x_2] + in[y_2*w1+x_2]);

            //    // Another trial to improve interpolation pattern..
            //    out[y2*w2 + x2] = 0.23077f*in[y*w1+x] + 0.15385f*(in[y*w1+x_1] + in[y*w1+x_2] + in[y_1*w1+x] + in[y_2*w1+x]) +
            //        0.03846f*(in[y_1*w1+x_1] + in[y_2*w1+x_1] + in[y_1*w1+x_2] + in[y_2*w1+x_2]);
        }

    }
}

__global__ void compute_spatial_grad( float const * const coord_gpu,
                                      char   *status_gpu,
                                      float  *gpu_neighbourhood_det,
                                      float  *gpu_neighbourhood_Iyy,
                                      float  *gpu_neighbourhood_Ixy,
                                      float  *gpu_neighbourhood_Ixx )
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x)
              + (blockIdx.y*blockDim.y + threadIdx.y) * gridDim.x * blockDim.x; // "2D" indexing

    if (idx >= LK_points)
        return;

    float x_pt = coord_gpu[2*idx];
    float y_pt = coord_gpu[2*idx+1];

    if(x_pt > (LK_width-1) ||
            y_pt > (LK_height-1)) // Useful check ?
        return;

    if(status_gpu[idx] == 0)
        return;

    x_pt *= LK_scaling;
    y_pt *= LK_scaling;

    int xx, yy;
    float Ix, Iy, sum_Ixy, sum_Iyy, sum_Ixx;

    // TODO : offset the coordinate to access derivatives array

    for(yy=-LK_patch; yy <= LK_patch; ++yy) {
        for(xx=-LK_win_size; xx <= LK_win_size; ++xx) {

            Ix = tex2D(gpu_textr_deriv_x, x_pt + LK_width_offset + xx, y_pt + yy);
            Iy = tex2D(gpu_textr_deriv_y, x_pt + LK_width_offset + xx, y_pt + yy);

            sum_Ixx += Ix * Ix;
            sum_Ixy += Ix * Iy;
            sum_Iyy += Iy * Iy;
        }
    }

    gpu_neighbourhood_det[idx] = sum_Ixx*sum_Iyy - sum_Ixy*sum_Ixy;
    gpu_neighbourhood_Iyy[idx] = sum_Iyy;
    gpu_neighbourhood_Ixy[idx] = sum_Ixy;
    gpu_neighbourhood_Ixx[idx] = sum_Ixx;

    // Deal with case : could not track (no gradient)
    if(gpu_neighbourhood_det[idx] < MIN_DET)
    {
        status_gpu[idx] = 0;
        return;
    }
}

// Kernel to compute the tracking
__global__ void track_pts_slim(float * const coord_gpu,
                               float *dx_gpu,
                               float *dy_gpu,
                               char  *status_gpu,
                               const float * const gpu_neighbourhood_det,
                               const float * const gpu_neighbourhood_Iyy,
                               const float * const gpu_neighbourhood_Ixy,
                               const float * const gpu_neighbourhood_Ixx)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x)
              + (blockIdx.y*blockDim.y + threadIdx.y) * gridDim.x * blockDim.x; // "2D" indexing

    if (idx >= LK_points)
        return;

    float x_pt = coord_gpu[2*idx];
    float y_pt = coord_gpu[2*idx+1];

    if(x_pt > (LK_width-1) || y_pt > (LK_height-1))
        return;

    if(status_gpu[idx] == 0)
        return;

    float Vx, Vy;            // Previous speed for this point
    float cur_x, cur_y;      // Current position
    float sum_Ixt, sum_Iyt;
    float Ix, Iy, It;
    int j;                   // Research window indexes
    float xx, yy;
    float vx, vy;

    x_pt *= LK_scaling;
    y_pt *= LK_scaling;

    if( LK_init_guess )
    {
        Vx = 0.f;
        Vy = 0.f;
        cur_x = x_pt;
        cur_y = y_pt;
    }
    else {
        Vx = dx_gpu[idx];
        Vy = dy_gpu[idx];
        cur_x = x_pt + Vx;
        cur_y = y_pt + Vy;
    }

    // Iteration part
    for(j=0; j < LK_iteration; ++j)
    {
        // If current speed vector drives the point out of bounds
        if( cur_x < 0.f || cur_x > LK_pyr_w
                || cur_y < 0.f || cur_y > LK_pyr_h)
        {
            dx_gpu[idx] = 0.f;
            dy_gpu[idx] = 0.f;
            status_gpu[idx] = 0;

            return;
        }

        sum_Ixt = 0.f;
        sum_Iyt = 0.f;

        // No explicit handling of pixels outside the image
        // Texture fetchs ensure calls are clamped to window size
        for( yy=-LK_patch; yy <= LK_patch; ++yy )
        {
            for( xx=-LK_win_size; xx <= LK_win_size; ++xx )
            {
                It = tex2D(gpu_textr_pict_1, cur_x + LK_width_offset + xx, cur_y + yy)
                     - tex2D(gpu_textr_pict_0, x_pt + LK_width_offset + xx, y_pt + yy);

                Ix = tex2D(gpu_textr_deriv_x, x_pt + LK_width_offset + xx, y_pt + yy);
                Iy = tex2D(gpu_textr_deriv_y, x_pt + LK_width_offset + xx, y_pt + yy);

                sum_Ixt += Ix*It;
                sum_Iyt += Iy*It;
            }
        }

        // Find the inverse of the 2x2 matrix using a mix of determinant and adjugate matrix
        // http://cnx.org/content/m19446/latest/
        vx = __fdividef((- gpu_neighbourhood_Iyy[idx] * sum_Ixt +
                         gpu_neighbourhood_Ixy[idx] * sum_Iyt), gpu_neighbourhood_det[idx]);

        vy = __fdividef(( gpu_neighbourhood_Ixy[idx] * sum_Ixt -
                          gpu_neighbourhood_Ixx[idx] * sum_Iyt), gpu_neighbourhood_det[idx]);

        Vx += vx;
        Vy += vy;
        cur_x += vx;
        cur_y += vy;

        // Stop if movement is very small
        if( fabsf(vx) < LK_threshold && fabsf(vy) < LK_threshold )
            break;
    }

    // Double speed vector to get to next scale
    if(LK_pyr_level != 0) {
        Vx += Vx;
        Vy += Vy;
    }

    dx_gpu[idx] = Vx;
    dy_gpu[idx] = Vy;

    // Shift coordinates of the points to track
    if (LK_pyr_level == 0)
    {
        coord_gpu[2*idx  ] -= Vx;
        coord_gpu[2*idx+1] -= Vy;
    }
}

// Kernel to compute the tracking
__global__ void track_pts_weighted(float * const coord_gpu,
                                   float * const dx_gpu,
                                   float * const dy_gpu,
                                   char  * const status_gpu,
                                   bool  rectified)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x)
              + (blockIdx.y*blockDim.y + threadIdx.y) * gridDim.x * blockDim.x; // "2D" indexing

    if (idx >= LK_points)
        return;

    float x_pt = coord_gpu[2*idx];
    float y_pt = coord_gpu[2*idx+1];

    if(x_pt > (LK_width-1) ||
            y_pt > (LK_height-1)) // Useful check ?
        return;

    if(status_gpu[idx] == 0)
        return;

    float Vx, Vy;            // Previous speed for this point
    float cur_x, cur_y;      // Current position
    float sum_Ixx = 0.f;
    float sum_Ixy = 0.f;
    float sum_Iyy = 0.f;
    float sum_Ixt, sum_Iyt;
    float Ix, Iy, It;
    int j;                   // Research window indexes
    float xx, yy;
    float det;
    float vx, vy;

    x_pt *= LK_scaling;
    y_pt *= LK_scaling;

    if(LK_init_guess) {
        Vx = 0.f;
        Vy = 0.f;
        cur_x = x_pt;
        cur_y = y_pt;
    }
    else {
        Vx = dx_gpu[idx];
        Vy = dy_gpu[idx];
        cur_x = x_pt + Vx;
        cur_y = y_pt + Vy;
    }

    float temp_weight = 1.f;

    // Compute spatial gradient (only once) from texture fetches
    int win_size;

    if (rectified) {
        win_size = 2;
    } else {
        win_size = LK_patch;
    }

    for(yy=-win_size; yy <= win_size; ++yy) {
        for(xx=-LK_patch; xx <= LK_patch; ++xx) {

            temp_weight = LK_Weight[(int)(yy + xx*LK_patch)];
            temp_weight *= temp_weight;

            Ix = tex2D(gpu_textr_deriv_x, x_pt + LK_width_offset + xx, y_pt + yy);
            Iy = tex2D(gpu_textr_deriv_y, x_pt + LK_width_offset + xx, y_pt + yy);

            sum_Ixx += Ix * Ix * temp_weight;
            sum_Ixy += Ix * Iy * temp_weight;
            sum_Iyy += Iy * Iy * temp_weight;
        }
    }

    det = sum_Ixx*sum_Iyy - sum_Ixy*sum_Ixy;

    // Deal with case : could not track (no gradient)
    if(det < MIN_DET) {
        status_gpu[idx] = 0;
        return;
    }

    // Iteration part
    for(j=0; j < LK_iteration; ++j) {
        // If current speed vector drives the point out of bounds
        if(cur_x < 0.f ||
                cur_x > LK_pyr_w ||
                cur_y < 0.f ||
                cur_y > LK_pyr_h) {
            status_gpu[idx] = 0;
            return;
        }

        sum_Ixt = 0.f;
        sum_Iyt = 0.f;

        // No explicit handling of pixels outside the image
        // Texture fetchs ensure calls are clamped to window size
        for(yy=-win_size; yy <= win_size; ++yy) {
            for(xx=-LK_patch; xx <= LK_patch; ++xx) {
                It = tex2D(gpu_textr_pict_1, cur_x + LK_width_offset + xx, cur_y + yy)
                     - tex2D(gpu_textr_pict_0, x_pt + LK_width_offset + xx, y_pt + yy);

                temp_weight = LK_Weight[(int)(yy + xx*LK_patch)];
                temp_weight *= temp_weight;

                Ix = tex2D(gpu_textr_deriv_x, x_pt + LK_width_offset + xx, y_pt + yy);
                Iy = tex2D(gpu_textr_deriv_y, x_pt + LK_width_offset + xx, y_pt + yy);

                sum_Ixt += Ix * It * temp_weight;
                sum_Iyt += Iy * It * temp_weight;
            }
        }

        // Find the inverse of the 2x2 matrix using a mix of determinant and adjugate matrix
        // http://cnx.org/content/m19446/latest/
        vx = __fdividef((-sum_Iyy*sum_Ixt + sum_Ixy*sum_Iyt), det);
        vy = __fdividef(( sum_Ixy*sum_Ixt - sum_Ixx*sum_Iyt), det);

        Vx += vx;
        Vy += vy;
        cur_x += vx;
        cur_y += vy;

        // Stop if movement is very small
        if(fabsf(vx) < LK_threshold &&
                fabsf(vy) < LK_threshold)
            break;
    }

    // Double speed vector to get to next scale
    if(LK_pyr_level != 0) {
        Vx += Vx;
        Vy += Vy;
    }

    dx_gpu[idx] = Vx;
    dy_gpu[idx] = Vy;

    // Shift coordinates of the points to track
    if (LK_pyr_level == 0) {
        coord_gpu[2*idx  ] -= Vx;
        coord_gpu[2*idx+1] -= Vy;
    }
}
