
#include <cuda.h>
#include <cuda_runtime.h>


// Compute spatial derivatives using Scharr operator - Naive implementation..
__global__ void kernelScharrX( float const *in, int _w, int _h, float *out) {
    // Pattern : // Indexes :
    // -3 -10 -3 // a1 b1 c1
    //  0   0  0 // a2 b2 c2
    //  3  10  3 // a3 b3 c3

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= _w || y >= _h)
        return;

    int a = max(y-1,0);
    int b = y;
    int c = min((y+1),_h -1);

    int a1, a3,
            b1, b3,
            c1, c3;

    int i1 = max(x-1, 0);
    int i3 = min(x+1, _w-1);

    a1 = a*_w + i1;
    a3 = a*_w + i3;

    b1 = b*_w + i1;
    b3 = b*_w + i3;

    c1 = c*_w + i1;
    c3 = c*_w + i3;

    out[y*_w+x] = __fdividef(3.0 * (-in[a1]  -in[c1] + in[a3] + in[c3])
                             + 10.0 * (in[b3] -in[b1]), 20.0);

    //  out[y*_w+x] = -3.0*in[a1] -10.0*in[b1] -3.0*in[c1] + 3.0*in[a3] + 10.0*in[b3] + 3.0*in[c3];
}

// Compute spatial derivatives using Scharr operator - Naive implementation..
__global__ void kernelScharrY( float const *in, int _w, int _h, float *out )
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= _w || y >= _h)
        return;

    // Pattern  // Indexes:
    //  -3 0  3 // a1 b1 c1
    // -10 0 10 // a2 b2 c2
    //  -3 0  3 // a3 b3 c3

    int a = max(y-1,0);
    int c = min((y+1),_h -1);

    int a1, a2, a3,
            c1, c2, c3;

    int i1 = max(x-1, 0);
    int i3 = min(x+1, _w-1);

    a1 = a*_w + i1;
    a2 = a*_w + x;
    a3 = a*_w + i3;

    c1 = c*_w + i1;
    c2 = c*_w + x;
    c3 = c*_w + i3;

    out[y*_w+x] = __fdividef(3.0*(- in[a1] -in[a3] +in[c1] +in[c3])
                             + 10.0*(in[c2] -in[a2]), 20.0);

    //  out[y*_w+x] = -3.0*in[a1] -10.0*in[a2] -3.0*in[a3] + 3.0*in[c1] + 10.0*in[c2] + 3.0*in[c3];
}

// Compute spatial derivatives using Sobel operator - Naive implementation..
__global__ void kernelSobelX(float const * const in, int _w, int _h, float * const out) {
    // Pattern : // Indexes :
    // -1 -2 -1 // a1 b1 c1
    //  0  0  0 // a2 b2 c2
    //  1  2  1 // a3 b3 c3

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= _w || y >= _h)
        return;

    int a = max(y-1,0);
    int b = y;
    int c = min((y+1),_h -1);

    int a1, a3,
            b1, b3,
            c1, c3;

    int i1 = max(x-1, 0);
    int i3 = min(x+1, _w-1);

    a1 = a*_w + i1;
    a3 = a*_w + i3;

    b1 = b*_w + i1;
    b3 = b*_w + i3;

    c1 = c*_w + i1;
    c3 = c*_w + i3;

    out[y*_w+x] = __fdividef(-1.0 * in[a1] -2.0 * in[b1] -1.0 * in[c1] + 1.0 * in[a3] + 2.0 * in[b3] + 1.0 * in[c3], 4.0);
}

// Compute spatial derivatives using Sobel operator - Naive implementation..
__global__ void kernelSobelY( float const *in,
                              int _w, int _h,
                              float *out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= _w || y >= _h)
        return;

    // Pattern  // Indexes:
    //  -1 0 1 // a1 b1 c1
    // - 2 0 2 // a2 b2 c2
    //  -1 0 1 // a3 b3 c3

    int a = max(y-1,0);
    int c = min((y+1),_h -1);

    int a1, a2, a3,
            c1, c2, c3;

    int i1 = max(x-1, 0);
    int i3 = min(x+1, _w-1);

    a1 = a*_w + i1;
    a2 = a*_w + x;
    a3 = a*_w + i3;

    c1 = c*_w + i1;
    c2 = c*_w + x;
    c3 = c*_w + i3;

    out[y*_w + x] = __fdividef(-1.0*in[a1] -2.0*in[a2] -1.0*in[a3] + 1.0*in[c1] + 2.0*in[c2] + 1.0*in[c3], 4.0);
}

__global__ void kernelAdd(float const *in1,
                          float const *in2,
                          int _w,
                          int _h,
                          float *out) {

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= _w || y >= _h)
        return;

    out[y*_w + x] = __fsqrt_rn(__fadd_rn(__fmul_rn(in1[y*_w + x],in1[y*_w + x]), __fmul_rn(in2[y*_w + x],in2[y*_w + x])));
}


// Low pass gaussian-like filtering before subsampling
__global__ void kernelSmoothX(float *in, int w, int h, float *out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= w || y >= h)
        return;

    int idx = y*w;

    int a = x-2;
    int b = x-1;
    int c = x;
    int d = x+1;
    int e = x+2;

    if(a < 0) a = 0;
    if(b < 0) b = 0;
    if(d >= w) d = w-1;
    if(e >= w) e = w-1;

    out[y*w+x] = 0.0625f*in[idx+a] + 0.25f*in[idx+b] + 0.375f*in[idx+c] + 0.25f*in[idx+d] + 0.0625f*in[idx+e];
}

// Low pass gaussian-like filtering before subsampling
__global__ void kernelSmoothY(float const * in,
                              int w, int h,
                              float * out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= w || y >= h)
        return;

    int a = y-2;
    int b = y-1;
    int c = y;
    int d = y+1;
    int e = y+2;

    if(a < 0) a = 0;
    if(b < 0) b = 0;
    if(d >= h) d = h-1;
    if(e >= h) e = h-1;

    out[y*w+x] = 0.0625f*in[a*w+x] + 0.25f*in[b*w+x] + 0.375f*in[c*w+x] + 0.25f*in[d*w+x] + 0.0625f*in[e*w+x];
}

/*
// Upsample a picture using the "magic" kernel
__global__ void kernelMagicUpsampleX(float *in, int _w, int _h, float *out) {
  // Coefficients : 1/4, 3/4, 3/4, 1/4 in each direction (doubles the size of the picture)

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if(x >= _w || y >= _h)
    return;

  // Duplicate the points at the same place (?)
  out[y*2*_w + 2*x] = in[y*_w+x];


  if ((x < (_w-2)) && (x > 1))
    out[y*2*_w + 2*x + 1] = __fdividef(3.0*(in[y*_w+x] + in[y*_w + x + 1]) + in[y*_w+x -1] + in[y*_w+x +2] , 8.0);

}
*/
