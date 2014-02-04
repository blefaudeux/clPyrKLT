
__constant __global int   LK_iteration;
__constant __global int   LK_patch;
__constant __global int   LK_points;
__constant __global int   LK_height;
__constant __global int   LK_width;
__constant __global int   LK_pyr_w;
__constant __global int   LK_pyr_h;
__constant __global int   LK_pyr_level;
__constant __global int   LK_width_offset;
__constant __global char  LK_init_guess;
__constant __global float LK_scaling;
__constant __global float LK_threshold;
__constant __global float LK_Weight[MAX_WEIGHT_VALUES];
__constant __global int   LK_win_size;


__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;


// Convert RGB Picture to grey/float
__kernel void convertRGBToGrey(unsigned char *d_in, float *d_out, int N)
{
int idx = get_global_id(0);

  if(idx < N)
  d_out[idx] = d_in[idx*3]*0.1144f + d_in[idx*3+1]*0.5867f + d_in[idx*3+2]*0.2989f;
  }

// Convert Grey uchar picture to float
__kernel void convertGreyToFloat(unsigned char *d_in, float *d_out, int N)
{
int idx = get_group_id(0)*get_num_groups(0) + get_local_id;

  if(idx < N)
  d_out[idx] = __fdividef((float) d_in[idx], 254.0);
  }

// Downsample picture to build pyramid lower level (naive implementation..)
__kernel void pyrDownsample(float *in, int w1, int h1, float *out, int w2, int h2)
{
// Input has to be greyscale
int x2 = get_global_id(0);
int y2 = get_global_id(1);

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


// Upsample a picture using the "magic" kernel
__kernel void kernelMagicUpsampleX(float *in, int _w, int _h, float *out) {
// Coefficients : 1/4, 3/4, 3/4, 1/4 in each direction (doubles the size of the picture)

  int x = get_global_id(0);
  int y = get_global_id(1);

  if(x >= _w || y >= _h)
  return;

  // Duplicate the points at the same place (?)
  out[y*2*_w + 2*x] = in[y*_w+x];


  if ((x < (_w-2)) && (x > 1))
  out[y*2*_w + 2*x + 1] = __fdividef(3.0*(in[y*_w+x] + in[y*_w + x + 1]) + in[y*_w+x -1] + in[y*_w+x +2] , 8.0);

}

/*
__global__ void kernelMagicUpsampleY(float *in, int w, int h, float *out) {
// Coefficients : 1/4, 3/4, 3/4, 1/4 in each direction (doubles the size of the picture)

}


__global__ void kernelDiffOperator(float *in, int w, int h, float *out) {

}
*/

// Compute spatial derivatives using Scharr operator - Naive implementation..
__kernel void kernelScharrX(const float *in, int _w, int _h, float *out) {
// Pattern : // Indexes :
// -3 -10 -3 // a1 b1 c1
//  0   0  0 // a2 b2 c2
//  3  10  3 // a3 b3 c3

  int x = get_global_id(0);
  int y = get_global_id(1);

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

  out[y*_w+x] = __fdividef(3.0 * (-in[a1]  -in[c1] + in[a3] + in[c3]) + 10.0 * (in[b3] -in[b1]), 20.0);
  //  out[y*_w+x] = -3.0*in[a1] -10.0*in[b1] -3.0*in[c1] + 3.0*in[a3] + 10.0*in[b3] + 3.0*in[c3];
  }

// Compute spatial derivatives using Scharr operator - Naive implementation..
__kernel void kernelScharrY(const float *in, int _w, int _h, float *out) {
int x = get_global_id(0);
int y = get_global_id(1);

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

  out[y*_w+x] = __fdividef(3.0*(- in[a1] -in[a3] +in[c1] +in[c3]) + 10.0*(in[c2] -in[a2]), 20.0);
  //  out[y*_w+x] = -3.0*in[a1] -10.0*in[a2] -3.0*in[a3] + 3.0*in[c1] + 10.0*in[c2] + 3.0*in[c3];
  }

// Compute spatial derivatives using Sobel operator - Naive implementation..
__kernel void kernelSobelX(const float *in, int _w, int _h, float *out) {
// Pattern : // Indexes :
// -1 -2 -1 // a1 b1 c1
//  0  0  0 // a2 b2 c2
//  1  2  1 // a3 b3 c3

  int x = get_global_id(0);
  int y = get_global_id(1);

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

  out[y*_w+x] = __fdividef(-1.0*in[a1] -2.0*in[b1] -1.0*in[c1] + 1.0*in[a3] + 2.0*in[b3] + 1.0*in[c3], 4.0);
  }

// Compute spatial derivatives using Scharr operator - Naive implementation..
__kernel void kernelSobelY(__global const float *in,
int _w,
int _h,
__global float *out) {

  int x = get_global_id(0);
  int y = get_global_id(1);

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

__kernel void kernelAdd(__global const float *in1,
                        __global const float *in2,
                        int _w,
                        int _h,
                        __global float *out) {

  int x = get_global_id(0);
  int y = get_global_id(1);

  out[y*_w + x] = __fsqrt_rn(__fadd_rn(__fmul_rn(in1[y*_w + x],in1[y*_w + x]), __fmul_rn(in2[y*_w + x],in2[y*_w + x])));

}


// Low pass gaussian-like filtering before subsampling
__kernel void kernelSmoothX(__global const float *in,
                            int w, int h,
                            __global float *out)
{
int x = get_global_id(0);
int y = get_global_id(1);

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
__kernel void kernelSmoothY(__global float *in,
                            int w, int h,
                            __global float *out)
{
int x = get_global_id(0);
int y = get_global_id(1);

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


__kernel void compute_spatial_grad( __global const float  *coord_gpu,
                                    __global char   *status_gpu,
                                    __global float  *gpu_neighbourhood_det,
                                    __global float  *gpu_neighbourhood_Iyy,
                                    __global float  *gpu_neighbourhood_Ixy,
                                    __global float  *gpu_neighbourhood_Ixx) {

  int idx = get_global_id(0);

  if (idx >= LK_points) return;

  float x_pt = coord_gpu[2*idx];
  float y_pt = coord_gpu[2*idx+1];

  if(x_pt > (LK_width-1) ||
    y_pt > (LK_height-1)) // Useful check ?

return;

  if(status_gpu[idx] == 0) return;

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
  if(gpu_neighbourhood_det[idx] < MIN_DET) {
  status_gpu[idx] = 0;
  return;
  }
  }

// Stripped-down kernel to compute the tracking part
__kernel void track_pts_slim (__global float *coord_gpu,
                              __global float *dx_gpu,
                              __global float *dy_gpu,
                              __global char  *status_gpu,
                              __global const float *gpu_neighbourhood_det,
                              __global const float *gpu_neighbourhood_Iyy,
                              __global const float *gpu_neighbourhood_Ixy,
                              __global const float *gpu_neighbourhood_Ixx)
{
int idx = get_global_id(0);

  if (idx >= LK_points)
  return;

  float x_pt = coord_gpu[2*idx];
  float y_pt = coord_gpu[2*idx+1];

  if(x_pt > (LK_width-1) ||
  y_pt > (LK_height-1))
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

  // Iteration part
  for(j=0; j < LK_iteration; ++j) {
  // If current speed vector drives the point out of bounds
  if(cur_x < 0.f ||
  cur_x > LK_pyr_w ||
  cur_y < 0.f ||
  cur_y > LK_pyr_h) {

          dx_gpu[idx] = 0.f;
          dy_gpu[idx] = 0.f;
          status_gpu[idx] = 0;
          return;
          }

      sum_Ixt = 0.f;
      sum_Iyt = 0.f;

      // No explicit handling of pixels outside the image
      // Texture fetchs ensure calls are clamped to window size
      for(yy=-LK_patch; yy <= LK_patch; ++yy) {
      for(xx=-LK_win_size; xx <= LK_win_size; ++xx) {
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

// Kernel to compute the tracking
__kernel void track_pts_weighted( __global float *coord_gpu,
                                  __global float *dx_gpu,
                                  __global float *dy_gpu,
                                  __global char  *status_gpu,
                                  bool  rectified)
{
int idx = get_global_id(0); // "2D" indexing

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
