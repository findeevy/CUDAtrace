#ifndef STOCHASTIC_SS_H
#define STOCHASTIC_SS_H
#include "vec3.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "ray.h"

__device__ vec3 color(const ray& r);

__global__ void ss_init(int max_x, int max_y, curandState *rand_state){
    //pixel coords i, j
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i>=max_x) || (j>= max_y)) return;

    // get index of pixel in the form of 1d array
    int pixel_idx = j*max_x+i;
    // use a fixed seed for each thread and pixel index as sequence
    // use pixel index as sequence number so pixel streams dont interfere wiith each other
    curand_init(1, pixel_idx,0, &rand_state[pixel_idx]);

}

__global__ void render_ss(vec3 *fb, int max_x, int max_y,int samples,
                       vec3 low_left_corner, vec3 horizontal,
                        vec3 vertical, vec3 origin, curandState *rand_state ){
    //pixel coords i, j
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i>=max_x) || (j>= max_y)) return;

    // get index of pixel in the form of 1d array
    int pixel_idx = j*max_x+i;
    // fast local register holds random number to compute with
    curandState local_rng = rand_state[pixel_idx];
    vec3 col(0,0,0);

    for (int s =0; s < samples; s++){

        float u = float(i + curand_uniform(&local_rng))/float(max_x);
        float v = float(j+ curand_uniform(&local_rng)) / float(max_y);
        ray r(origin,low_left_corner+u*horizontal+v*vertical);
        col = col+ color(r);
            
    }
    // frame buffer array is 1d array of pixels
    fb[pixel_idx] = col/float(samples);
    rand_state[pixel_idx] = local_rng;
    

}

#endif