#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../vec3.h"
#include <cuda_runtime.h>


__global__ void render(vec3 *fb, int max_x, int max_y){
    //pixel coords i, j
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i>=max_x) || (j>= max_y)) return;

    // get index of pixel in the form of 1d array
    int pixel_idx = j*max_x+i;
    float r = float(i) / (max_x - 1);
    float g = float(j) / (max_y - 1);
    float b = 0.0f;

    // frame buffer array is 1d array of pixels
    fb[pixel_idx] = vec3(r,g,b);
    

}

int main(){

    int image_width = 256;
    int image_height = 256;
    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // unified memory, allocated frame buffer, device memory
    vec3 *frame_buffer;
    cudaError_t err = cudaMallocManaged((void **)&frame_buffer, fb_size);
        if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    int threads_x =16;
    int threads_y=16;
    dim3 blocks(image_width/threads_x +1, image_height/threads_y +1);
    dim3 threads(threads_x, threads_y);

    // LAUNCH KERNEL
    render<<<blocks, threads>>>(frame_buffer, image_width, image_height);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(frame_buffer);
        return 1;
    }

    // GPU finish
    cudaDeviceSynchronize();

    // output frame buffer as ppm (done with CPU)
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for(int j = 0; j < image_height; j++) {
        for(int i = 0; i < image_width; i++) {
            size_t pixel_index = j*image_width + i;

            int ir = int(255.999 * frame_buffer[pixel_index].x);
            int ig = int(255.999 * frame_buffer[pixel_index].y);
            int ib = int(255.999 * frame_buffer[pixel_index].z);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    cudaFree(frame_buffer);
    return 0;
}