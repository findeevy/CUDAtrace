#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../ray.h"
#include <cuda_runtime.h>


__device__ bool hit_sphere(const vec3& center, float radius, const ray& r){

    //vec3 from sphere center to origin
    vec3 oc = r.origin() - center;
    float a = r.direction().dot(r.direction());
    float b = 2.0f * oc.dot(r.direction());
    float c = oc.dot(oc) - radius*radius;
    float discriminant = b*b - 4.0f*a*c;
    // sphere is intersected if >= 0
    return discriminant > 0.0f;


}

__device__ vec3 color(const ray& r){

    // color red for sphere hit
    if(hit_sphere(vec3(0,0,-1),0.5,r)){
        return vec3(1,0,0);
    }
    vec3 unit_direction = r.direction().normalized();
    float t = 0.5f*unit_direction.y +1.0f;
    // sky gradient
    return (1.0f-t)*vec3(1.0,1.0,1.0)+ t*vec3(0.5,0.7,1.0);

}

__global__ void render(vec3 *fb, int max_x, int max_y,
                       vec3 low_left_corner, vec3 horizontal, vec3 vertical, vec3 origin ){
    //pixel coords i, j
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i>=max_x) || (j>= max_y)) return;

    // get index of pixel in the form of 1d array
    int pixel_idx = j*max_x+i;

    float u = float(i)/float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin,low_left_corner+u*horizontal+v*vertical);
    // frame buffer array is 1d array of pixels
    fb[pixel_idx] = color(r);
    

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
    render<<<blocks, threads>>>(frame_buffer, image_width, image_height,vec3(-2.0, -1.0, -1.0),
                                vec3(4.0, 0.0, 0.0),
                                vec3(0.0, 2.0, 0.0),
                                vec3(0.0, 0.0, 0.0));

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