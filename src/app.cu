#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include "../ray.h"
#include "../vec3.h"
#include "../phong.h"
#include "../fxaa.h"
#include "../stochastic_ss.h"
#include <cuda_runtime.h>


__device__ float hit_sphere(const vec3& center, float radius, const ray& r){

    //vec3 from sphere center to origin
    vec3 oc = r.origin() - center;
    float a = r.direction().dot(r.direction());
    float b = 2.0f * oc.dot(r.direction());
    float c = oc.dot(oc) - radius*radius;
    float discriminant = b*b - 4.0f*a*c;

    if (discriminant < 0.0f) return -1.0f;  // no hit
    float sqrt_d = sqrtf(discriminant);

    float t1 = (-b - sqrt_d) / (2.0f * a);
    if (t1 > 0.0f) return t1;  // closer hit

    float t2 = (-b + sqrt_d) / (2.0f * a);
    if (t2 > 0.0f) return t2;  // farthest hit

    return -1.0f;  // both are negative, behind ray origin
}

__device__ vec3 color(const ray& r){

    float hit_distance = hit_sphere(vec3(0,0,-1), 0.5, r);
    if (hit_distance > 0.0f) {
        // hit the sphere, compute blinn phong lighting
        vec3 hit_point = r.point_at_parameter(hit_distance);
        vec3 normal = (hit_point - vec3(0,0,-1)).normalized();
        
        // point light position 
        vec3 light_pos(1, 1, 1);
        vec3 light_dir = (light_pos - hit_point).normalized();
        
        // camera direction (ray is coming from camera, so reverse it)
        vec3 cam_dir = -1* r.direction().normalized();
        
        // material properties
        vec3 material_color(1, 0, 0);  
        float shininess = 32.0f;
        
        // ambient lighting
        float ambient_strength = 0.1f;
        vec3 ambient = ambient_strength * material_color;
        
        // diffuse component
        float diff = diffuse(normal, light_dir);
        vec3 diffuse_color = diff * material_color;
        
        // specular component (blinn phong)
        float spec = specular(normal, light_dir, cam_dir, shininess);
        vec3 specular_color = spec * vec3(1, 1, 1);  //white highlight
        
        // combine all lighting
        vec3 result = ambient + diffuse_color + specular_color;
        result.x = fminf(1.0f, fmaxf(0.0f, result.x));
        result.y = fminf(1.0f, fmaxf(0.0f, result.y));
        result.z = fminf(1.0f, fmaxf(0.0f, result.z));
        return result;
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

int main(int argc, char** argv) {
    
    bool use_fxaa = false;
    bool use_fxaa_shm = false;
    bool use_ssaa = false;
    int ssaa_samples = 500;
    
    for (int a = 1; a < argc; ++a){
        if (strcmp(argv[a], "--fxaa") == 0) use_fxaa = true;
        if (strcmp(argv[a], "--fxaa_shm") == 0) use_fxaa_shm = true;
        if (strcmp(argv[a], "--ssaa") == 0) use_ssaa = true;
        
    }
    int image_width = 1024;
    int image_height = 1024;
    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // unified memory, allocated frame buffer, device memory
    vec3 *raw_frame_buffer;
    cudaError_t err = cudaMallocManaged((void **)&raw_frame_buffer, fb_size);
        if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    vec3 *fxaa_buffer = nullptr;
    if (use_fxaa || use_fxaa_shm)
        cudaMallocManaged((void **)&fxaa_buffer, fb_size);

    curandState *d_rand_state = nullptr;
    if (use_ssaa){
        cudaMalloc((void**)&d_rand_state, num_pixels*sizeof(curandState));
    }
    int threads_x =16;
    int threads_y=16;
    dim3 blocks(image_width/threads_x +1, image_height/threads_y +1);
    dim3 threads(threads_x, threads_y);

    auto start = std::chrono::high_resolution_clock::now();

    if(use_ssaa){
        // initialize random state
        ss_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
        cudaDeviceSynchronize();
        
        // LAUNCH KERNEL WITH STOCHASTIC SUPER SAMPLING ANTI-ALIASING
        render_ss<<<blocks, threads>>>(raw_frame_buffer, image_width, image_height,ssaa_samples,vec3(-2.0, -2.0, -1.0),
                                    vec3(4.0, 0.0, 0.0),
                                    vec3(0.0, 4.0, 0.0),
                                    vec3(0.0, 0.0, 0.0),
                                    d_rand_state);
        
    }else{
        // LAUNCH KERNEL with NO ANTI-ALIASING
        render<<<blocks, threads>>>(raw_frame_buffer, image_width, image_height,vec3(-2.0, -2.0, -1.0),
                                    vec3(4.0, 0.0, 0.0),
                                    vec3(0.0, 4.0, 0.0),
                                    vec3(0.0, 0.0, 0.0));

    }
                                err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(raw_frame_buffer);
        return 1;
    }

    // GPU finish
    cudaDeviceSynchronize();
    // do fxaa post-processing if flag is true
    if (use_fxaa) {
        fxaa_pass<<<blocks, threads>>>(raw_frame_buffer, fxaa_buffer,
                                       image_width, image_height);
        cudaDeviceSynchronize();
    }
    if (use_fxaa_shm) {
        fxaa_pass_shared<<<blocks, threads>>>(raw_frame_buffer, fxaa_buffer,
                                       image_width, image_height);
        cudaDeviceSynchronize();
    }
    vec3 *out = (use_fxaa || use_fxaa_shm)? fxaa_buffer : raw_frame_buffer; 

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Frame time was " << elapsed.count() << " seconds.\n";

    // output frame buffer as ppm (done with CPU)
    std::ofstream outFile("output.ppm");

    outFile << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;

            int ir = int(255.999 * out[pixel_index].x);
            int ig = int(255.999 * out[pixel_index].y);
            int ib = int(255.999 * out[pixel_index].z);

            outFile << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
    outFile.close();

    cudaFree(raw_frame_buffer);
    if (fxaa_buffer) cudaFree(fxaa_buffer);
    if(d_rand_state) cudaFree(d_rand_state);
    return 0;
}