#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>


struct vec3 {
    float x, y, z;

    __host__ __device__ inline
    vec3() : x(0.0f), y(0.0f), z(0.0f) {}

    __host__ __device__ inline
    vec3(float ix, float iy, float iz) : x(ix), y(iy), z(iz) {}

    __host__ __device__ inline
    vec3 operator+(float other) const {
        return vec3(x + other, y + other, z + other);
    }

    __host__ __device__ inline
    vec3 operator-(float other) const {
        return vec3(x - other, y - other, z - other);
    }

    __host__ __device__ inline
    vec3 operator*(float other) const {
        return vec3(x * other, y * other, z * other);
    }

    __host__ __device__ inline
    vec3 operator/(float other) const {
        if (fabsf(other) < 1e-8f) return vec3(0.0f, 0.0f, 0.0f);
        return vec3(x / other, y / other, z / other);
    }

    __host__ __device__ inline
    float dot(const vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ inline
    vec3 cross(const vec3& other) const {
        return vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    __host__ __device__ inline
    float magnitude() const {
        return sqrtf(x*x + y*y + z*z);
    }

    __host__ __device__ inline
    vec3 normalized() const {
      float mag = magnitude();
      return *this/mag;
    }

    __host__ __device__
    void display() const {
    #ifdef __CUDA_ARCH__
        printf("(%f, %f, %f)\n", x, y, z);
    #else
        std::cout << "(" << x << ", " << y << ", " << z << ")" << std::endl;
    #endif
    }
};

// external operaters

__host__ __device__ inline
vec3 operator+(const vec3& a, const vec3& b) {
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline
vec3 operator-(const vec3& a, const vec3& b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline
vec3 operator*(float scalar, const vec3& v) {
    return v * scalar;
}
