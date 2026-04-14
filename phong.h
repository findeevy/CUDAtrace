#ifndef PHONG_H
#define PHONG_H
#include "vec3.h"

__device__ float diffuse(vec3 normal, vec3 dir) {
    return fmaxf(0.0f, normal.dot(dir));
}


__device__ float specular(vec3 normal, vec3 dir, vec3 cam, float shine) {
  vec3 ndir = dir.normalized();
  vec3 ncam = cam.normalized(); 
  vec3 half = (ndir+ncam).normalized();
  return powf(fmaxf(0.0f, normal.dot(half)), shine); 
}
#endif