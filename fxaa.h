#ifndef FXAA_H
#define FXAA_H
#include "vec3.h"

// luminance perceived by the human eye, https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf (2.5.1)
__device__ inline float luma(vec3 c) { 
    return c.x * 0.299f + c.y * 0.587f + c.z * 0.114f;
}

__device__ inline float clampf(float v, float lo, float hi) {
    return fmaxf(lo, fminf(hi, v));
}

__global__ void fxaa_pass(const vec3* src, vec3* dst, int w, int h) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= w || j >= h) return;

    // --- tunables ---
    const float EDGE_THRESHOLD     = 0.125f;
    const float EDGE_THRESHOLD_MIN = 0.0312f;

    //safe pixel fetch (clamps to border)
    auto fetch = [&](int x, int y) -> vec3 {
        x = max(0, min(w - 1, x));
        y = max(0, min(h - 1, y));
        return src[y * w + x];
    };

    vec3 C  = fetch(i,   j);
    vec3 N  = fetch(i,   j-1);
    vec3 S  = fetch(i,   j+1);
    vec3 W  = fetch(i-1, j);
    vec3 E  = fetch(i+1, j);

    float lumaC = luma(C);
    float lumaN = luma(N);
    float lumaS = luma(S);
    float lumaW = luma(W);
    float lumaE = luma(E);

    float lumaMin = fminf(lumaC, fminf(fminf(lumaN, lumaS), fminf(lumaW, lumaE)));
    float lumaMax = fmaxf(lumaC, fmaxf(fmaxf(lumaN, lumaS), fmaxf(lumaW, lumaE)));
    
    float contrast = lumaMax - lumaMin;

    // Skip pixels that are not on an edge
    if (contrast < fmaxf(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD)) {
        dst[j * w + i] = C;
        return;
    }

    //diagonal neighbors for a better gradient estimate
    vec3 NW = fetch(i-1, j-1);
    vec3 NE = fetch(i+1, j-1);
    vec3 SW = fetch(i-1, j+1);
    vec3 SE = fetch(i+1, j+1);

    float lumaNW = luma(NW), lumaNE = luma(NE);
    float lumaSW = luma(SW), lumaSE = luma(SE);

    //horizontal + vertical edge detection via Sobel-style weights
    float edgeH = fabsf(lumaNW + 2.0f*lumaN + lumaNE - lumaSW - 2.0f*lumaS - lumaSE);
    float edgeV = fabsf(lumaNW + 2.0f*lumaW + lumaSW - lumaNE - 2.0f*lumaE - lumaSE);
    bool isHorizontal = edgeH >= edgeV;

    //choose the two pixels straddling the edge
    float luma1 = isHorizontal ? lumaN : lumaW;
    float luma2 = isHorizontal ? lumaS : lumaE;
    vec3  pix1  = isHorizontal ? N     : W;
    vec3  pix2  = isHorizontal ? S     : E;

    //gradient magnitudes toward each neighbor
    float grad1 = fabsf(luma1 - lumaC);
    float grad2 = fabsf(luma2 - lumaC);

    bool towards1 = grad1 >= grad2;
    float edgeBlend = towards1 ? grad1 / (grad1 + grad2 + 1e-5f)
                                : grad2 / (grad1 + grad2 + 1e-5f);
    edgeBlend = clampf(edgeBlend * 0.5f, 0.0f, 0.5f);

    //sub-pixel blend factor from the full 3x3 neighbourhood average
    // cardinals weighted 2, diagonals weighted 1: sum = 4*2 + 4*1 = 10,
    //excludes the center so we measure how much it deviates from the surroundings.
    float lumaAvg = (lumaN + lumaS + lumaW + lumaE) * (2.0f / 10.0f)
                  + (lumaNW + lumaNE + lumaSW + lumaSE) * (1.0f / 10.0f);
    float subBlend = fabsf(lumaAvg - lumaC) / contrast;
    subBlend = clampf(subBlend * subBlend * 0.75f, 0.0f, 1.0f);

    //take the stronger of the two blend values
    float blend = fmaxf(edgeBlend, subBlend);

    //lerp between center pixel and the neighbour on the stronger-gradient side
    vec3 chosen = towards1 ? pix1 : pix2;
    vec3 result;
    result.x = C.x + blend * (chosen.x - C.x);
    result.y = C.y + blend * (chosen.y - C.y);
    result.z = C.z + blend * (chosen.z - C.z);

    dst[j * w + i] = result;
}

#endif