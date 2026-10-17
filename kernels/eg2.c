#include <gpuintrin.h>
#include "abi.h"
#define WAVE __AMDGCN_WAVEFRONT_SIZE

// No OCML/device-library calls: only LLVM AMDGPU intrinsics and plain C.
static inline unsigned index1(void) {
    return __gpu_thread_id_x() + __gpu_block_id_x() * __gpu_num_threads_x();
}
static inline float weight(const void *w, unsigned i, unsigned type) {
    if (type == 1) return (float)((const _Float16 *)w)[i];
    if (type == 0) return ((const float *)w)[i];
    if (type == 30) return __builtin_bit_cast(float, (unsigned)((const unsigned short *)w)[i] << 16);
    // GGUF Q8_0: little-endian fp16 scale, then 32 signed int8 values.
    const unsigned char *b = (const unsigned char *)w + (i / 32) * 34;
    unsigned short bits = b[0] | ((unsigned short)b[1] << 8);
    return (float)__builtin_bit_cast(_Float16, bits) * ((const signed char *)(b + 2))[i % 32];
}
static inline float wave_sum(float x) {
    unsigned lane = __gpu_thread_id_x() & (WAVE-1);
    for (unsigned s = WAVE/2; s; s >>= 1)
        x += __gpu_shuffle_idx_f32(~0ULL, lane ^ s, x, WAVE);
    return x;
}
static inline float ex(float x) { return __builtin_amdgcn_exp2f(x * 1.4426950408889634f); }
static inline float gelu(float x) {
    float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    // x*(1+tanh(u))/2, expressed without an external tanhf symbol.
    return x / (1.0f + ex(-2.0f * u));
}

__gpu_kernel void eg2_saxpy(float *y, const float *x, float a, unsigned n) {
    unsigned i = __gpu_thread_id_x() + __gpu_block_id_x() * __gpu_num_threads_x();
    if (i < n) y[i] = a * x[i] + y[i];
}

__gpu_kernel void eg2_gather(EgGather p) {
    unsigned i = index1();
    if (i < p.n * 512) p.out[i] = weight(p.weight, (unsigned)p.ids[i / 512] * 512 + i % 512, 8) * 22.62741699796952f;
}

// First correctness backend: 16x16 FP32 tiled GEMM, Q8_0/BF16/F32 weights
// decoded into LDS without a second full-sized floating-point model copy.
__gpu_kernel void eg2_mm(EgMM p) {
    [[clang::loader_uninitialized]] static __gpu_local float a[16][16], b[16][16];
    unsigned lane = __gpu_thread_id_x(), r = lane / 16, c = lane % 16;
    unsigned row = __gpu_block_id_y() * 16 + r, col = __gpu_block_id_x() * 16 + c;
    float acc = 0;
    for (unsigned base = 0; base < p.k; base += 16) {
        a[r][c] = row < p.n && base + c < p.k ? p.x[row * p.k + base + c] : 0;
        unsigned wr = __gpu_block_id_x() * 16 + r;
        b[r][c] = wr < p.m && base + c < p.k ? weight(p.weight, wr * p.k + base + c, p.type) : 0;
        __gpu_sync_threads();
        for (unsigned j = 0; j < 16; ++j) acc = __builtin_fmaf(a[r][j], b[c][j], acc);
        __gpu_sync_threads();
    }
    if (row < p.n && col < p.m) p.out[row * p.m + col] = acc;
}

// Opt-in measured candidate: four outputs/thread, padded LDS to eliminate the
// stride-16 bank conflicts of the baseline. Same FP32 accumulation order in K.
__gpu_kernel void eg2_mm64(EgMM p) {
    [[clang::loader_uninitialized]] static __gpu_local float a[16][33], b[64][33];
    unsigned lane=__gpu_thread_id_x(), r=lane/16, c=lane%16;
    unsigned row=__gpu_block_id_y()*16+r, col=__gpu_block_id_x()*64+c;
    float acc[4]={0};
    for (unsigned base=0; base<p.k; base+=32) {
        for (unsigned j=0; j<2; ++j)
            a[r][c+j*16]=row<p.n && base+c+j*16<p.k ? p.x[row*p.k+base+c+j*16] : 0;
        for (unsigned j=0; j<8; ++j) {
            unsigned br=lane/32+j*8, bc=lane%32, wr=__gpu_block_id_x()*64+br;
            b[br][bc]=wr<p.m && base+bc<p.k ? weight(p.weight,wr*p.k+base+bc,p.type) : 0;
        }
        __gpu_sync_threads();
        #pragma unroll
        for (unsigned j=0; j<32; ++j) {
            float av=a[r][j];
            #pragma unroll
            for (unsigned o=0; o<4; ++o) acc[o]=__builtin_fmaf(av,b[c+o*16][j],acc[o]);
        }
        __gpu_sync_threads();
    }
    for (unsigned o=0; o<4; ++o) if (row<p.n && col+o*16<p.m) p.out[row*p.m+col+o*16]=acc[o];
}

// Same 64x64 output tile and FP32 K order. STEP=8 reduces LDS from
// 16896 to 4608 bytes and VGPRs from 62 to 43 on Clang 21 / gfx90c.
#define MM_LARGE(NAME, STEP) \
__gpu_kernel void NAME(EgMM p) {\
    [[clang::loader_uninitialized]] static __gpu_local float a[64][STEP+1], b[64][STEP+1];\
    unsigned lane=__gpu_thread_id_x(), r=lane/16, c=lane%16;\
    unsigned row=__gpu_block_id_y()*64+r, col=__gpu_block_id_x()*64+c;\
    float acc[4][4]={{0}};\
    for (unsigned base=0; base<p.k; base+=STEP) {\
        for (unsigned j=0; j<STEP/4; ++j) {\
            unsigned ar=lane/STEP+j*(256/STEP), ac=lane%STEP, xr=__gpu_block_id_y()*64+ar;\
            a[ar][ac]=xr<p.n && base+ac<p.k ? p.x[xr*p.k+base+ac] : 0;\
        }\
        for (unsigned j=0; j<STEP/4; ++j) {\
            unsigned br=lane/STEP+j*(256/STEP), bc=lane%STEP, wr=__gpu_block_id_x()*64+br;\
            b[br][bc]=wr<p.m && base+bc<p.k ? weight(p.weight,wr*p.k+base+bc,p.type) : 0;\
        }\
        __gpu_sync_threads();\
        _Pragma("unroll")\
        for (unsigned j=0; j<STEP; ++j) {\
            _Pragma("unroll")\
            for (unsigned rr=0; rr<4; ++rr) {\
                float av=a[r+rr*16][j];\
                _Pragma("unroll")\
                for (unsigned o=0; o<4; ++o) acc[rr][o]=__builtin_fmaf(av,b[c+o*16][j],acc[rr][o]);\
            }\
        }\
        __gpu_sync_threads();\
    }\
    for (unsigned rr=0; rr<4; ++rr) for (unsigned o=0; o<4; ++o)\
        if (row+rr*16<p.n && col+o*16<p.m) p.out[(row+rr*16)*p.m+col+o*16]=acc[rr][o];\
}
MM_LARGE(eg2_mm64x64, 32)
MM_LARGE(eg2_mmstep8, 8)
#undef MM_LARGE

// One wave per row. In-place operation is valid: all reads precede the writes.
__gpu_kernel void eg2_norm(EgNorm p) {
    unsigned row = __gpu_block_id_x(), lane = __gpu_thread_id_x();
    if (row >= p.rows) return;
    float ss = 0;
    for (unsigned z = lane; z < p.d; z += WAVE) {
        float x = p.x[row * p.d + z] * p.scale;
        ss += x * x;
    }
    float inv = __builtin_amdgcn_rsqf(wave_sum(ss) / (float)p.d + 1.0e-6f);
    for (unsigned z = lane; z < p.d; z += WAVE)
        p.out[row * p.d + z] = p.x[row * p.d + z] * p.scale * inv * (p.weight ? p.weight[z] : 1.0f);
}

__gpu_kernel void eg2_rope(EgRope p) {
    unsigned i = index1(), half = p.d / 2;
    if (i >= p.n * p.heads * half) return;
    unsigned z = i % half, row = i / half, t = row / p.heads;
    float c = p.table[t * p.d + z], s = p.table[t * p.d + half + z];
    float a = p.x[row * p.d + z], b = p.x[row * p.d + half + z];
    p.x[row * p.d + z] = a * c - b * s;
    p.x[row * p.d + half + z] = a * s + b * c;
}

// Streaming softmax: no N*N score allocation, no cross-sequence attention.
// Specialized d keeps per-lane Q and accumulator arrays in registers.
#define ATTENTION(D) \
__gpu_kernel void eg2_attention##D(EgAttention p) { \
    unsigned row = __gpu_block_id_x(), lane = __gpu_thread_id_x(); \
    unsigned t = row / 4, h = row % 4, kh = h / (4 / (512 / D)); \
    if (t >= p.n) return; \
    unsigned lo = p.bounds[2*t], hi = p.bounds[2*t+1]; \
    if (p.window && t > p.window && lo < t-p.window) lo = t-p.window; \
    if (p.window && hi > t+p.window+1) hi = t+p.window+1; \
    float q[D/WAVE], acc[D/WAVE] = {0}; \
    for (unsigned j = 0; j < D/WAVE; ++j) q[j] = p.q[row*D + lane+j*WAVE]; \
    float max = -__builtin_inff(), den = 0; \
    for (unsigned s = lo; s < hi; ++s) { \
        float dot = 0; \
        for (unsigned j = 0; j < D/WAVE; ++j) dot += q[j] * p.k[s*512+kh*D+lane+j*WAVE]; \
        dot = wave_sum(dot); \
        float next = __builtin_fmaxf(max, dot), alpha = ex(max-next), beta = ex(dot-next); \
        den = den*alpha + beta; \
        for (unsigned j = 0; j < D/WAVE; ++j) acc[j] = acc[j]*alpha + beta*p.v[s*512+kh*D+lane+j*WAVE]; \
        max = next; \
    } \
    for (unsigned j = 0; j < D/WAVE; ++j) p.out[row*D+lane+j*WAVE] = acc[j]/den; \
}
ATTENTION(256)
ATTENTION(512)

// mode 0: residual/scale; 1: GELU gate; 2: layer-indexed PLE gate.
__gpu_kernel void eg2_element(EgElement p) {
    unsigned i = index1();
    if (i >= p.n) return;
    if (p.mode == 0) p.out[i] = (p.x[i] + p.other[i]) * p.scale;
    else if (p.mode == 1) p.out[i] = gelu(p.x[i]) * p.other[i];
    else p.out[i] = gelu(p.x[i]) * p.other[(i/512*24+p.layer)*512+i%512];
}

__gpu_kernel void eg2_pool(EgPool p) {
    unsigned i = index1(), seq = i / 768, z = i % 768;
    if (seq >= p.batch) return;
    unsigned lo = p.offsets[seq], hi = p.offsets[seq+1];
    float sum = 0;
    for (unsigned t = lo; t < hi; ++t) sum += p.x[t*768+z];
    p.out[i] = sum / (float)(hi-lo);
}

// Preserve the separate-kernel arithmetic order: the normalized product must
// round before residual addition / RoPE multiplication (no contracted FMA).
__gpu_kernel void eg2_norm_residual(EgNormResidual p) {
    #pragma clang fp contract(off)
    unsigned row=__gpu_block_id_x(), lane=__gpu_thread_id_x();
    float ss=0;
    for (unsigned z=lane; z<p.d; z+=WAVE) { float x=p.x[row*p.d+z]; ss+=x*x; }
    float inv=__builtin_amdgcn_rsqf(wave_sum(ss)/(float)p.d+1.0e-6f);
    for (unsigned z=lane; z<p.d; z+=WAVE) {
        unsigned i=row*p.d+z;
        p.out[i]=(p.out[i]+p.x[i]*inv*p.weight[z])*p.scale;
    }
}

__gpu_kernel void eg2_norm_rope(EgNormRope p) {
    unsigned row=__gpu_block_id_x(), lane=__gpu_thread_id_x(), half=p.d/2;
    float ss=0;
    for (unsigned z=lane; z<p.d; z+=WAVE) { float x=p.x[row*p.d+z]; ss+=x*x; }
    float inv=__builtin_amdgcn_rsqf(wave_sum(ss)/(float)p.d+1.0e-6f);
    for (unsigned z=lane; z<half; z+=WAVE) {
        float a=p.x[row*p.d+z]*inv*p.weight[z];
        float b=p.x[row*p.d+half+z]*inv*p.weight[half+z];
        unsigned t=row/p.heads;
        float c=p.table[t*p.d+z], s=p.table[t*p.d+half+z];
        p.x[row*p.d+z]=a*c-b*s;
        p.x[row*p.d+half+z]=a*s+b*c;
    }
}
