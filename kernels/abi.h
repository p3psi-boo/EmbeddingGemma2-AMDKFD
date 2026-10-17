#ifndef EG2_KFD_ABI_H
#define EG2_KFD_ABI_H
typedef struct { float *out; const void *weight; const int *ids; unsigned n; } EgGather;
typedef struct { float *out; const float *x; const void *weight; unsigned n, k, m, type; } EgMM;
typedef struct { float *out; const float *x; const float *weight; unsigned rows, d; float scale; } EgNorm;
typedef struct { float *x; const float *table; unsigned n, heads, d; } EgRope;
typedef struct { float *out; const float *q, *k, *v; const unsigned *bounds; unsigned n, window; } EgAttention;
typedef struct { float *out; const float *x, *other; unsigned n, mode, layer; float scale; } EgElement;
typedef struct { float *out; const float *x; const unsigned *offsets; unsigned batch; } EgPool;
typedef struct { float *out; const float *x, *weight; unsigned rows, d; float scale; } EgNormResidual;
typedef struct { float *x; const float *weight, *table; unsigned rows, d, heads; } EgNormRope;
#endif
