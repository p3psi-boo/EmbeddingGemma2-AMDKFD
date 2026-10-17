#ifndef EG2_KFD_H
#define EG2_KFD_H
#ifdef __cplusplus
extern "C" {
#endif
#include "gguf.h"
void *ei_gemma2_kfd_create(const ei_gguf *g, char *err, size_t len);
void ei_gemma2_kfd_free(void *ctx);
bool ei_gemma2_kfd_batch(void *ctx, const int32_t *ids, const size_t *offsets,
                       size_t batch, float *out, char *err, size_t len);
#ifdef __cplusplus
}
#endif
#endif
