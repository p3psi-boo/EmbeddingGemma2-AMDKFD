#ifndef EG2_GPU_TARGET_H
#define EG2_GPU_TARGET_H
#include <libkfd/libkfd.h>

namespace eg2 {
inline const char *target_name(uint32_t version) {
    switch (version) {
    case 90012: return "gfx90c";
    case 110003: return "gfx1103";
    default: return nullptr;
    }
}
inline unsigned wave_size(const kfd::Device &device) {
    return device.gfx_version()==110003 ? 32 : 64;
}
inline kfd::Device *select_device(kfd::Context &context) {
    for (auto &device : context.devices())
        if (target_name(device.gfx_version())) return &device;
    return nullptr;
}
}
#endif
