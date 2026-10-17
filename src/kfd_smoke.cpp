#include "process_lock.h"
#include "gpu_target.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

// A bounded, small dispatch. Deliberately no fault, preemption or stress tests.
int main(int argc, char **argv) try {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s eg2-TARGET.elf\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary | std::ios::ate);
    if (!f || f.tellg() <= 0) return 2;
    std::vector<std::byte> image(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char *>(image.data()), image.size())) return 2;
    eg2::ProcessLock process_lock;
    auto ctx = KFD_EXPECT(kfd::Context::create());
    kfd::Device *selected = nullptr;
    for (auto &dev : ctx.devices()) {
        std::printf("GPU: %.*s gfx=%u device=0x%x\n", int(dev.name().size()),
                    dev.name().data(), dev.gfx_version(), dev.properties().device_id);
        if (!selected && eg2::target_name(dev.gfx_version())) selected = &dev;
    }
    if (!selected) {
        std::fprintf(stderr, "supported device not found\n");
        return 1;
    }
    auto &dev = *selected;
    auto queue = KFD_EXPECT(kfd::ComputeQueue::create(dev));
    auto exe = KFD_EXPECT(kfd::Executable::load(dev, image, queue));
    auto kernel = KFD_EXPECT(exe.kernel("eg2_saxpy.kd"));
    constexpr unsigned n = 65539; // Include a partial final workgroup.
    auto alloc = [&]() {
        auto b = KFD_EXPECT(kfd::Buffer::allocate(dev, n * sizeof(float), kfd::MemType::GTT));
        KFD_EXPECT(b.map(dev));
        return b;
    };
    auto xb = alloc(), yb = alloc();
    auto *x = static_cast<float *>(xb.data()), *y = static_cast<float *>(yb.data());
    for (unsigned i = 0; i < n; ++i) { x[i] = float(i % 1024); y[i] = -float(i % 97); }
    struct Args { float *y; const float *x; float a; unsigned n; };
    kfd::DispatchConfig cfg{.grid = {.x = (n + 255) / 256}, .block = {.x = 256}};
    auto args = KFD_EXPECT(kernel.alloc());
    kernel.fill(args, Args{y, x, 2.0f, n}, cfg);
    auto done = KFD_EXPECT(kfd::Signal::create(ctx));
    auto start = std::chrono::steady_clock::now();
    KFD_EXPECT(queue.command().dispatch(kernel, cfg, args).signal(done).submit());
    KFD_EXPECT(done.wait(kfd::Condition::EQ, 0, 5'000'000'000ULL));
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    unsigned errors = 0;
    for (unsigned i = 0; i < n; ++i) {
        float ref = 2.0f * float(i % 1024) - float(i % 97);
        if (!std::isfinite(y[i]) || y[i] != ref) {
            if (errors < 5) std::fprintf(stderr, "mismatch[%u]: %g != %g\n", i, y[i], ref);
            ++errors;
        }
    }
    std::printf("{\"test\":\"saxpy_gtt\",\"gfx\":%u,\"n\":%u,\"errors\":%u,"
                "\"dispatch_wait_us\":%.3f,\"runtime\":\"libkfd\"}\n", dev.gfx_version(), n, errors, us);
    return errors ? 1 : 0;
}

catch (const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
