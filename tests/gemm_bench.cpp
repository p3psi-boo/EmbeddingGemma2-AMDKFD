#include "../src/process_lock.h"
#include "../kernels/abi.h"
#include "../src/gpu_target.h"
#include <libkfd/libkfd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

// Standalone, bounded, no model allocation; each sample includes completion and
// cache dependencies identical to the backend. Deterministic inputs, verified
// against FP32 CPU FMA (all weight representations carry identical values).
int main(int argc, char **argv) try {
    if (argc != 2) return 2;
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0) return 2;
    std::vector<std::byte> image(size_t(file.tellg())); file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(image.data()), image.size())) return 2;
    eg2::ProcessLock process_lock;
    auto ctx = KFD_EXPECT(kfd::Context::create()); kfd::Device *dev = nullptr;
    dev=eg2::select_device(ctx);
    if (!dev) return 1;
    auto queue = KFD_EXPECT(kfd::ComputeQueue::create(*dev));
    auto exe = KFD_EXPECT(kfd::Executable::load(*dev, image, queue));
    auto alloc = [&](size_t bytes) {
        auto b = KFD_EXPECT(kfd::Buffer::allocate(*dev, bytes, kfd::MemType::GTT));
        KFD_EXPECT(b.map(*dev)); return b;
    };
    auto xb=alloc(1024*2048*4), wb=alloc(2048*2048*4), yb=alloc(1024*2048*4);
    auto fence=alloc(64), clocks=alloc(64);
    auto *x=static_cast<float *>(xb.data()), *y=static_cast<float *>(yb.data());
    auto *clock=static_cast<uint64_t *>(clocks.data());
    auto done=KFD_EXPECT(kfd::Signal::create(ctx));
    uint64_t seq=0; *static_cast<uint64_t *>(fence.data())=0;
    unsigned failures=0;
    struct Shape { unsigned n,k,m; };
    for (auto [n,k,m] : {Shape{40,512,512}, Shape{160,512,2048}, Shape{640,512,2048}, Shape{640,2048,512}}) {
        for (unsigned i=0; i<n*k; ++i) x[i]=std::sin(float(i)*0.13f)*0.125f;
        std::vector<float> ref(n*m);
        // Sample 64 output positions including tails, rather than spend seconds
        // performing a full scalar GEMM on the shared workstation.
        std::vector<unsigned> indices;
        for (unsigned j=0; j<64; ++j) {
            unsigned i=unsigned(uint64_t(j)*(n*m-1)/63); indices.push_back(i);
            for (unsigned z=0; z<k; ++z)
                ref[i]=std::fma(x[(i/m)*k+z], float(int(((i%m)*k+z)%23)-11)*0.125f, ref[i]);
        }
        for (unsigned type : {8u,1u,0u}) {
            for (unsigned i=0; i<k*m; ++i) {
                float w=float(int(i%23)-11)*0.125f;
                if (type==0) static_cast<float *>(wb.data())[i]=w;
                else if (type==1) static_cast<_Float16 *>(wb.data())[i]=(_Float16)w;
                else {
                    auto *b=static_cast<unsigned char *>(wb.data())+(i/32)*34;
                    b[0]=0; b[1]=0x30;
                    reinterpret_cast<signed char *>(b+2)[i%32]=int(i%23)-11;
                }
            }
            for (const char *name : {"eg2_mm64.kd", "eg2_mm64x64.kd", "eg2_mmstep8.kd"}) {
                unsigned height=std::strcmp(name,"eg2_mm64.kd") ? 64 : 16;
                auto kernel=KFD_EXPECT(exe.kernel(name)); auto args=KFD_EXPECT(kernel.alloc());
                kfd::DispatchConfig cfg{.grid={.x=(m+63)/64,.y=(n+height-1)/height},.block={.x=256}};
                kernel.fill(args,EgMM{y,x,wb.data(),n,k,m,type},cfg);
                std::vector<double> samples,wall;
                for (unsigned rep=0; rep<12; ++rep) {
                    KFD_EXPECT(done.reset());
                    auto start=std::chrono::steady_clock::now();
                    KFD_EXPECT(queue.command().acquire_mem().read_gpu_clock(clock)
                        .dispatch(kernel,cfg,args).release_mem(kfd::pm4::REL_ALL,fence.data(),++seq)
                        .wait_reg_mem(fence.data(),kfd::Condition::EQ,seq).acquire_mem()
                        .read_gpu_clock(clock+1).signal(done).submit(5'000'000'000ULL));
                    KFD_EXPECT(done.wait(kfd::Condition::EQ,0,5'000'000'000ULL));
                    if (rep>=3) {
                        samples.push_back(double(clock[1]-clock[0])*1000.0/dev->drm_properties().timestamp_freq);
                        wall.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
                    }
                }
                float error=0;
                for (unsigned i:indices) {
                    error=std::max(error,std::abs(y[i]-ref[i]));
                    if (!std::isfinite(y[i]) || std::abs(y[i]-ref[i])>2e-5f) ++failures;
                }
                std::sort(samples.begin(),samples.end()); std::sort(wall.begin(),wall.end());
                double ms=samples[samples.size()/2];
                std::printf("{\"kernel\":\"%s\",\"type\":%u,\"n\":%u,\"k\":%u,\"m\":%u,\"gpu_ms\":%.6f,\"wall_ms\":%.6f,\"gflops\":%.3f,\"max_abs_error\":%.9g,\"gpu_samples_ms\":[",
                    name,type,n,k,m,ms,wall[wall.size()/2],2.0*n*k*m/ms/1e6,error);
                for (size_t j=0;j<samples.size();++j) std::printf("%s%.6f",j?",":"",samples[j]);
                std::printf("],\"wall_samples_ms\":[");
                for (size_t j=0;j<wall.size();++j) std::printf("%s%.6f",j?",":"",wall[j]);
                std::printf("]}\n");
                std::fflush(stdout);
            }
        }
    }
    return failures?1:0;
}

catch (const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
