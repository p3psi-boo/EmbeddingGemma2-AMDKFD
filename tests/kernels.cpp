#include "../src/process_lock.h"
#include "../kernels/abi.h"
#include "../src/gpu_target.h"
#include <libkfd/libkfd.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <optional>

int main(int argc, char **argv) try {
    if (argc!=2 && !(argc==3 && std::strcmp(argv[2],"--aql")==0)) return 2;
    std::ifstream f(argv[1],std::ios::binary|std::ios::ate);
    if (!f || f.tellg()<=0) return 2;
    std::vector<std::byte> image(size_t(f.tellg())); f.seekg(0);
    if (!f.read(reinterpret_cast<char *>(image.data()),image.size())) return 2;
    eg2::ProcessLock process_lock;
    auto ctx=KFD_EXPECT(kfd::Context::create()); kfd::Device *dev=nullptr;
    dev=eg2::select_device(ctx);
    if (!dev) return 1;
    std::optional<kfd::ComputeQueue> queue;
    queue.emplace(KFD_EXPECT(kfd::ComputeQueue::create(*dev)));
    auto exe=KFD_EXPECT(kfd::Executable::load(*dev,image,*queue));
    std::optional<kfd::AQLQueue> aql;
    if (argc==3) {
        queue.reset(); // Loader has waited; never overlap PM4 and AQL queues.
        aql.emplace(KFD_EXPECT(kfd::AQLQueue::create(*dev)));
    }
    auto alloc=[&](size_t size) {
        auto b=KFD_EXPECT(kfd::Buffer::allocate(*dev,size,kfd::MemType::GTT));
        KFD_EXPECT(b.map(*dev)); return b;
    };
    auto xb=alloc(1<<20), wb=alloc(1<<20), yb=alloc(1<<20);
    auto *x=static_cast<float *>(xb.data()), *y=static_cast<float *>(yb.data());
    auto run=[&]<class T>(const char *name,T p,unsigned gx,unsigned block=256,unsigned gy=1) {
        if (block==64) block=eg2::wave_size(*dev);
        auto k=KFD_EXPECT(exe.kernel(name)); auto args=KFD_EXPECT(k.alloc());
        kfd::DispatchConfig cfg{.grid={.x=gx,.y=gy},.block={.x=block}}; k.fill(args,p,cfg);
        auto done=KFD_EXPECT(kfd::Signal::create(ctx));
        if (aql) {
            constexpr kfd::aql::Ordering order{true,kfd::aql::FENCE_SCOPE_SYSTEM,kfd::aql::FENCE_SCOPE_SYSTEM};
            KFD_EXPECT(aql->command().dispatch(k,cfg,args,order,&done).submit(5'000'000'000ULL));
        } else {
            KFD_EXPECT(queue->command().acquire_mem().dispatch(k,cfg,args).signal(done).submit(5'000'000'000ULL));
        }
        KFD_EXPECT(done.wait(kfd::Condition::EQ,0,5'000'000'000ULL));
    };
    unsigned failures=0, cases=0; float worst=0;
    auto compare=[&](float got,float expected,float tolerance) {
        float e=std::abs(got-expected); worst=std::max(e,worst);
        if (!std::isfinite(got) || e>tolerance) {
            if (failures<5) std::fprintf(stderr,"got %g expected %g error %g\n",got,expected,e);
            ++failures;
        }
    };
    for (unsigned variant:{0u,1u,2u,3u}) for (unsigned type:{0u,1u,8u,30u}) for (unsigned n:{1u,17u,33u,65u}) {
        unsigned width=variant?64:16, height=variant>=2?64:16;
        constexpr unsigned k=64,m=35;
        std::vector<float> w(k*m);
        for (unsigned i=0;i<n*k;++i) x[i]=float(int(i%19)-9)*0.03125f;
        for (unsigned i=0;i<k*m;++i) w[i]=float(int(i%23)-11)*0.125f;
        if (type==0) std::memcpy(wb.data(),w.data(),w.size()*4);
        if (type==1) for (unsigned i=0;i<k*m;++i) static_cast<_Float16 *>(wb.data())[i]=(_Float16)w[i];
        if (type==30) for (unsigned i=0;i<k*m;++i) {
            uint32_t bits; std::memcpy(&bits,&w[i],4);
            static_cast<uint16_t *>(wb.data())[i]=uint16_t(bits>>16);
        }
        if (type==8) for (unsigned i=0;i<k*m/32;++i) {
            auto *b=static_cast<unsigned char *>(wb.data())+i*34;
            b[0]=0; b[1]=0x30; // fp16 0.125
            for (unsigned z=0;z<32;++z) reinterpret_cast<signed char *>(b+2)[z]=int((i*32+z)%23)-11;
        }
        run(variant==3?"eg2_mmstep8.kd":variant==2?"eg2_mm64x64.kd":(variant==1?"eg2_mm64.kd":"eg2_mm.kd"),EgMM{y,x,wb.data(),n,k,m,type},(m+width-1)/width,256,(n+height-1)/height);
        for (unsigned t=0;t<n;++t) for (unsigned r=0;r<m;++r) {
            float sum=0; for (unsigned z=0;z<k;++z) sum+=x[t*k+z]*w[r*k+z];
            compare(y[t*m+r],sum,2e-5f);
        }
        ++cases;
    }
    for (unsigned d:{256u,512u}) {
        auto *w=static_cast<float *>(wb.data());
        for (unsigned z=0;z<d;++z) w[z]=0.5f+float(z%7)*0.1f;
        for (unsigned i=0;i<3*d;++i) x[i]=std::sin(float(i)*0.13f);
        run("eg2_norm.kd",EgNorm{y,x,w,3,d,1},3,64);
        for (unsigned t=0;t<3;++t) {
            double ss=0; for (unsigned z=0;z<d;++z) ss+=double(x[t*d+z])*x[t*d+z];
            for (unsigned z=0;z<d;++z) compare(y[t*d+z],x[t*d+z]*w[z]/std::sqrt(float(ss/d)+1e-6f),2e-5f);
        }
        ++cases;
    }
    // Fused operators compared to the exact unfused GPU sequence, including
    // in-place aliases and per-sequence RoPE tables / per-head weights.
    auto rb=alloc(6*512*4), tb=alloc(3*512*4), save=alloc(6*512*4);
    auto *res=static_cast<float *>(rb.data()), *table=static_cast<float *>(tb.data());
    for (unsigned d:{256u,512u}) {
        auto *w=static_cast<float *>(wb.data());
        for (unsigned z=0;z<d;++z) w[z]=0.5f+float(z%7)*0.1f;
        for (unsigned i=0;i<6*d;++i) { x[i]=std::sin(float(i)*0.13f); res[i]=std::cos(float(i)*0.17f); }
        std::memcpy(save.data(),res,6*d*4);
        run("eg2_norm.kd",EgNorm{y,x,w,6,d,1},6,64);
        run("eg2_element.kd",EgElement{res,res,y,6*d,0,0,0.7f},(6*d+255)/256);
        run("eg2_norm_residual.kd",EgNormResidual{static_cast<float *>(save.data()),x,w,6,d,0.7f},6,64);
        for (unsigned i=0;i<6*d;++i) compare(static_cast<float *>(save.data())[i],res[i],2e-6f);
        for (unsigned t=0;t<3;++t) for (unsigned z=0;z<d/2;++z) {
            table[t*d+z]=std::cos(float(t*z)*0.01f); table[t*d+d/2+z]=std::sin(float(t*z)*0.01f);
        }
        run("eg2_norm.kd",EgNorm{y,x,w,6,d,1},6,64);
        run("eg2_rope.kd",EgRope{y,table,3,2,d},(6*d/2+255)/256);
        run("eg2_norm_rope.kd",EgNormRope{x,w,table,6,d,2},6,64);
        for (unsigned i=0;i<6*d;++i) compare(x[i],y[i],2e-6f);
        cases+=2;
    }
    // Attention covers GQA, streaming softmax, two disjoint sequences and SWA.
    auto qb=alloc(8*2048*4), kb=alloc(8*512*4), vb=alloc(8*512*4), bb=alloc(16*4);
    auto *q=static_cast<float *>(qb.data()), *k=static_cast<float *>(kb.data()), *v=static_cast<float *>(vb.data());
    auto *bounds=static_cast<unsigned *>(bb.data());
    for (unsigned t=0;t<8;++t) { bounds[2*t]=t<3?0:3; bounds[2*t+1]=t<3?3:8; }
    for (unsigned i=0;i<8*2048;++i) q[i]=std::sin(float(i)*0.07f)*0.1f;
    for (unsigned i=0;i<8*512;++i) { k[i]=std::cos(float(i)*0.11f)*0.1f; v[i]=std::sin(float(i)*0.03f); }
    for (unsigned d:{256u,512u}) for (unsigned window:{0u,1u}) {
        run(d==256?"eg2_attention256.kd":"eg2_attention512.kd",EgAttention{y,q,k,v,bounds,8,window},8*4,64);
        for (unsigned t=0;t<8;++t) for (unsigned h=0;h<4;++h) {
            unsigned lo=bounds[2*t],hi=bounds[2*t+1],kh=h/(4/(512/d));
            if (window) { if (t>window) lo=std::max(lo,t-window); hi=std::min(hi,t+window+1); }
            double scores[8]={},sum=0;
            for (unsigned s=lo;s<hi;++s) {
                double dot=0; for (unsigned z=0;z<d;++z) dot+=double(q[(t*4+h)*d+z])*k[s*512+kh*d+z];
                scores[s]=std::exp(dot); sum+=scores[s];
            }
            for (unsigned z=0;z<d;++z) {
                double ref=0; for (unsigned s=lo;s<hi;++s) ref+=scores[s]/sum*v[s*512+kh*d+z];
                compare(y[(t*4+h)*d+z],float(ref),2e-5f);
            }
        }
        ++cases;
    }
    aql.reset(); queue.reset(); // Tear down queues before releasing test storage.
    std::printf("{\"test\":\"kernels\",\"queue\":\"%s\",\"cases\":%u,\"failures\":%u,\"max_abs_error\":%.9g}\n",
                argc==3?"aql":"pm4",cases,failures,worst);
    return failures?1:0;
}

catch (const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
