#include "gemma2_kfd.h"
#include "gpu_target.h"
#include "process_lock.h"
#include "pm4_validate.h"
#include "../kernels/abi.h"
#include <libkfd/libkfd.h>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr unsigned CAP = 2048; // Explicit initial memory/latency bound, not model context size.
constexpr uint64_t TIMEOUT = 30'000'000'000ULL;
template<class T> T take(std::expected<T, kfd::Error> r) {
    if (!r) throw std::runtime_error(kfd::strerror(r));
    return std::move(*r);
}
void check(std::expected<void, kfd::Error> r) {
    if (!r) throw std::runtime_error(kfd::strerror(r));
}
struct Backend {
    // Destruction order matters: buffers/executable/queue before device context.
    eg2::ProcessLock process_lock; // Outlives GPU context teardown.
    kfd::Context context;
    kfd::Device *device = nullptr;
    std::optional<kfd::ComputeQueue> queue;
    std::optional<kfd::AQLQueue> aql_queue;
    std::optional<kfd::Executable> executable;
    std::optional<kfd::Signal> done;
    kfd::Buffer weights, arena, args, fence, timestamps;
    std::unordered_map<std::string, kfd::Kernel> kernels;
    const ei_gguf *model;
    std::mutex mutex;
    bool poisoned = false;
    bool profile = false;
    bool diagnose = false;
    bool tile64 = false;
    bool tile64x64 = false;
    bool step8 = false;
    bool fused = false;
    unsigned chunk_layers = 1;
    std::vector<std::string> timed_kernels;
    std::unordered_map<std::string, double> timings;
    size_t cursor = 0, arg_cursor = 0;
    uint64_t sequence = 0;
    kfd::ComputeQueue::CommandBuffer *command = nullptr;
    kfd::AQLQueue::CommandBuffer *aql_command = nullptr;
    static constexpr kfd::aql::Ordering aql_order{
        true, kfd::aql::FENCE_SCOPE_SYSTEM, kfd::aql::FENCE_SCOPE_SYSTEM};

    kfd::Buffer alloc(size_t bytes, bool coherent = false) {
        auto flags = kfd::MemFlags::WRITABLE;
        if (coherent) flags = flags | kfd::MemFlags::COHERENT | kfd::MemFlags::UNCACHED;
        auto b = take(kfd::Buffer::allocate(*device, bytes, kfd::MemType::GTT, flags));
        check(b.map(*device));
        return b;
    }
    template<class T> T *work(size_t n) {
        cursor = (cursor + 63) & ~size_t(63);
        if (n > (arena.size() - cursor) / sizeof(T)) throw std::runtime_error("KFD workspace exhausted");
        auto *p = reinterpret_cast<T *>(static_cast<std::byte *>(arena.data()) + cursor);
        cursor += n * sizeof(T);
        return p;
    }
    const ei_tensor *tensor(const std::string &name) const {
        auto *t = ei_gguf_tensor(model, name.c_str(), false);
        if (!t) throw std::runtime_error("missing tensor: " + name);
        return t;
    }
    const void *data(const ei_tensor *t) const {
        return static_cast<const std::byte *>(weights.data()) + t->offset;
    }
    const float *norm_weight(const std::string &name) const {
        const auto *t = tensor(name);
        if (t->type != EI_T_F32) throw std::runtime_error("norm requires F32: " + name);
        return static_cast<const float *>(data(t));
    }
    explicit Backend(const ei_gguf *g) : context(take(kfd::Context::create())), model(g) {
        device = eg2::select_device(context);
        if (!device) throw std::runtime_error("supported KFD device not found (gfx90c/gfx1103)");
        const char *queue_kind=std::getenv("EG2_KFD_QUEUE");
        if (queue_kind && std::strcmp(queue_kind,"pm4") && std::strcmp(queue_kind,"aql"))
            throw std::runtime_error("EG2_KFD_QUEUE must be pm4 or aql");
        bool use_aql=queue_kind && !std::strcmp(queue_kind,"aql");
        queue.emplace(take(kfd::ComputeQueue::create(*device, 1u << 20)));
        std::filesystem::path path;
        std::string object=std::string("eg2-")+eg2::target_name(device->gfx_version())+".elf";
        if (const char *env = std::getenv("EG2_KFD_KERNEL")) path = env;
        else {
            auto bin = std::filesystem::canonical("/proc/self/exe").parent_path();
            path = bin / object;
            if (!std::filesystem::exists(path)) path = bin / "../share/embeddinggemma-kfd" / object;
        }
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() <= 0) throw std::runtime_error("missing KFD kernel code object: " + path.string());
        std::vector<std::byte> image(static_cast<size_t>(f.tellg()));
        f.seekg(0);
        if (!f.read(reinterpret_cast<char *>(image.data()), image.size())) throw std::runtime_error("short kernel read");
        executable.emplace(take(kfd::Executable::load(*device, image, *queue)));
        for (const char *name : {"gather", "mm", "mm64", "mm64x64", "mmstep8", "norm_residual", "norm_rope", "norm", "rope", "attention256", "attention512", "element", "pool"})
            kernels.emplace(name, take(executable->kernel(std::string("eg2_") + name + ".kd")));
        // Keep GGUF storage quantized. No 500+ MB full dequantized duplicate.
        weights = alloc(g->map_len - g->data_off);
        // Buffer::size() is page-rounded; the mmap source has only logical bytes.
        std::memcpy(weights.data(), g->map + g->data_off, g->map_len - g->data_off);
        // Max CAP packed tokens, ~190 MiB including PLE. Reused across batches.
        arena = alloc(size_t(CAP) * (512*5 + 2048*4 + 512*24 + 768*2) * 4 + (1u<<20));
        args = alloc(512 * 1024, true);
        fence = alloc(64, true);
        auto flag = [](const char *name) {
            const char *v=std::getenv(name);
            if (v && std::strcmp(v,"0") && std::strcmp(v,"1")) throw std::runtime_error(std::string(name)+" must be 0 or 1");
            return v && !std::strcmp(v,"1");
        };
        profile = flag("EG2_KFD_PROFILE");
        if (use_aql && profile) throw std::runtime_error("AQL requires EG2_KFD_PROFILE=0; use wall-clock measurements");
        diagnose = flag("EG2_KFD_DIAGNOSE");
        fused = flag("EG2_KFD_FUSE");
        if (const char *v=std::getenv("EG2_KFD_CHUNK_LAYERS")) {
            char *end=nullptr; long value=std::strtol(v,&end,10);
            if (!*v || *end || value<1 || value>24) throw std::runtime_error("EG2_KFD_CHUNK_LAYERS must be 1..24");
            chunk_layers=unsigned(value);
        }
        const char *tile = std::getenv("EG2_KFD_TILE");
        if (tile && std::strcmp(tile,"16") && std::strcmp(tile,"64") && std::strcmp(tile,"64x64") && std::strcmp(tile,"step8"))
            throw std::runtime_error("EG2_KFD_TILE must be 16, 64, 64x64 or step8");
        step8 = tile && !std::strcmp(tile,"step8");
        tile64x64 = step8 || (tile && !std::strcmp(tile,"64x64"));
        tile64 = tile64x64 || (tile && !std::strcmp(tile,"64"));
        if (profile) timestamps = alloc(1024*2*sizeof(uint64_t), true);
        *static_cast<uint64_t *>(fence.data()) = 0;
        done.emplace(take(kfd::Signal::create(context)));
        if (use_aql) {
            // Executable loader has waited for its copy and I-cache invalidate.
            // All working storage is ready; never overlap the two GPU queues.
            queue.reset();
            aql_queue.emplace(take(kfd::AQLQueue::create(*device, 1u<<20)));
        }
        if (diagnose) {
            auto s=aql_queue?aql_queue->snapshot():queue->snapshot();
            std::fprintf(stderr,"KFD-DIAG cwsr=%p bytes=%zu svm=%d ring_dwords=%zu\n",
                         s.cwsr_address,s.cwsr_bytes,s.cwsr_svm,size_t(1u<<18));
        }
        std::fprintf(stderr,"KFD: queue=%s\n",use_aql?"aql":"pm4");
        std::fprintf(stderr, "KFD: %s (wave%u), quantized weights %.1f MiB, workspace %.1f MiB, cap %u packed tokens\n",
                     eg2::target_name(device->gfx_version()), eg2::wave_size(*device), weights.size()/1048576.0, arena.size()/1048576.0, CAP);
    }
    template<class T> void launch(const char *name, const T &p, unsigned groups, unsigned threads = 256, unsigned gy = 1) {
        // The graph uses 64 to denote one-wave reductions; RDNA uses wave32.
        if (threads==64) threads=eg2::wave_size(*device);
        auto &kernel = kernels.at(name);
        size_t bytes = (kernel.kernarg_size() + 63) & ~size_t(63);
        if (arg_cursor + bytes > args.size()) throw std::runtime_error("kernarg arena exhausted");
        auto region = std::span(static_cast<std::byte *>(args.data()) + arg_cursor, bytes);
        arg_cursor += bytes;
        kfd::DispatchConfig cfg{.grid={.x=groups, .y=gy}, .block={.x=threads}};
        kernel.fill(region, p, cfg);
        if (aql_command) {
            // BARRIER orders dependent dispatches; system scopes cover both
            // inter-kernel data and CPU-written inputs / CPU-read output.
            aql_command->dispatch(kernel,cfg,region,aql_order);
            ++sequence;
            return;
        }
        uint64_t *clock = nullptr;
        if (profile) {
            if (timed_kernels.size() >= 1024) throw std::runtime_error("profile arena exhausted");
            clock = static_cast<uint64_t *>(timestamps.data()) + timed_kernels.size()*2;
            timed_kernels.emplace_back(name);
            command->read_gpu_clock(clock);
        }
        // A true completion dependency, not merely cache invalidation. Keep each
        // kernarg alive and untouched until the entire chunk has completed.
        command->dispatch(kernel, cfg, region)
            .release_mem(kfd::pm4::REL_ALL, fence.data(), ++sequence)
            .wait_reg_mem(fence.data(), kfd::Condition::EQ, sequence)
            .acquire_mem();
        if (clock) command->read_gpu_clock(clock+1);
    }
    void flush() {
        if (aql_command) { flush_aql(); return; }
        check(done->reset());
        poisoned = true;
        command->signal(*done);
        if (diagnose && eg2::pm4_invalid_offset(command->data())!=command->size())
            throw std::runtime_error("malformed PM4 packet stream before submission");
        auto snapshot = [&](const char *stage) {
            if (!diagnose) return;
            auto s=queue->snapshot();
            auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            auto reached=__atomic_load_n(static_cast<uint64_t *>(fence.data()),__ATOMIC_ACQUIRE);
            std::fprintf(stderr,"KFD-DIAG ns=%lld stage=%s dwords=%zu args=%zu target=%llu reached=%llu rp=%llu wp=%llu err=%llu\n",
                         (long long)ns,stage,command->size(),arg_cursor,
                         (unsigned long long)sequence,(unsigned long long)reached,
                         (unsigned long long)s.read_ptr,(unsigned long long)s.write_ptr,
                         (unsigned long long)s.error_payload);
        };
        snapshot("before");
        auto submitted=command->submit(TIMEOUT);
        snapshot(submitted?"submitted":"submit-error");
        check(submitted);
        // After any failure, do not reuse buffers that could still be in flight.
        auto r = done->wait(kfd::Condition::EQ, 0, TIMEOUT);
        snapshot(r?"complete":"wait-error");
        if (!r) { poisoned = true; check(r); }
        poisoned = false;
        if (profile) {
            auto *clock = static_cast<uint64_t *>(timestamps.data());
            double hz = double(device->drm_properties().timestamp_freq);
            for (size_t i=0; i<timed_kernels.size(); ++i)
                timings[timed_kernels[i]] += double(clock[2*i+1]-clock[2*i])/hz*1000.0;
            timed_kernels.clear();
        }
        command->reset();
        arg_cursor = 0;
    }
    void flush_aql() {
        check(done->reset());
        poisoned=true;
        aql_command->barrier({},aql_order,&*done);
        auto snapshot=[&](const char *stage) {
            if (!diagnose) return;
            auto s=aql_queue->snapshot();
            std::fprintf(stderr,"AQL-DIAG stage=%s packets=%zu kernels=%llu rp=%llu wp=%llu err=%llu\n",
                stage,aql_command->size(),(unsigned long long)sequence,
                (unsigned long long)s.read_ptr,(unsigned long long)s.write_ptr,
                (unsigned long long)s.error_payload);
        };
        snapshot("before");
        check(aql_command->submit(TIMEOUT));
        snapshot("submitted");
        auto r=done->wait(kfd::Condition::EQ,0,TIMEOUT);
        snapshot(r?"complete":"wait-error");
        check(r);
        poisoned=false;
        aql_command->reset();
        arg_cursor=0;
    }
    void mm(const std::string &name, const float *x, unsigned n, float *out) {
        auto *w = tensor(name);
        if (w->type != EI_T_F32 && w->type != EI_T_Q8_0 && w->type != EI_T_BF16)
            throw std::runtime_error("unsupported KFD tensor type");
        unsigned k = unsigned(w->ne[0]), m = unsigned(w->ne[1]);
        unsigned width=tile64?64:16;
        bool large=tile64x64 && n>=64;
        unsigned height=large?64:16;
        launch(large?(step8?"mmstep8":"mm64x64"):(tile64?"mm64":"mm"), EgMM{out,x,data(w),n,k,m,w->type}, (m+width-1)/width, 256, (n+height-1)/height);
    }
    void norm(float *out, const float *x, unsigned rows, unsigned d, const float *w, float scale=1) {
        launch("norm", EgNorm{out,x,w,rows,d,scale}, rows, 64);
    }
    void element(float *out, const float *x, const float *other, unsigned n, unsigned mode=0, unsigned layer=0, float scale=1) {
        launch("element", EgElement{out,x,other,n,mode,layer,scale}, (n+255)/256);
    }
    void residual_norm(float *residual, float *x, unsigned n, const float *w, float scale=1) {
        if (fused) launch("norm_residual", EgNormResidual{residual,x,w,n,512,scale},n,64);
        else { norm(x,x,n,512,w); element(residual,residual,x,n*512,0,0,scale); }
    }
    void batch(const int32_t *ids, const size_t *offsets, size_t batch, float *out) {
        std::lock_guard guard(mutex); // Single GPU executor; never race shared workspaces.
        if (poisoned) throw std::runtime_error("KFD context failed; restart process before reuse");
        if (!ids || !offsets || !out || !batch || batch > CAP || offsets[0] || offsets[batch] > CAP)
            throw std::runtime_error("KFD batch requires 1..2048 packed tokens and nonempty sequences");
        for (size_t s=0; s<batch; ++s) if (offsets[s+1] <= offsets[s]) throw std::runtime_error("invalid KFD sequence offsets");
        unsigned n = unsigned(offsets[batch]);
        if (diagnose) std::fprintf(stderr,"KFD-DIAG batch_tokens=%u batch_sequences=%zu\n",n,batch);
        auto start = std::chrono::steady_clock::now();
        timings.clear();
        for (unsigned i=0; i<n; ++i) if (ids[i]<0 || ids[i]>=262144) throw std::runtime_error("token ID out of range");
        cursor=0; arg_cursor=0;
        float *x=work<float>(n*512), *h=work<float>(n*512), *q=work<float>(n*2048),
              *k=work<float>(n*512), *v=work<float>(n*512), *a=work<float>(n*2048),
              *up=work<float>(n*2048), *gate=work<float>(n*2048), *ple=work<float>(n*512*24),
              *rl=work<float>(n*256), *rg=work<float>(n*512), *pooled=work<float>(batch*768);
        auto *tokens=work<int32_t>(n); auto *bounds=work<unsigned>(2*n); auto *off=work<unsigned>(batch+1);
        std::memcpy(tokens, ids, n*sizeof(*ids));
        for (size_t s=0; s<=batch; ++s) off[s]=unsigned(offsets[s]);
        for (size_t s=0; s<batch; ++s) for (size_t t=offsets[s]; t<offsets[s+1]; ++t) {
            bounds[2*t]=off[s]; bounds[2*t+1]=off[s+1];
            for (unsigned d : {256u,512u}) {
                float *r=d==256 ? rl : rg; float base=d==256 ? 10000.0f : 1000000.0f;
                for (unsigned z=0; z<d/2; ++z) {
                    float theta=float(t-offsets[s])*std::pow(base,-2.0f*float(z)/float(d));
                    r[t*d+z]=std::cos(theta); r[t*d+d/2+z]=std::sin(theta);
                }
            }
        }
        std::optional<kfd::ComputeQueue::CommandBuffer> cmd;
        std::optional<kfd::AQLQueue::CommandBuffer> aqcmd;
        if (aql_queue) {
            aqcmd.emplace(aql_queue->command()); aql_command=&*aqcmd;
        } else {
            cmd.emplace(queue->command()); command=&*cmd;
            cmd->acquire_mem();
        }
        launch("gather", EgGather{x,data(tensor("token_embd.weight")),tokens,n}, (n*512+255)/256);
        mm("per_layer_model_proj.weight",x,n,ple);
        norm(ple,ple,n*24,512,norm_weight("per_layer_proj_norm.weight"),1.0f/std::sqrt(512.0f));
        if (chunk_layers==1) flush();
        for (unsigned l=0; l<24; ++l) {
            std::string prefix="blk."+std::to_string(l)+".";
            auto w=[&](const char *name) { return prefix+name+".weight"; };
            unsigned d=l%6==5 ? 512 : 256, kv=512/d;
            norm(h,x,n,512,norm_weight(w("attn_norm")));
            mm(w("attn_q"),h,n,q); mm(w("attn_k"),h,n,k); mm(w("attn_v"),h,n,v);
            if (fused) {
                launch("norm_rope", EgNormRope{q,norm_weight(w("attn_q_norm")),d==256?rl:rg,n*4,d,4},n*4,64);
                launch("norm_rope", EgNormRope{k,norm_weight(w("attn_k_norm")),d==256?rl:rg,n*kv,d,kv},n*kv,64);
                norm(v,v,n*kv,d,nullptr);
            } else {
                norm(q,q,n*4,d,norm_weight(w("attn_q_norm")));
                norm(k,k,n*kv,d,norm_weight(w("attn_k_norm"))); norm(v,v,n*kv,d,nullptr);
                launch("rope",EgRope{q,d==256?rl:rg,n,4,d},(n*4*(d/2)+255)/256);
                launch("rope",EgRope{k,d==256?rl:rg,n,kv,d},(n*kv*(d/2)+255)/256);
            }
            launch(d==256?"attention256":"attention512",EgAttention{a,q,k,v,bounds,n,d==256?512u:0u},n*4,64);
            mm(w("attn_output"),a,n,h); residual_norm(x,h,n,norm_weight(w("post_attention_norm")));
            norm(h,x,n,512,norm_weight(w("ffn_norm")));
            mm(w("ffn_up"),h,n,up); mm(w("ffn_gate"),h,n,gate);
            element(gate,gate,up,n*2048,1); mm(w("ffn_down"),gate,n,h);
            residual_norm(x,h,n,norm_weight(w("post_ffw_norm")));
            mm(w("inp_gate"),x,n,h); element(h,h,ple,n*512,2,l); mm(w("proj"),h,n,a);
            residual_norm(x,a,n,norm_weight(w("post_norm")),*norm_weight(w("layer_output_scale")));
            if ((l+1)%chunk_layers==0 && (chunk_layers==1 || l+1<24)) flush();
        }
        norm(x,x,n,512,norm_weight("output_norm.weight")); mm("output.weight",x,n,a);
        launch("pool",EgPool{pooled,a,off,unsigned(batch)},(unsigned(batch)*768+255)/256); flush();
        for (size_t s=0; s<batch; ++s) {
            double ss=0;
            for (unsigned z=0; z<768; ++z) { float f=pooled[s*768+z]; if (!std::isfinite(f)) throw std::runtime_error("nonfinite GPU output"); ss+=double(f)*f; }
            if (ss<=0) throw std::runtime_error("zero GPU embedding");
            float inv=float(1/std::sqrt(ss));
            for (unsigned z=0; z<768; ++z) out[s*768+z]=pooled[s*768+z]*inv;
        }
        command=nullptr;
        aql_command=nullptr;
        if (profile) {
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            std::fprintf(stderr,"KFD_PROFILE {\"tokens\":%u,\"batch\":%zu,\"wall_ms\":%.3f,\"command_ms\":{",n,batch,ms);
            bool first=true;
            for (const auto &[name,value]:timings) { std::fprintf(stderr,"%s\"%s\":%.3f",first?"":",",name.c_str(),value); first=false; }
            std::fprintf(stderr,"}}\n");
        }
    }
    ~Backend() { aql_queue.reset(); queue.reset(); } // Queues before GPU-visible storage.
};
}

extern "C" void *ei_gemma2_kfd_create(const ei_gguf *g, char *err, size_t len) {
    try { return new Backend(g); }
    catch (const std::exception &e) { std::snprintf(err,len,"KFD: %s",e.what()); return nullptr; }
}
extern "C" void ei_gemma2_kfd_free(void *ctx) { delete static_cast<Backend *>(ctx); }
extern "C" bool ei_gemma2_kfd_batch(void *ctx, const int32_t *ids, const size_t *offsets, size_t batch, float *out, char *err, size_t len) {
    try { if (!ctx) throw std::runtime_error("empty context"); static_cast<Backend *>(ctx)->batch(ids,offsets,batch,out); return true; }
    catch (const std::exception &e) { std::snprintf(err,len,"KFD: %s",e.what()); return false; }
}
