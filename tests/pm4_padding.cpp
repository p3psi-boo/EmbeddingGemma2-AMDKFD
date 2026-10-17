#include <libkfd/packets/pm4.h>
#include <cstdio>
#include <vector>
#include "../src/pm4_validate.h"

// CPU-only structural regression. Never send deliberately invalid packets to
// a real GPU: decode headers and prove the tail ends exactly at the boundary.
int main() {
    std::vector<unsigned> lengths;
    for (unsigned n=0; n<128; ++n) lengths.push_back(n);
    for (unsigned boundary : {0x4000u,0x8000u,0x10000u,0x40000u})
        for (int delta=-3; delta<=3; ++delta) lengths.push_back(boundary+delta);
    for (unsigned n : lengths) {
        constexpr uint32_t guard=0x12345678;
        std::vector<uint32_t> words(n+2,guard);
        kfd::pm4::nop_fill(words.data()+1,n);
        if (words[0]!=guard || words[n+1]!=guard) return 1;
        unsigned pos=0;
        while (pos<n) {
            uint32_t h=words[pos+1];
            if ((h>>30)!=3 || ((h>>8)&0xff)!=unsigned(kfd::pm4::Opcode::NOP)) return 1;
            unsigned count=(h>>16)&0x3fff;
            unsigned span=count==0x3fff ? 1 : count+2;
            if (span>n-pos) return 1;
            pos+=span;
        }
        if (pos!=n) return 1;
        if (eg2::pm4_invalid_offset({words.data()+1,n})!=n) return 1;
    }
    // Malformed inputs are tested on CPU only.
    const uint32_t truncated[]={kfd::pm4::header(kfd::pm4::Opcode::WRITE_DATA,3),0};
    const uint32_t bad_type[]={0};
    if (eg2::pm4_invalid_offset(truncated)!=0 || eg2::pm4_invalid_offset(bad_type)!=0) return 1;
    std::printf("{\"test\":\"pm4-padding\",\"cases\":%zu,\"failures\":0}\n",lengths.size());
}
