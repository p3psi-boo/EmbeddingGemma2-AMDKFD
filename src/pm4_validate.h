#pragma once
#include <cstddef>
#include <cstdint>
#include <span>

namespace eg2 {
// Structural check only: not a proof of opcode/register semantics or GPU safety.
// Return the first malformed header index, or words.size() for an exact parse.
inline size_t pm4_invalid_offset(std::span<const uint32_t> words) {
    size_t pos=0;
    while (pos<words.size()) {
        const uint32_t h=words[pos];
        if ((h>>30)!=3) return pos;
        const unsigned count=(h>>16)&0x3fff, opcode=(h>>8)&0xff;
        const size_t length=(opcode==0x10 && count==0x3fff)?1:count+2;
        if (length>words.size()-pos) return pos;
        pos+=length;
    }
    return pos;
}
}
