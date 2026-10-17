#!/usr/bin/env python3
"""Offline summary of one diagnostic capture; never opens a GPU device."""
import json
from pathlib import Path
import re
import sys


def main():
    directory=Path(sys.argv[1])
    log=(directory/"layer.log").read_text()
    metadata=re.search(r"cwsr=(0x[0-9a-f]+) bytes=(\d+) svm=(\d+) ring_dwords=(\d+)", log)
    address, size, svm, capacity=metadata.groups()
    capacity=int(capacity)
    rows=[dict(re.findall(r"(\w+)=([^ ]+)", line)) for line in log.splitlines() if " stage=" in line]
    submissions=[r for r in rows if r["stage"]=="submitted"]
    completed=[r for r in rows if r["stage"]=="complete"]
    before=[r for r in rows if r["stage"]=="before"]
    last=submissions[-1]
    wraps=sum(int(b["wp"])%capacity+int(s["dwords"])>capacity for b,s in zip(before,submissions))
    smaps=(directory/"layer-smaps.txt").read_text()
    mapping=""
    for block in re.split(r"(?=^[0-9a-f]+-[0-9a-f]+ )",smaps,flags=re.M):
        m=re.match(r"([0-9a-f]+)-([0-9a-f]+)",block)
        if m and int(m[1],16)<=int(address,16)<int(m[2],16):
            mapping=block
            break
    (directory/"cwsr-smaps.txt").write_text(mapping)
    result=dict(submissions=len(submissions), completed=len(completed), ring_wraps=wraps,
                bad_completed_fences=sum(r["target"]!=r["reached"] for r in completed),
                nonzero_error_snapshots=sum(int(r["err"])!=0 for r in rows),
                last_submission=last,
                last_submission_wraps=int(before[-1]["wp"])%capacity+int(last["dwords"])>capacity,
                cwsr=dict(address=address, bytes=int(size), svm=bool(int(svm))),
                limitations=["Snapshots are non-transactional; RP may lag the completion signal.",
                             "Packet validation checks lengths/types, not register/opcode semantics.",
                             "Ftrace has stacks but not MMU notifier range arguments."])
    (directory/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
