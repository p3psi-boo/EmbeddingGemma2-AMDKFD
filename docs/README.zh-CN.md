# EmbeddingGemma 2：直接使用 `/dev/kfd`

面向 AMD APU 的实验性文本 Embedding HTTP Server，兼容 OpenAI embeddings
接口。直接通过 libkfd 管理 GPU 内存与 PM4/AQL 队列，不依赖 ROCm、HIP、ROCr、
rocBLAS、OCML、PyTorch 或 llama.cpp 推理运行时。

## 快速开始

需要 x86-64 Linux、启用 flakes 的 Nix；运行时需要 amdgpu/KFD 和设备访问权限。
以下命令均从仓库根目录执行：

```bash
nix develop -c scripts/build.sh
nix develop -c ctest --test-dir build --output-on-failure
scripts/fetch-model.sh

# gfx1103 上已验证的显式配置；不是库或启动脚本的默认配置
LIBKFD_CWSR_BACKEND=gtt EG2_KFD_QUEUE=pm4 \
EG2_KFD_TILE=step8 EG2_KFD_FUSE=1 EG2_KFD_CHUNK_LAYERS=24 EG2_KFD_PROFILE=0 \
scripts/run.sh
```

```bash
curl http://127.0.0.1:42780/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2","input":["你好","hello world"],"dimensions":768,"encoding_format":"float"}'
```

- 默认仅监听 loopback，没有内置鉴权或 TLS。
- 支持字符串/字符串数组、128/256/512/768 维和 float/base64 输出。
- 单条和 packed wave 上限均为 2048 tokens，含特殊 token；超限返回 400。
- 仅实现固定 Q8_0 模型的文本图，不支持多模态或任意 GGUF。
- 下载脚本固定模型 revision 并校验 SHA-256；模型权重不进入仓库。

## 验证状态

`gfx1103` 的 GTT CWSR + PM4/AQL 均通过 74 项算子测试、HTTP 精度和有限持续
请求测试。测试矩阵内 PM4 最高约 **3167 tokens/s**，来自短文本、每请求 16 条、
HTTP 并发 4 的短时测量；不是生产吞吐保证，也不是同机 CPU 加速比。

`gfx90c` 曾通过早期文本推理验证，但新的 GTT CWSR/AQL 组合尚未实机验证。
其他 GPU 架构不在当前模型后端支持范围。

早期测试曾触发 GPU reset 和 MES 队列移除错误；GTT CWSR 避开了已观察到的
触发路径，但不代表修复 MES 固件。长期稳定性、独立模型基准对齐和稳态性能
仍待完成。公开文档保留这些失败事实，不包含原始主机日志、内存转储或个人路径。

详细配置见 [主 README](../README.md)，方法与数据见 [Benchmarks](benchmarks.md)，
故障证据与边界见 [MES investigation](mes.md)。

自有代码使用 [WTFPL v2](../LICENSE)；第三方代码保留 Apache-2.0/MIT，
详见 [第三方声明](../THIRD_PARTY_NOTICES.md)。
