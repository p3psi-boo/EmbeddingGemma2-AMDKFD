# Licensing and provenance

Project-owned code, scripts, kernels, tests, and documentation are available under
WTFPL version 2; see `LICENSE`. The copyright line in that standard license text
identifies the license document's author, not this project's author.

Third-party material is not relicensed by the root license:

| Component | Pinned upstream revision | License |
| --- | --- | --- |
| [libkfd](https://github.com/jhuber6/libkfd) | `dabb4cdeb3bc2354573daa3e3f7afaf7e08ee2e2` | [Apache-2.0](vendor/libkfd/LICENSE) |
| [embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c) | `d8f9e336393e7ba53d47184beac5c9ce9e734311` | [MIT](vendor/embeddinggemma.c/LICENSE) |

Original copyright and license notices remain in the vendored sources.
Modifications to vendored code retain the applicable upstream license and are
listed in [vendor/PATCHES.md](vendor/PATCHES.md). Machine-readable upstream pins
are in [vendor/versions.json](vendor/versions.json).

## Model and toolchain

Model weights are not distributed here. The pinned
[Unsloth model card](https://huggingface.co/unsloth/embeddinggemma-2-GGUF/blob/031f0d4b35536f69ab3509d4893c923264fcf253/README.md)
identifies Apache-2.0 as the model license. Obtain the weights and their terms
from upstream; the root source-code license does not replace those terms.

The Nix lock file pins the build environment. Compiler, system libraries, kernel,
and firmware retain their respective licenses; they are not relicensed by this
repository. Upstream project names indicate provenance, not endorsement.
