# Dependency locks and reproducibility boundaries

GitHub Actions and all repository workflows are disabled. Run builds and tests
locally using the locked Nix commands below. Workflow files are not shipped.

## Supported KFD build

The root Nix flake is the supported build path. `flake.lock` fixes nixpkgs to
`b4fd65b198c599cbe814fcb9f42d25d021595ec9` plus its NAR content hash.
`flake.nix` names the same revision rather than a moving branch. That revision
fixes the complete package definitions and transitive source hashes for Clang,
LLVM/lld, CMake, Ninja, pkg-config, libdrm, Python, Git, curl, Gitleaks, Bash,
coreutils, and the host compiler/runtime libraries. Package versions are not
independently selected from system PATH when checks run through `nix develop`.

| Input | Lock |
| --- | --- |
| Optional Nix installer | 2.35.3; installer SHA-256 `69d0fe8fba8fe88f94c98b09964691ac9950abc7ecbf2ba861dc0533cfd901e9` |
| libkfd | Commit in `vendor/versions.json`; modified source included in this repository |
| embeddinggemma.c | Commit in `vendor/versions.json`; modified source included in this repository |
| Primary Q8_0 model | Revision and SHA-256 in `scripts/fetch-model.sh` |

`scripts/fetch-nix-installer.sh` only downloads and verifies the optional Nix
installer; it does not install or upgrade Nix. The verified upstream installer
embeds platform-specific tarball hashes. No workflow executes it automatically.

Run Python, Bash syntax checks, Gitleaks and builds in the locked Nix environment.
Use `--no-update-lock-file` to fail rather than silently rewriting missing/stale
input locks; the deployment build already sets this option.

## Retained optional upstream paths

These files are retained for provenance and optional upstream development; they
are not dependencies of the root KFD package:

- Optional libkfd tests: Catch2 v3.13.0 commit
  `29c9844f688acb27c87338c39cd186ebfe41aa19`, with shallow fetch disabled.
  Root builds disable those upstream GPU/fault tests.
- Upstream XPU dependency script: both dependencies already use full commit IDs.
- Legacy 300M model profile: revision
  `8dd0ca2a66a8f14470acb0e2a71f801afbc5fb73`, not `main`.
  This does not add KFD support for that model or certify its golden results.
- Upstream binary installer: v0.3.1 plus a pinned checksum-manifest SHA-256.
  The manifest verifies each selected platform asset. Release overrides outside
  that lock fail; updating only a version string is insufficient.
## What is not fixed by a source lock

The host kernel, amdgpu driver, APU firmware, hardware and concurrent workloads
remain external. A manual Nix bootstrap initially uses host tools; subsequent
build/test tools come from the pinned flake. The system Nix on a developer
machine is not replaced automatically. GitHub Actions is disabled.

The retained upstream multi-platform release notes describe historical,
non-Nix recipes (including apt-based container builds). Those are not the
supported KFD release procedure and are not claimed to be reproducible. Follow
the root README/flake for this project. Informational documentation links are
not build inputs. User-supplied model/kernel overrides are outside the default
model lock.

## Deliberate updates

1. Change the intended revision/version and its content hash together. Resolve
   annotated tags to commits before using them as source pins.
2. For nixpkgs, change the explicit input revision and deliberately regenerate
   `flake.lock`; review the lock diff. Keep toolchain updates deliberate.
3. Run the offline pin checks and all CPU regressions in the locked environment:

   ```bash
   nix develop --no-update-lock-file -c python3 -m unittest discover -s tests -p 'test_*.py'
   nix build --no-update-lock-file --print-build-logs
   ```

4. For compiler/model/driver changes, rerun relevant manual hardware and parity
   tests before carrying forward performance or stability claims.

The pinning tests enforce workflow absence and cover Nix locks/bootstrap, model download,
optional Catch2/XPU sources, and the retained upstream installer. Download-gate
unit tests reject checksum mismatches and existing/symlink destinations without
using the network or executing installers.
