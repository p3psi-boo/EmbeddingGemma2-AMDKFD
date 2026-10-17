#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
target="${1:?usage: deploy-nix.sh SSH_HOST gfx90c|gfx1103}"
arch="${2:?specify gfx90c or gfx1103}"
case "$arch" in gfx90c|gfx1103) ;; *) echo "unsupported architecture: $arch" >&2; exit 2;; esac
nix build --cores "${JOBS:-8}"
nix copy --to "ssh://$target" ./result
package="$(readlink -f result)"
# No daemon, firewall, driver or host configuration changes.
ssh "$target" "timeout 30s $package/bin/kfd-smoke $package/share/embeddinggemma-kfd/eg2-$arch.elf"
ssh "$target" "timeout 60s $package/bin/kfd-kernel-test $package/share/embeddinggemma-kfd/eg2-$arch.elf"
printf 'Installed package: %s\n' "$package"
