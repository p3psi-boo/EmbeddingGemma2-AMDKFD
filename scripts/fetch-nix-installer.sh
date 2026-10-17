#!/usr/bin/env bash
# Download only; execution is delegated to the pinned CI installation action.
set -euo pipefail
output="${1:?usage: fetch-nix-installer.sh OUTPUT_FILE}"
version=2.35.3
hash=69d0fe8fba8fe88f94c98b09964691ac9950abc7ecbf2ba861dc0533cfd901e9
url="https://releases.nixos.org/nix/nix-${version}/install"
if [[ -e "$output" || -L "$output" ]]; then
  printf 'Output already exists: %s\n' "$output" >&2
  exit 1
fi
temporary="$(mktemp "${output}.XXXXXX")"
trap 'rm -f "$temporary"' EXIT
curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$temporary" "$url"
printf '%s  %s\n' "$hash" "$temporary" | sha256sum -c -
chmod 0644 "$temporary"
mv "$temporary" "$output"
