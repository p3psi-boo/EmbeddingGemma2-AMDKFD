"""Offline pinning policy checks; no network, GPU or model is needed."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT / name).read_text()


class DependencyPinTests(unittest.TestCase):
    def test_flake_inputs_are_content_locked(self):
        lock = json.loads(read('flake.lock'))
        for name, node in lock['nodes'].items():
            if name == 'root':
                continue
            self.assertRegex(node['locked']['rev'], r'^[0-9a-f]{40}$')
            self.assertRegex(node['locked']['narHash'], r'^sha256-[A-Za-z0-9+/]{43}=$')
            self.assertEqual(node['original']['rev'], node['locked']['rev'])
        self.assertIn(lock['nodes']['nixpkgs']['locked']['rev'], read('flake.nix'))

    def test_actions_use_full_object_pins(self):
        workflows = [ROOT / '.github/workflows/ci.yml',
                     ROOT / 'vendor/embeddinggemma.c/.github/workflows/ci.yml']
        for path in workflows:
            source = path.read_text()
            for action in re.findall(r'uses:\s*([^\s#]+)', source):
                self.assertRegex(action, r'^[\w.-]+/[\w.-]+@[0-9a-f]{40}$')
            self.assertNotIn('ubuntu-latest', source)
        # The install action's former SHA was an annotated tag object, not a commit.
        self.assertIn('install-nix-action@448cde739abec1eaf0646249e3d25bb8d9bc7702', workflows[0].read_text())

    def test_ci_uses_locked_tools_after_bootstrap(self):
        source = read('.github/workflows/ci.yml')
        self.assertIn('install_url: file://${{ runner.temp }}/nix-install', source)
        self.assertIn('scripts/fetch-nix-installer.sh', source)
        self.assertNotRegex(source, r'run:\s*python3\b')
        self.assertNotIn('nix develop -c', source)
        self.assertIn('nix build --no-update-lock-file', source)
        self.assertIn('nix develop --no-update-lock-file -c bash', source)
        self.assertIn('pkgs.bash pkgs.coreutils', read('flake.nix'))
        self.assertIn('--no-update-lock-file', read('scripts/deploy-nix.sh'))

    def test_nix_installer_has_version_and_digest(self):
        source = read('scripts/fetch-nix-installer.sh')
        version = re.search(r'^version=(\d+\.\d+\.\d+)$', source, re.M).group(1)
        self.assertRegex(source, r'(?m)^hash=[0-9a-f]{64}$')
        self.assertIn('sha256sum -c -', source)
        self.assertIn(f"nix (Nix) {version}", read('.github/workflows/ci.yml'))
        self.assertNotIn('releases/latest', source)

    def test_model_download_has_revision_and_digest(self):
        source = read('scripts/fetch-model.sh')
        self.assertRegex(source, r'/resolve/[0-9a-f]{40}/')
        self.assertRegex(source, r'(?m)^hash=[0-9a-f]{64}$')
        self.assertIn('sha256sum -c -', source)
        self.assertNotIn('/resolve/main/', read('vendor/embeddinggemma.c/src/profiles.h'))
        self.assertNotIn('/resolve/main/', read('vendor/embeddinggemma.c/PORT_SPEC.md'))

    def test_vendor_source_revisions(self):
        for value in json.loads(read('vendor/versions.json')).values():
            self.assertRegex(value['commit'], r'^[0-9a-f]{40}$')
        xpu = read('vendor/embeddinggemma.c/scripts/fetch-xpu-deps.sh')
        self.assertRegex(xpu, r'(?m)^vllm_commit=[0-9a-f]{40}$')
        self.assertRegex(xpu, r'(?m)^cutlass_commit=[0-9a-f]{40}$')

    def test_catch2_fetch_is_commit_pinned(self):
        source = read('vendor/libkfd/tests/CMakeLists.txt')
        self.assertRegex(source, r'GIT_TAG\s+[0-9a-f]{40}\b')
        self.assertRegex(source, r'GIT_SHALLOW\s+FALSE')

    def test_retained_upstream_installer_is_pinned(self):
        source = read('vendor/embeddinggemma.c/install.sh')
        self.assertIn('version=${EMBEDDINGGEMMA_VERSION:-v0.3.1}', source)
        self.assertRegex(source, r'(?m)^checksums_hash=[0-9a-f]{64}$')
        self.assertIn('[ "$version" = v0.3.1 ] || die', source)
        self.assertIn('[ "$(sha256_file "$checksums")" = "$checksums_hash" ]', source)
        self.assertNotIn('/releases/latest/', source)


class InstallerDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b'installer fixture\n'
        source = read('scripts/fetch-nix-installer.sh')
        # Exercise the real gate with a tiny fixture instead of a network fetch.
        source = re.sub(r'(?m)^hash=[0-9a-f]{64}$',
                        'hash=' + hashlib.sha256(self.payload).hexdigest(), source)
        self.script = self.root / 'fetch.sh'
        self.script.write_text(source)
        mock = self.root / 'curl'
        mock.write_text('''#!/bin/sh
while [ "$#" -gt 0 ]; do
  if [ "$1" = -o ]; then output=$2; shift; fi
  shift
done
printf '%s' "$FIXTURE_PAYLOAD" > "$output"
''')
        mock.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ['PATH'],
                        FIXTURE_PAYLOAD=self.payload.decode())
        self.output = self.root / 'installer'

    def run_fetch(self):
        return subprocess.run(['bash', str(self.script), str(self.output)],
                              env=self.env, capture_output=True, text=True)

    def test_verified_download(self):
        self.assertEqual(self.run_fetch().returncode, 0)
        self.assertEqual(self.output.read_bytes(), self.payload)
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o644)

    def test_mismatch_does_not_publish_file(self):
        self.env['FIXTURE_PAYLOAD'] = 'corrupt bytes'
        self.assertNotEqual(self.run_fetch().returncode, 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('installer.*')), [])

    def test_existing_destination_is_not_overwritten(self):
        self.output.write_text('keep')
        self.assertNotEqual(self.run_fetch().returncode, 0)
        self.assertEqual(self.output.read_text(), 'keep')

    def test_symlink_destination_is_not_followed(self):
        target = self.root / 'other'
        target.write_text('keep')
        self.output.symlink_to(target)
        self.assertNotEqual(self.run_fetch().returncode, 0)
        self.assertEqual(target.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
