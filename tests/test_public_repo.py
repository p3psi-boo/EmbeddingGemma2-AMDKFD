"""CPU-only regression tests for publication boundaries; no external packages."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('public_repo', Path(__file__).parents[1] / 'scripts/public_repo.py')
public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public)


class PublicRepoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'source'
        self.root.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        for name in public.REQUIRED:
            self.write(name, 'example source\n')
        self.write('.gitignore', '/results/\n/.local/\n')
        self.destination = Path(self.tmp.name) / 'release'

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        return path

    def findings(self):
        return public.inspect(self.root, public.inventory(self.root))

    def test_export_excludes_private_artifacts_and_git(self):
        self.write('results/raw.txt', 'private diagnostics')
        self.write('.local/config', 'private config')
        script = self.write('scripts/example.sh', '#!/bin/sh\nexit 0\n')
        script.chmod(0o755)
        public.export(self.root, self.destination)
        self.assertFalse((self.destination / '.git').exists())
        self.assertFalse((self.destination / 'results').exists())
        self.assertFalse((self.destination / '.local').exists())
        copied = self.destination / 'scripts/example.sh'
        self.assertEqual(copied.read_bytes(), script.read_bytes())
        self.assertEqual(copied.stat().st_mode & 0o777, 0o755)

    def test_existing_destination_is_never_overwritten(self):
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            public.export(self.root, self.destination)
        self.assertTrue(self.destination.exists())

    def test_tracked_ignored_artifact_fails(self):
        self.write('results/raw.txt', 'private')
        subprocess.run(['git', '-C', str(self.root), 'add', '-f', 'results/raw.txt'], check=True)
        self.assertIn('results/raw.txt: unexpected-source-path', self.findings())

    def test_secret_failure_does_not_echo_value(self):
        secret = 'ghp_' + 'A' * 36
        self.write('docs/notes.md', secret)
        with self.assertRaises(ValueError) as error:
            public.export(self.root, self.destination)
        self.assertIn('github-token', str(error.exception))
        self.assertNotIn(secret, str(error.exception))
        self.assertFalse(self.destination.exists())

    def test_local_paths_and_network_addresses(self):
        for value in ('/' + 'home/person/project/file', '/' + 'Users/person/project/file',
                      '.'.join(['192', '168', '3', '4'])):
            self.assertTrue(public.content_findings(value.encode()))
        self.assertEqual(public.content_findings(b'http://127.0.0.1:42780'), [])

    def test_local_denylist(self):
        self.write('README.md', 'sensitive-host-alias')
        with self.assertRaises(ValueError):
            public.export(self.root, self.destination, (b'sensitive-host-alias',))

    def test_sensitive_filename_fails(self):
        self.write('docs/sensitive-host-alias.md', 'otherwise clean')
        with self.assertRaises(ValueError):
            public.export(self.root, self.destination, (b'sensitive-host-alias',))

    def test_external_and_absolute_symlinks_fail(self):
        outside = Path(self.tmp.name) / 'private.txt'
        outside.write_text('private')
        link = self.root / 'docs/link.md'
        link.parent.mkdir()
        for target in (str(outside), '../../private.txt'):
            link.symlink_to(target)
            self.assertIn('docs/link.md: unsafe-symlink', self.findings())
            link.unlink()

    def test_internal_symlink_preserved(self):
        self.write('docs/one.md', 'source')
        (self.root / 'docs/two.md').symlink_to('one.md')
        public.export(self.root, self.destination)
        self.assertTrue((self.destination / 'docs/two.md').is_symlink())
        self.assertEqual((self.destination / 'docs/two.md').read_text(), 'source')

    def test_symlink_directory_fails(self):
        self.write('docs/one.md', 'source')
        subprocess.run(['git', '-C', str(self.root), 'add', 'docs/one.md'], check=True)
        (self.root / 'docs/one.md').unlink()
        (self.root / 'docs').rmdir()
        outside = Path(self.tmp.name) / 'private'
        outside.mkdir()
        (outside / 'one.md').write_text('private')
        (self.root / 'docs').symlink_to(outside)
        self.assertIn('docs/one.md: symlink-directory', self.findings())

    def test_missing_license_fails(self):
        (self.root / 'LICENSE').unlink()
        self.assertIn('LICENSE: required-file-missing', self.findings())

    def test_history_detects_removed_sensitive_content(self):
        value = '/' + 'home/person/project/file'
        self.write('README.md', value)
        subprocess.run(['git', '-C', str(self.root), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)
        self.write('README.md', 'clean source')
        self.assertEqual(self.findings(), [])
        self.assertTrue(public.inspect_history(self.root))


if __name__ == '__main__':
    unittest.main()
