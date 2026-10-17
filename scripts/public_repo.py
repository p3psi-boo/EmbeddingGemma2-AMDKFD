#!/usr/bin/env python3
"""Review a source tree and export working files without Git history or host metadata.

A deterministic hygiene gate, not a comprehensive secret detector. Review the
export manually as well. Findings print file names/rule names, never matched values.
"""
import argparse
import ipaddress
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

ROOT_FILES = {
    '.gitignore', 'CMakeLists.txt', 'README.md', 'LICENSE', 'CONTRIBUTING.md',
    'SECURITY.md', 'THIRD_PARTY_NOTICES.md', 'flake.nix', 'flake.lock',
}
SOURCE_DIRS = {'.github', 'benchmarks', 'docs', 'kernels', 'scripts', 'src', 'tests', 'vendor'}
REQUIRED = {'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
            'vendor/libkfd/LICENSE', 'vendor/embeddinggemma.c/LICENSE'}
PATTERNS = {
    'private-key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'github-token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
    'aws-access-key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'huggingface-token': re.compile(rb'\bhf_[A-Za-z0-9]{30,}\b'),
    'api-key': re.compile(rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b'),
    'credential-url': re.compile(rb'https?://[^\s/:]+:[^\s/@]+@'),
    'personal-path': re.compile(rb'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
    'windows-personal-path': re.compile(rb'[A-Za-z]:\\Users\\[^\s\\]+\\'),
    'nix-store-build-path': re.compile(rb'/nix/store/[a-z0-9]{32}-'),
}
IPV4 = re.compile(rb'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def inventory(root):
    # Includes intended new source files as well as unstaged edits. Ignored local
    # artifacts are absent, but accidentally tracked ignored files are checked.
    names = git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z')
    return sorted({os.fsdecode(p) for p in names.split(b'\0') if p})


def content_findings(data, denied=()):
    rules = [rule for rule, pattern in PATTERNS.items() if pattern.search(data)]
    for match in IPV4.finditer(data):
        try:
            address = ipaddress.ip_address(match.group().decode())
        except ValueError:
            continue
        if not address.is_loopback and not address.is_unspecified:
            rules.append('literal-network-address')
            break
    if any(term.lower() in data.lower() for term in denied):
        rules.append('local-denylist')
    return rules


def inspect(root, names, denied=()):
    findings = [f'{name}: required-file-missing' for name in sorted(REQUIRED - set(names))]
    for name in names:
        findings.extend(f'{name}: {rule}' for rule in content_findings(os.fsencode(name), denied))
        relative = PurePosixPath(name)
        parts = relative.parts
        if (relative.is_absolute() or '..' in parts or not parts or
                (name not in ROOT_FILES and parts[0] not in SOURCE_DIRS)):
            findings.append(f'{name}: unexpected-source-path')
            continue
        if (any(p in {'.git', '__pycache__', '.ssh', '.local', 'node_modules'} or
                p == '.env' or p.startswith('.env.') for p in parts) or
                relative.suffix.lower() in {'.gguf', '.log', '.core', '.pyc', '.pem', '.key', '.gz'}):
            findings.append(f'{name}: local-artifact')
            continue
        path = root / name
        if any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
            findings.append(f'{name}: symlink-directory')
            continue
        if path.is_symlink():
            try:
                target = path.resolve(strict=True).relative_to(root.resolve()).as_posix()
                # Preserve only relative links to another selected source file.
                if os.path.isabs(os.readlink(path)) or target not in names:
                    raise ValueError()
            except (ValueError, OSError, RuntimeError):
                findings.append(f'{name}: unsafe-symlink')
                continue
        if not path.is_file():
            findings.append(f'{name}: missing-or-nonregular-file')
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            findings.append(f'{name}: oversized-source-file')
            continue
        data = path.read_bytes()
        try:
            data.decode('utf-8')
        except UnicodeDecodeError:
            findings.append(f'{name}: non-text-source-file')
            continue
        findings.extend(f'{name}: {rule}' for rule in content_findings(data, denied))
    return findings


def inspect_history(root, denied=()):
    findings = []
    for row in git(root, 'rev-list', '--objects', '--all').splitlines():
        oid = row.split(b' ', 1)[0].decode()
        kind = git(root, 'cat-file', '-t', oid).strip()
        if kind not in {b'blob', b'commit', b'tag'}:
            continue
        data = git(root, 'cat-file', kind.decode(), oid)
        findings.extend(f'history object {oid[:12]}: {rule}'
                        for rule in content_findings(data, denied))
    return findings


def export(root, destination, denied=()):
    names = inventory(root)
    findings = inspect(root, names, denied)
    if findings:
        raise ValueError('\n'.join(findings))
    # No merge/overwrite, .git directory, timestamps, xattrs, or owner metadata.
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name in names:
            source, target = root / name, destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                target.symlink_to(os.readlink(source))
            else:
                target.write_bytes(source.read_bytes())
                target.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
        findings = inspect(destination, names, denied)
        if findings:
            raise ValueError('\n'.join(findings))
    except BaseException:
        shutil.rmtree(destination)
        raise
    return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deny-file', type=Path,
                        help='local-only file with one sensitive literal per line; not copied')
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check')
    check.add_argument('--history', action='store_true', help='also scan reachable Git objects')
    release = commands.add_parser('export')
    release.add_argument('destination', type=Path, help='new directory, without Git metadata')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    denied = tuple(line.strip() for line in args.deny_file.read_bytes().splitlines()
                   if line.strip()) if args.deny_file else ()
    try:
        if args.command == 'export':
            count = export(root, args.destination, denied)
            print(f'Exported {count} reviewed source files; no Git history or repository config.')
        else:
            names = inventory(root)
            findings = inspect(root, names, denied)
            if args.history:
                findings += inspect_history(root, denied)
            if findings:
                raise ValueError('\n'.join(findings))
            print(f'Public-source checks passed: {len(names)} files. Manual review still required.')
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
