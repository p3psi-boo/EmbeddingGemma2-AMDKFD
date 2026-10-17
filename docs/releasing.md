# Preparing a public source release

## Separate source, evidence, and local state

- Public: source, pinned dependencies, licenses, reviewed documentation, numeric
  benchmark samples and anonymized hardware characteristics.
- Local only: `results/`, `.local/`, `build/`, `dist/`, model weights, credentials,
  raw kernel logs, traces, smaps, device dumps, and deployment details.

Ignoring or deleting a file does not remove its previous Git revisions. Before a
first publication, check both the working source tree and existing history:

```bash
python3 scripts/public_repo.py check
python3 scripts/public_repo.py check --history
```

The optional `--deny-file` global argument accepts a local-only newline-separated
list of private names, addresses, usernames or sensitive literals. Store it in
`.local/`; the tool prints rule names rather than matching values. The scan is a
heuristic hygiene gate, not a guarantee that every kind of secret is detected.
Review the final source diff and run an independent secret scanner as well.
The pinned development shell includes [Gitleaks](https://github.com/gitleaks/gitleaks):

```bash
nix develop -c gitleaks dir --redact --no-banner dist/public-source
```

## History-free export

When development history contains private artifacts, preserve that repository
locally and export a new source directory instead of pushing its history:

```bash
python3 scripts/public_repo.py export dist/public-source
```

The destination must not exist. Export includes current working versions of
tracked files plus non-ignored new source files. It rejects unexpected source
locations, common secret formats, personal absolute paths, literal network
addresses other than loopback/wildcard, unsafe symlinks and common artifact types.
It copies neither Git objects/configuration/remotes nor original file metadata.
Internal relative source symlinks and executable permissions are preserved.

For a personal denylist:

```bash
python3 scripts/public_repo.py --deny-file .local/public-denylist.txt export dist/public-source
```

Inspect the export, then initialize a separate repository there. Choose the
public commit author name/email explicitly; do not inherit private identity or
copy the development `.git` directory. Stage all source files before building a
Nix flake from a new Git repository, because Nix's Git source excludes untracked
files. Alternatively build the exported directory as a path flake before Git init.

## Release checklist

- [ ] Project license chosen; vendor license texts and downstream changes retained.
- [ ] README accurately describes tested hardware, API, defaults and open limitations.
- [ ] No secrets, personal directories, SSH host aliases or host inventory in source.
- [ ] History is reviewed, or publication starts from a verified history-free export.
- [ ] Public author/committer identity and any signed-tag identity reviewed separately.
- [ ] CPU tests and clean exported-source build pass; GPU claims match recorded evidence.
- [ ] Package includes project and third-party license texts.
- [ ] Model weights and private diagnostics absent from source and release attachments.
- [ ] Review CI logs/artifacts before publishing them; do not upload raw diagnostics.
- [ ] Configure private vulnerability reporting on the hosting provider if available.

Creating an export does not create a remote, commit, tag, push, GitHub repository,
or public release. Those are separate publication steps.
