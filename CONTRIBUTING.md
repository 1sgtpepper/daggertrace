# Contributing

DaggerTrace is an alpha research checker. This guide applies to contributors with
repository access.

## Before you start

Read the [README](README.md), then use the document that matches the change:

- [Semantic contract](docs/SEMANTICS.md) for admitted language and result meanings;
- [Certificate and replay format](docs/CERTIFICATES.md) for packet structure and integrity;
- [Development](docs/DEVELOPMENT.md) for setup and validation commands;
- [Environment](docs/ENVIRONMENT.md) for the pinned GPU profile; and
- [References and provenance](docs/REFERENCES.md) before changing checked or vendored data.

Keep changes focused. If a change affects checked packets, generated evidence, or an upstream
source snapshot, regenerate the owned artifacts and document what changed. Never commit
credentials, private runtime output, or unreviewed packets.

Use [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for focused validation commands. For GPU
changes, also run the relevant checks in the pinned environment described in
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Issues and pull requests

Use the issue templates for a reproducible bug or a narrowly scoped feature request. Bug reports
need the exact revision, command, expected and observed behavior, and relevant environment
details. Feature requests should state the concrete user need and observable outcome. Scrub
sensitive paths and runtime data before attaching anything.

Pull requests should describe the behavior change, include tests for the changed and opposite
paths, list the validation commands that were run, and update affected documentation or
artifacts. Call out schema, compatibility, dependency, and evidence changes when they apply.
