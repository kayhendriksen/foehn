# Security Policy

## Supported Versions

foehn is currently pre-1.0. Security fixes are provided for the latest published release.

## Reporting a Vulnerability

Please do not open a public issue for suspected security vulnerabilities.

Use GitHub's private vulnerability reporting from the repository Security tab:

https://github.com/kayhendriksen/foehn/security/advisories/new

If private reporting is unavailable, contact the maintainer through GitHub with a brief non-sensitive message requesting a private security channel:

https://github.com/kayhendriksen

Please include:

- A description of the issue and affected feature.
- Steps to reproduce or a minimal proof of concept, if available.
- Any known impact or workaround.

You can expect an initial response within 7 days. Confirmed vulnerabilities will be fixed in a patch release and documented in the release notes when appropriate.

Published security advisories are listed at:

https://github.com/kayhendriksen/foehn/security/advisories

## Verifying a Release

From 0.5.0 on, every release is built and published by GitHub Actions from the tagged commit, and carries signed [SLSA build provenance](https://slsa.dev/spec/v1.0/provenance) for its wheel and sdist, plus a signed SBOM attestation. The GitHub release lists the wheel, the sdist, the CycloneDX SBOM and the attestation bundles (`*.intoto.jsonl`) next to each other. PyPI also carries [PEP 740](https://peps.python.org/pep-0740/) attestations for the same files.

To check that a file was built by this repository's release workflow, with the [GitHub CLI](https://cli.github.com/):

```bash
gh attestation verify foehn-0.5.0-py3-none-any.whl --repo kayhendriksen/foehn
```

Or against the bundle attached to the release, without querying GitHub's attestation API:

```bash
gh attestation verify foehn-0.5.0-py3-none-any.whl --repo kayhendriksen/foehn \
  --bundle foehn-0.5.0.intoto.jsonl
```

Releases before 0.5.0 carry no provenance.
