# Dependency and workflow policy

`Cargo.lock` is authoritative. CI uses Rust 1.97.0 and locked Cargo commands;
`rust-toolchain.toml` selects the same local compiler and components. Correctness
checks run on Windows, Linux and macOS without opening a native window. Native
interactive GUI acceptance is separate and is not implied by a headless build.

The independent dependency workflow runs on relevant pull requests, weekly and
on demand. It installs the pinned `cargo-audit` 0.22.2 with its lockfile, checks
this repository's complete lockfile against current RustSec data, and fails on
vulnerabilities or unreviewed warnings. Database/network failure is not a pass.
It does not install or upgrade application dependencies.

GitHub Actions are pinned to full commit IDs. Weekly Dependabot pull requests
cover those action revisions and Cargo dependencies; updates still need human
review and the locked correctness gates. Review an action's source and changes,
including its nested actions, before updating its pin. Audit-tool and compiler
pins are reviewed explicitly alongside dependency changes; they do not float.
This policy is not a comprehensive license audit or a proof that dependencies
have no unknown vulnerabilities.

## Reviewed maintenance warnings

The 2026-09-29 check of the actual lockfile with cargo-audit 0.22.2 found **zero
known vulnerability advisories** and the two maintenance warnings below.
Database revision: `f23b768236fe2880e4cfa167da662cad8ca79240`.

| Advisory | Locked package | Review and disposition |
| --- | --- | --- |
| [RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436.html) | `paste` 1.0.15 | Unmaintained macro crate retained in the lockfile; `cargo tree --locked --target all --invert paste` found no active feature path. No vulnerability is alleged by this advisory. |
| [RUSTSEC-2026-0192](https://rustsec.org/advisories/RUSTSEC-2026-0192.html) | `ttf-parser` 0.25.1 | Unmaintained transitive font parser through `owned_ttf_parser` / `ab_glyph`, used by the GUI stack. Keep it visible as dependency debt; do not claim the parser is risk-free. An upstream-compatible maintained replacement needs a separately reviewed GUI dependency update. |

Only these two advisory IDs are ignored by the warning-denied job. Reassess
them on the next GUI/lockfile update and no later than 2026-12-01; remove an
exception when its package/advisory no longer applies. New advisories fail the
job. Do not add broad package ignores, suppress database errors, or label an
unreviewed exception as a clean audit.

Local reproduction:

```powershell
cargo audit --file Cargo.lock --deny warnings --ignore RUSTSEC-2024-0436 --ignore RUSTSEC-2026-0192
cargo tree --locked --target all --invert ttf-parser@0.25.1
```

Sources: [RustSec tooling](https://rustsec.org/) and
[GitHub Dependabot configuration](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference).
