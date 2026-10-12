# Dependency and workflow policy

`Cargo.lock` is authoritative. CI uses Rust 1.97.0 and locked Cargo commands;
`rust-toolchain.toml` selects the same local compiler and components. Correctness
checks run on Windows, Linux and macOS without opening a native window. Native
interactive GUI acceptance is separate and is not implied by a headless build.

CI uses Cargo's `line-tables-only` test debug information. This retains file/line
backtraces but avoids repeatedly hashing bulky embedded DWARF in the executable
identity checks. The test command additionally applies
`--config 'profile.test.package.sha2.opt-level=3'`: only the existing SHA-256
dependency is optimized to reduce full-file hashing overhead in tiny fixture budgets.
Application test code stays unoptimized. Debug assertions, overflow checks,
test concurrency, executable-content rechecks and cumulative deadlines are unchanged;
release builds, dependency versions and application behavior are unaffected.
See [Cargo profile debug settings and package overrides](https://doc.rust-lang.org/cargo/reference/profiles.html).

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

The 2026-10-12 reassessment of the actual lockfile with cargo-audit 0.22.2
passed the warning-denied command with exactly the two documented maintenance
exceptions below, against 1,296 advisories. No application dependencies or
exceptions changed. Database revision:
`7eebec69c352c7191b1f13eb95dd510eeca5d1de` (2026-10-09). The 2026-10-05
close-out check used `ef6173cbc5c50ec8166f9a5b28f07834144373ee`.

| Advisory | Locked package | Review and disposition |
| --- | --- | --- |
| [RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436.html) | `paste` 1.0.15 | Unmaintained macro crate retained in the lockfile; `cargo tree --locked --target all --invert paste` still finds no active path, with or without feature edges. No vulnerability is alleged by this advisory. |
| [RUSTSEC-2026-0192](https://rustsec.org/advisories/RUSTSEC-2026-0192.html) | `ttf-parser` 0.25.1 | Unmaintained transitive font parser through `owned_ttf_parser` / `ab_glyph` under `epaint` 0.32.3, and under `sctk-adwaita` for Linux window decorations. `ttf-parser`, `owned_ttf_parser` and `ab_glyph` have no newer release. Keep it visible as dependency debt; do not claim the parser is risk-free. |

A fix path now exists for the second exception: `eframe`/`epaint` 0.36.2
(published 2026-09-08, minimum Rust 1.95) no longer depends on `ab_glyph`; its
text stack is `skrifa`, `harfrust` and `vello_cpu`. Moving from 0.32.3 to 0.36
crosses four minor releases of the GUI framework, so it is a separately
reviewed GUI update with fresh native acceptance, not part of this
reassessment. Whether the Linux `sctk-adwaita` path still locks `ttf-parser`
after that update must be checked on the resulting lockfile.

Only these two advisory IDs are ignored by the warning-denied job. Reassess
them at that GUI update or on the next lockfile change, and no later than
2027-02-01; remove an exception when its package/advisory no longer applies.
New advisories fail the job. Do not add broad package ignores, suppress
database errors, or label an unreviewed exception as a clean audit.

Local reproduction:

```powershell
cargo audit --file Cargo.lock --deny warnings --ignore RUSTSEC-2024-0436 --ignore RUSTSEC-2026-0192
cargo tree --locked --target all --invert ttf-parser@0.25.1
```

Sources: [RustSec tooling](https://rustsec.org/) and
[GitHub Dependabot configuration](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference).
