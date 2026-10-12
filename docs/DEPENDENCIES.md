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

The 2026-10-12 check of the actual lockfile with cargo-audit 0.22.2, after the
updates listed below, passed the warning-denied command with exactly the one
documented maintenance exception, against 1,307 advisories. Database revision:
`57dcc0f2192e025138907a495c1ac9a7e1e3e123` (2026-10-12). The reassessment
earlier that day used `7eebec69c352c7191b1f13eb95dd510eeca5d1de`, and the
2026-10-05 close-out check used `ef6173cbc5c50ec8166f9a5b28f07834144373ee`.

| Advisory | Locked package | Review and disposition |
| --- | --- | --- |
| [RUSTSEC-2026-0192](https://rustsec.org/advisories/RUSTSEC-2026-0192.html) | `ttf-parser` 0.25.1 | Unmaintained transitive font parser. `epaint` 0.36 renders text with `skrifa` and no longer uses `ab_glyph`. The remaining path is `winit` 0.30.13, `sctk-adwaita` 0.10.1, `ab_glyph`, `owned_ttf_parser`: the Wayland window-decoration title font, on Linux and BSD targets only. The audit scans the whole lockfile, so the exception stays although Windows and macOS builds do not compile the parser. Keep it visible as dependency debt; do not claim the parser is risk-free. Removing it needs an upstream `winit` or `sctk-adwaita` release without `ab_glyph`. |

Only this advisory ID is ignored by the warning-denied job. Reassess it on the
next GUI or lockfile update and no later than 2027-02-01; remove the exception
when its package or advisory no longer applies. New advisories fail the job.
Do not add broad package ignores, suppress database errors, or label an
unreviewed exception as a clean audit.

Lockfile changes on 2026-10-12, all before the prospective freeze was made:

- `eframe` 0.32.3 to 0.36.2 (minimum Rust 1.95). The text stack is now
  `skrifa`, `harfrust` and `vello_cpu`, and eframe's default renderer is wgpu
  instead of glow. The `windows` crate is locked at the single version 0.62.2
  that `wgpu-hal` requires. `paste` 1.0.15 left the lockfile, so its former
  RUSTSEC-2024-0436 exception was dropped. Native GUI acceptance was repeated
  on the resulting build; see
  [EXPERIMENTS.md](EXPERIMENTS.md#october-12-native-acceptance).
- The three open Dependabot updates: `rayon` 1.11.0 to 1.12.0, `serde` 1.0.228
  to 1.0.229 and `toml` 0.8.23 to 1.0.6.
- `quinn-proto` 0.11.15 to 0.11.19, which also moves `rand` 0.9.3 to 0.10.3.
  Five advisories published after the morning check (RUSTSEC-2026-0338 through
  RUSTSEC-2026-0342) name 0.11.15. The crate is locked only through
  `reqwest`'s optional HTTP/3 support, which this project does not enable:
  `cargo tree --locked --target all --invert quinn-proto` prints nothing and
  the release executables were byte-identical before and after. The lockfile
  entry was updated instead of adding an exception.

Local reproduction:

```powershell
cargo audit --file Cargo.lock --deny warnings --ignore RUSTSEC-2026-0192
cargo tree --locked --target all --invert ttf-parser@0.25.1
```

Sources: [RustSec tooling](https://rustsec.org/) and
[GitHub Dependabot configuration](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference).
