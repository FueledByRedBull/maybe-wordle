# Atomic artifact replacement

Maybe Wordle writes generated configs, models, studies, evidence, books, and formal artifacts through `atomic_write`.

Authoritative manual-answer additions, seed merges and initial configuration creation also use this protocol. They serialize and validate complete replacements before publishing. Seed read-modify-write operations share an OS-backed edit lock; overlapping edits fail explicitly and can be retried. Configuration creation/save has a destination-scoped lock. The ignored `.*.mwedit.lock` sidecars remain on disk so concurrent processes always lock the same file; the operating system releases their locks when the file handle closes or the process exits. Do not delete a sidecar while a writer may be active. These are cooperating-writer locks, not protection against an external editor. See the [Rust file-lock contract](https://doc.rust-lang.org/std/fs/struct.File.html#method.try_lock).

`atomic_write_with` streams large payloads into the same owned temporary without constructing an extra complete buffer. Relative destinations (including bare filenames and dot components) are resolved once using native path rules. Windows ordinary paths are resolved before verbatim conversion; POSIX parent traversal retains directory-symlink semantics. Native Windows regression checks cover Unicode and long paths; native UNC-share durability is not claimed.

The protocol is:

1. remove only stale sibling temporaries matching the exact destination-specific `mwatomic-v1` ownership grammar and at least 24 hours old;
2. create a uniquely named, versioned sibling temporary file with `create_new`;
3. write all bytes;
4. flush userspace buffers and call `sync_all` on the temporary file;
5. atomically replace the destination;
6. on Unix, open and `sync_all` the containing directory after `rename`;
7. on Windows, call `MoveFileExW` with `REPLACE_EXISTING | WRITE_THROUGH`.

Sibling placement is required: atomic rename/replace is only assumed within one filesystem and directory. The implementation does not claim durability on network filesystems or filesystems that do not honor the documented platform primitives.

Owned temporaries use the exact form `.<destination>.mwatomic-v1.<decimal-pid>.<decimal-nonce>.tmp`. Cleanup is scoped to the same destination name, version marker, two decimal ownership fields, sibling directory, regular-file type, and 24-hour minimum age. Malformed names, old formats, another destination's temporaries, unrelated `.tmp` files, future timestamps, and fresh writes are preserved. Enumeration, metadata, and removal errors fail loudly except for a concurrent `NotFound`.

Injected tests cover failures after temporary creation, after writing, after temporary-file sync, immediately before replacement, immediately around the real platform replacement primitive, and around Unix parent-directory sync. At every pre-replace failure, the previous destination remains byte-for-byte valid and the owned temporary file is removed. The Windows-gated test executes the real `MoveFileExW`: injection before the call preserves the old bytes, while injection after a successful call reports an error with the new bytes visible. Unix-gated tests apply the same distinction to `rename` and inject before and after the real directory `sync_all`.

After a successful rename but before Unix directory sync completes, the new pathname can be visible while power-loss durability remains uncertain. Rolling back to the old bytes at that point would itself require another non-atomic replacement and is not attempted. A directory-sync error is therefore reported even though the new file may be visible. Windows `WRITE_THROUGH` provides the corresponding strongest available replacement request through the used API.

No broad directory sweep or suffix-only deletion is used. The stale-cleanup regression creates an exact owned sibling plus malformed, old-format, other-destination, and user-style temporary names, then proves only the exact owned file is removed.

## Predictive books

Predictive book manifest v3 uses a new policy-computation identity and filenames
after the audit's shared replay/terminal changes. Earlier v2 books are not selected
as current policy artifacts; they remain rebuildable historical caches. A present
current-version book with invalid identity or content is an error, not a missing
book that silently triggers another computation.

## Formal generations

A formal build writes a new immutable `generations/gen-.../` directory under
`data/formal/<model>/`. Its manifest, policy, values, certificate, metadata,
small-state table, pattern table and prior snapshot belong to that one generation.
Only after every member is written and synced does the builder atomically publish
`current.json`, which binds the member names, byte lengths and SHA-256 digests.
Readers resolve that pointer once, validate the complete set, and reject missing,
mixed or corrupt members; loading never repairs a generation in place. A failure
before pointer publication leaves the previous generation selected. A failure
reported after replacement may leave the new, complete generation selected.

Bounded formal readers and generation digests require regular files before opening
them and validate the opened handle. Directories, symlinks and static special files
are rejected; the Unix FIFO regression runs only on Unix. These checks do not
protect against an adversary replacing paths between inspection and opening.

The root `prior.toml` is editable input, not the selected generation's prior
snapshot. The root pattern table is a rebuildable input cache. Older flat proof
sets must be rebuilt; they are not silently adopted. New-directory power-loss
durability on Windows is not independently established by the replacement tests.

## Sealed evaluation ownership

The once-only sealed-test marker uses a separate acquisition protocol. Solver
setup, exact inclusive unique-date coverage, frozen identities, output-path
checks and the private window-data digest precede it. A cooperating edit lock
serializes claims in `benchmarks/predictive/sealed-windows/`; a claim is rejected
if any recorded window overlaps it, even when the candidate or contract changes.
The legacy global marker is preserved and its report's consumed dates participate
in overlap checks. Unknown or corrupt ownership records fail closed.
`create_new` allows only one evaluator to acquire a date-named marker; the winner
writes and syncs it, then syncs its parent directory on Unix before evaluation.
A write or sync error leaves the marker in place and does not start evaluation.
The completed status uses `atomic_write`. The Windows new-file acquisition
uses file `sync_all`, but its directory-entry survival across power loss has
not been independently established; do not equate it with the Windows
`MoveFileExW` replacement guarantee above. Failure or interruption does not release
ownership. Source and window-data identities are rechecked before exclusively
creating the report; output paths cannot alias ownership files or enter the ledger
directory. A newly allowed non-overlapping window can be claimed without deleting
the historical record. No real reserved window was consumed by audit regressions.

The historical sealed-test report (before the window-digest protocol) binds the development-cutoff inputs and
contains its evaluated game targets and paths, but does not separately bind a
digest of the sealed-window source rows or recheck those rows before report
publication. Its once-only result cannot be rerun to repair that provenance
gap; do not infer the stronger prospective guarantee from it.

The separate prospective workflow writes its frozen candidate with exclusive
`create_new` acquisition, file `sync_all`, and Unix parent-directory sync;
an existing freeze file cannot be replaced. Evaluation first validates the
frozen identity, source, configuration, complete future date coverage, and
history digest. Freezing also requires complete daily history after the
development cutoff through the UTC freeze date; its digest is bound to the
freeze and rechecked before reservation and publication. Evaluation then
exclusively creates a global one-window reservation
and a date-named marker before scoring any target. The marker records a
digest, not target words. A failed or interrupted run leaves the reservation
in place. A completed report and marker-status update use `atomic_write`;
the same Windows new-file directory-entry durability caveat applies to both
prospective `create_new` acquisitions. Output paths are checked for aliases of
either marker and cannot share their directory; the raw report defaults to ignored
`target/diagnostics/`.
