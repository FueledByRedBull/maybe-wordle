# Atomic artifact replacement

Maybe Wordle writes generated configs, models, studies, evidence, books, and formal artifacts through `atomic_write`.

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
