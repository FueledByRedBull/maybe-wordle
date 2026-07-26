use sha2::{Digest, Sha256};
use std::io::{self, Read};

pub const IDENTITY_FORMAT: &str = "sha256-length-delimited-v1";
pub const TAGGED_DIGEST_PREFIX: &str = "sha256-v1:";

pub struct CanonicalSha256 {
    hasher: Sha256,
}

impl CanonicalSha256 {
    pub fn new(domain: &str) -> Self {
        let mut digest = Self {
            hasher: Sha256::new(),
        };
        digest.field(domain.as_bytes());
        digest
    }

    pub fn field(&mut self, value: &[u8]) -> &mut Self {
        self.hasher.update((value.len() as u64).to_le_bytes());
        self.hasher.update(value);
        self
    }

    pub fn field_reader(&mut self, reader: &mut impl Read, length: u64) -> io::Result<&mut Self> {
        self.hasher.update(length.to_le_bytes());
        let mut remaining = length;
        let mut buffer = [0u8; 64 * 1024];
        while remaining > 0 {
            let limit = remaining.min(buffer.len() as u64) as usize;
            let read = reader.read(&mut buffer[..limit])?;
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "identity field became shorter while hashing",
                ));
            }
            self.hasher.update(&buffer[..read]);
            remaining -= read as u64;
        }
        let mut trailing = [0u8; 1];
        if reader.read(&mut trailing)? != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "identity field became longer while hashing",
            ));
        }
        Ok(self)
    }

    pub fn finish(self) -> [u8; 32] {
        self.hasher.finalize().into()
    }

    pub fn finish_hex(self) -> String {
        hex(&self.finish())
    }

    pub fn finish_tagged(self) -> String {
        format!("{TAGGED_DIGEST_PREFIX}{}", self.finish_hex())
    }
}

pub fn digest_bytes(domain: &str, value: &[u8]) -> [u8; 32] {
    let mut digest = CanonicalSha256::new(domain);
    digest.field(value);
    digest.finish()
}

pub fn digest_bytes_hex(domain: &str, value: &[u8]) -> String {
    hex(&digest_bytes(domain, value))
}

pub fn digest_bytes_tagged(domain: &str, value: &[u8]) -> String {
    tag(&digest_bytes(domain, value))
}

pub fn tag(digest: &[u8; 32]) -> String {
    format!("{TAGGED_DIGEST_PREFIX}{}", hex(digest))
}

pub fn is_tagged_digest(value: &str) -> bool {
    value.len() == TAGGED_DIGEST_PREFIX.len() + 64
        && value.starts_with(TAGGED_DIGEST_PREFIX)
        && value[TAGGED_DIGEST_PREFIX.len()..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_fields_are_deterministic_and_boundary_safe() {
        let mut first = CanonicalSha256::new("test");
        first.field(b"ab").field(b"c");
        let first = first.finish_tagged();

        let mut repeat = CanonicalSha256::new("test");
        repeat.field(b"ab").field(b"c");
        assert_eq!(first, repeat.finish_tagged());

        let mut different_boundary = CanonicalSha256::new("test");
        different_boundary.field(b"a").field(b"bc");
        assert_ne!(first, different_boundary.finish_tagged());
        assert_eq!(first.len(), TAGGED_DIGEST_PREFIX.len() + 64);
    }

    #[test]
    fn domains_separate_identical_payloads() {
        assert_ne!(
            digest_bytes_hex("first", b"same"),
            digest_bytes_hex("second", b"same")
        );
    }

    #[test]
    fn streamed_and_one_shot_fields_match() {
        let payload = (0..200_000)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let mut one_shot = CanonicalSha256::new("stream-test");
        one_shot.field(&payload);

        let mut streamed = CanonicalSha256::new("stream-test");
        streamed
            .field_reader(&mut std::io::Cursor::new(&payload), payload.len() as u64)
            .expect("stream field");
        assert_eq!(one_shot.finish(), streamed.finish());
    }
}
