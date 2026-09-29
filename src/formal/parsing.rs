use std::{fmt, marker::PhantomData};

use serde::{
    Deserializer,
    de::{DeserializeOwned, Error, SeqAccess, Visitor},
};

use super::*;

// These are format resource limits, not claims that exact solving at these sizes is practical.
pub(super) const MAX_WORDS: usize = u16::MAX as usize;
pub(super) const MAX_PATTERN_CELLS: usize = 64 * 1024 * 1024;
pub(super) const MAX_INPUT_BYTES: u64 = 1024 * 1024;
pub(super) const MAX_JSON_BYTES: u64 = 1024 * 1024;
pub(super) const MAX_CERTIFICATE_BYTES: u64 = 32 * 1024 * 1024;

pub(super) fn validate_component(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 96
        || value.starts_with('.')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        bail!("formal model/generation name must be a simple ASCII path component");
    }
    Ok(())
}

pub(super) fn open_regular_file(path: &Path) -> Result<std::fs::File> {
    // These checks reject malformed special files; they are not a lock against
    // a concurrent replacement between metadata inspection and opening.
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?;
    if !metadata.file_type().is_file() {
        bail!("formal resource must be a regular file: {}", path.display());
    }
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    if !file.metadata()?.file_type().is_file() {
        bail!("formal resource is not a regular file: {}", path.display());
    }
    Ok(file)
}

pub(super) fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let file = open_regular_file(path)?;
    let length = file.metadata()?.len();
    if length > maximum {
        bail!(
            "{} exceeds the {} byte formal resource limit",
            path.display(),
            maximum
        );
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        bail!("{} grew beyond its formal resource limit", path.display());
    }
    Ok(bytes)
}

pub(super) fn read_json<T: DeserializeOwned>(path: &Path, maximum: u64) -> Result<T> {
    serde_json::from_slice(&read_bounded(path, maximum)?).with_context(|| {
        format!(
            "invalid formal artifact {}; rebuild formal artifacts",
            path.display()
        )
    })
}

pub(super) fn read_small_table(path: &Path) -> Result<SmallStateTable> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Table {
        version: u32,
        max_size: usize,
        #[serde(deserialize_with = "bounded_vec::<_, _, { SMALL_STATE_LIMIT + 1 }>")]
        expected_lower_bound_by_size: Vec<f64>,
    }
    let table: Table = read_json(path, 4096)?;
    Ok(SmallStateTable {
        version: table.version,
        max_size: table.max_size,
        expected_lower_bound_by_size: table.expected_lower_bound_by_size,
    })
}

pub(super) fn bounded_vec<'de, D, T, const N: usize>(
    deserializer: D,
) -> std::result::Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Bounded<T, const N: usize>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for Bounded<T, N> {
        type Value = Vec<T>;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "an array of at most {N} entries")
        }
        fn visit_seq<A: SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            if sequence.size_hint().is_some_and(|length| length > N) {
                return Err(A::Error::custom("formal array resource limit exceeded"));
            }
            let mut values = Vec::new();
            while values.len() < N {
                let Some(value) = sequence.next_element()? else {
                    return Ok(values);
                };
                values.try_reserve(1).map_err(A::Error::custom)?;
                values.push(value);
            }
            if sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {
                return Err(A::Error::custom("formal array resource limit exceeded"));
            }
            Ok(values)
        }
    }
    deserializer.deserialize_seq(Bounded::<T, N>(PhantomData))
}
