use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use super::*;

const FORMAT_VERSION: u32 = 1;
const POINTER_NAME: &str = "current.json";
const MEMBERS: [(&str, u64); 8] = [
    (PRIOR_SPEC_NAME, parsing::MAX_INPUT_BYTES),
    (
        FORMAL_PATTERN_TABLE_NAME,
        parsing::MAX_PATTERN_CELLS as u64 + 112,
    ),
    (MANIFEST_NAME, parsing::MAX_JSON_BYTES),
    (SMALL_STATE_TABLE_NAME, 4096),
    (CERTIFICATE_NAME, parsing::MAX_CERTIFICATE_BYTES),
    (VALUES_NAME, MAX_FORMAL_BINARY_BYTES),
    (POLICY_NAME, MAX_FORMAL_BINARY_BYTES),
    (METADATA_NAME, parsing::MAX_JSON_BYTES),
];
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(0);

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerationPointer {
    format_version: u32,
    model_id: String,
    manifest_hash: String,
    generation: String,
    #[serde(deserialize_with = "parsing::bounded_vec::<_, _, 8>")]
    members: Vec<MemberDigest>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemberDigest {
    name: String,
    bytes: u64,
    sha256: String,
}

fn digest_file(path: &Path, maximum: u64) -> Result<(u64, String)> {
    let mut file = parsing::open_regular_file(path)?;
    let length = file.metadata()?.len();
    if length == 0 || length > maximum {
        bail!(
            "formal generation member exceeds resource limits: {}",
            path.display()
        );
    }
    let mut hash = CanonicalSha256::new("maybe-wordle-formal-generation-member-v1");
    hash.field_reader(&mut file, length)?;
    Ok((length, hash.finish_tagged()))
}

pub(super) fn resolve(paths: &ProjectPaths, model_id: &str) -> Result<PolicyArtifactSet> {
    parsing::validate_component(model_id)?;
    let inputs = PolicyArtifactSet::for_model(paths, model_id);
    let pointer: GenerationPointer = parsing::read_json(&inputs.model_dir.join(POINTER_NAME), 8192)
        .context("no valid published formal generation; rebuild formal artifacts (raw prior.toml is preserved)")?;
    parsing::validate_component(&pointer.generation)?;
    if pointer.format_version != FORMAT_VERSION
        || pointer.model_id != model_id
        || !crate::identity::is_tagged_digest(&pointer.manifest_hash)
        || !pointer.generation.starts_with("gen-")
        || pointer.members.len() != MEMBERS.len()
    {
        bail!("invalid formal generation pointer; rebuild formal artifacts");
    }
    let generation_root = inputs.model_dir.join("generations");
    let directory = generation_root.join(&pointer.generation);
    let canonical = directory.canonicalize()?;
    let canonical_generations = generation_root.canonicalize()?;
    if canonical_generations.parent() != Some(inputs.model_dir.canonicalize()?.as_path())
        || canonical.parent() != Some(canonical_generations.as_path())
    {
        bail!("formal generation pointer escapes its generation directory");
    }
    for (member, (expected_name, maximum)) in pointer.members.iter().zip(MEMBERS) {
        if member.name != expected_name {
            bail!("formal generation member set is not canonical");
        }
        let path = directory.join(expected_name);
        if path.canonicalize()?.parent() != Some(canonical.as_path())
            || !fs::symlink_metadata(&path)?.file_type().is_file()
        {
            bail!("formal generation member must be a regular, local file");
        }
        let (length, digest) = digest_file(&path, maximum)?;
        if member.bytes != length || member.sha256 != digest {
            bail!(
                "formal generation checksum mismatch for {expected_name}; rebuild formal artifacts"
            );
        }
    }
    let artifacts = PolicyArtifactSet::in_directory(directory);
    let manifest: FormalManifest =
        parsing::read_json(&artifacts.manifest, parsing::MAX_JSON_BYTES)?;
    if manifest.model_id != pointer.model_id || manifest.manifest_hash != pointer.manifest_hash {
        bail!("formal generation pointer and manifest identity disagree");
    }
    Ok(artifacts)
}

pub(super) fn publish(
    model: &FormalModel,
    memo: &HashMap<StateKey, StoredState>,
    mut metadata: ProofMetadata,
    certificate: &ProofCertificate,
    paths: &ProjectPaths,
    total_started: Instant,
    cancelled: &dyn Fn() -> bool,
) -> Result<()> {
    if memo.is_empty()
        || memo.len() > MAX_FORMAL_STATES
        || certificate.states.len() > MAX_FORMAL_STATES
    {
        bail!("formal generation exceeds persisted state resource limits");
    }
    let inputs = PolicyArtifactSet::for_model(paths, &model.manifest.model_id);
    let generations = inputs.model_dir.join("generations");
    fs::create_dir_all(&generations)?;
    if generations.canonicalize()?.parent() != Some(inputs.model_dir.canonicalize()?.as_path()) {
        bail!("formal generation directory escapes its model directory");
    }
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let generation = format!(
        "gen-{timestamp}-{}-{}",
        std::process::id(),
        NEXT_GENERATION.fetch_add(1, Ordering::Relaxed)
    );
    let directory = generations.join(&generation);
    // Exclusive creation prevents either concurrent builders or interrupted retries overwriting a generation.
    fs::create_dir(&directory)?;
    let artifacts = PolicyArtifactSet::in_directory(directory);
    let mut entries = memo.iter().collect::<Vec<_>>();
    entries.sort_by(|(left, _), (right, _)| {
        left.state_hash()
            .cmp(&right.state_hash())
            .then_with(|| left.cmp_storage(right, model.answers.len()))
    });
    let mut members = Vec::with_capacity(MEMBERS.len());
    for (index, (name, maximum)) in MEMBERS.iter().copied().enumerate() {
        check_cancelled(cancelled)?;
        let path = artifacts.model_dir.join(name);
        match name {
            PRIOR_SPEC_NAME => atomic_write(&path, &model.raw_prior)?,
            FORMAL_PATTERN_TABLE_NAME => {
                crate::atomic_file::atomic_write_with(&path, |destination| {
                    let mut source = parsing::open_regular_file(&inputs.pattern_table)?;
                    let length = source.metadata()?.len();
                    if length > maximum {
                        bail!("formal pattern table exceeds resource limit");
                    }
                    let copied = std::io::copy(&mut (&mut source).take(maximum + 1), destination)?;
                    if copied != length {
                        bail!("formal pattern input changed during publication");
                    }
                    Ok(())
                })?;
            }
            MANIFEST_NAME => atomic_write(&path, &serde_json::to_vec_pretty(&model.manifest)?)?,
            SMALL_STATE_TABLE_NAME => {
                atomic_write(&path, &serde_json::to_vec_pretty(&model.small_state_table)?)?
            }
            CERTIFICATE_NAME => atomic_write(&path, &serde_json::to_vec_pretty(certificate)?)?,
            VALUES_NAME => write_values(&path, model, &entries)?,
            POLICY_NAME => write_policy(&path, model, &entries)?,
            METADATA_NAME => {
                metadata.pre_publish_millis = total_started.elapsed().as_millis();
                atomic_write(&path, &serde_json::to_vec_pretty(&metadata)?)?;
            }
            _ => unreachable!("fixed generation member set"),
        }
        let (bytes, sha256) = digest_file(&path, maximum)?;
        members.push(MemberDigest {
            name: name.to_string(),
            bytes,
            sha256,
        });
        publication_stage(index + 1)?;
    }
    // Structural load only. Independent mathematical verification remains an explicit command.
    FormalPolicyRuntime::load_generation(paths, &model.manifest.model_id, artifacts)?;
    check_cancelled(cancelled)?;
    #[cfg(unix)]
    {
        // Persist newly-created directory entries before a durable pointer can name them.
        File::open(&generations)?.sync_all()?;
        File::open(&inputs.model_dir)?.sync_all()?;
    }
    publication_stage(MEMBERS.len() + 1)?;
    let pointer = GenerationPointer {
        format_version: FORMAT_VERSION,
        model_id: model.manifest.model_id.clone(),
        manifest_hash: model.manifest.manifest_hash.clone(),
        generation,
        members,
    };
    atomic_write(
        &inputs.model_dir.join(POINTER_NAME),
        &serde_json::to_vec_pretty(&pointer)?,
    )
}

fn publication_stage(_stage: usize) -> Result<()> {
    #[cfg(test)]
    if FAIL_AFTER_STAGE.get() == Some(_stage) {
        bail!("injected pre-publication interruption");
    }
    Ok(())
}

#[cfg(test)]
thread_local! { pub(super) static FAIL_AFTER_STAGE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) }; }
