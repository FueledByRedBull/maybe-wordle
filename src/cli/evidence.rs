use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use maybe_wordle::{atomic_file::atomic_write, solver::Solver};

pub(super) fn benchmark_docs(
    evidence: &Path,
    markdown_output: &Path,
    readme: &Path,
    update: bool,
) -> Result<()> {
    let bytes = std::fs::read(evidence)
        .with_context(|| format!("failed to read {}", evidence.display()))?;
    let artifact: maybe_wordle::solver::PredictiveEvidenceArtifact = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", evidence.display()))?;
    let generated = Solver::render_development_evidence_markdown(&artifact)?;
    let readme_text = std::fs::read_to_string(readme)
        .with_context(|| format!("failed to read {}", readme.display()))?;
    let updated_readme =
        replace_generated_section(&readme_text, &generated, EvidenceKind::Predictive)?;
    if update {
        atomic_write(markdown_output, generated.as_bytes())?;
        atomic_write(readme, updated_readme.as_bytes())?;
        println!(
            "updated_markdown={} updated_readme={}",
            markdown_output.display(),
            readme.display()
        );
    } else {
        let existing_markdown = std::fs::read_to_string(markdown_output)
            .with_context(|| format!("failed to read {}", markdown_output.display()))?;
        verify_predictive_evidence_docs(
            &existing_markdown,
            &generated,
            &readme_text,
            &updated_readme,
            evidence,
        )?;
        println!("predictive evidence documentation is current");
    }
    Ok(())
}

pub(super) fn rolling_docs(
    comparison: &[PathBuf],
    markdown_output: &Path,
    readme: &Path,
    update: bool,
) -> Result<()> {
    let comparisons = comparison
        .iter()
        .map(|path| {
            let bytes = std::fs::read(path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            serde_json::from_slice::<maybe_wordle::solver::RollingComparisonArtifact>(&bytes)
                .with_context(|| format!("failed to parse {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let generated = Solver::render_rolling_comparison_markdown(&comparisons)?;
    let readme_text = std::fs::read_to_string(readme)
        .with_context(|| format!("failed to read {}", readme.display()))?;
    let updated_readme =
        replace_generated_section(&readme_text, &generated, EvidenceKind::Rolling)?;
    if update {
        atomic_write(markdown_output, generated.as_bytes())?;
        atomic_write(readme, updated_readme.as_bytes())?;
        println!(
            "updated_markdown={} updated_readme={}",
            markdown_output.display(),
            readme.display()
        );
    } else {
        let existing = std::fs::read_to_string(markdown_output)
            .with_context(|| format!("failed to read {}", markdown_output.display()))?;
        if canonical_newlines(&existing) != canonical_newlines(&generated)
            || canonical_newlines(&updated_readme) != canonical_newlines(&readme_text)
        {
            bail!("rolling evidence documentation is stale; rerun with --update");
        }
        println!("rolling evidence documentation is current");
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum EvidenceKind {
    Predictive,
    Rolling,
}

fn replace_generated_section(readme: &str, generated: &str, kind: EvidenceKind) -> Result<String> {
    let (start_marker, end_marker, context) = match kind {
        EvidenceKind::Predictive => (
            "<!-- BEGIN GENERATED PREDICTIVE EVIDENCE -->",
            "<!-- END GENERATED PREDICTIVE EVIDENCE -->",
            "evidence",
        ),
        EvidenceKind::Rolling => (
            "<!-- BEGIN GENERATED ROLLING EVIDENCE -->",
            "<!-- END GENERATED ROLLING EVIDENCE -->",
            "rolling evidence",
        ),
    };
    let start = readme
        .find(start_marker)
        .ok_or_else(|| anyhow!("README is missing the generated {context} start marker"))?;
    let end_start = readme[start..]
        .find(end_marker)
        .map(|offset| start + offset)
        .ok_or_else(|| anyhow!("README is missing the generated {context} end marker"))?;
    let end = end_start + end_marker.len();
    let mut updated = String::with_capacity(readme.len() + generated.len());
    updated.push_str(&readme[..start]);
    updated.push_str(generated.trim_end());
    updated.push_str(&readme[end..]);
    Ok(updated)
}

fn canonical_newlines(text: &str) -> String {
    text.replace("\r\n", "\n")
}

fn verify_predictive_evidence_docs(
    existing_markdown: &str,
    generated: &str,
    readme_text: &str,
    updated_readme: &str,
    evidence: &Path,
) -> Result<()> {
    if canonical_newlines(existing_markdown) != canonical_newlines(generated) {
        bail!(
            "generated evidence fragment is stale: run benchmark-evidence-docs --evidence {} --update",
            evidence.display()
        );
    }
    if canonical_newlines(updated_readme) != canonical_newlines(readme_text) {
        bail!(
            "README evidence fragment is stale: run benchmark-evidence-docs --evidence {} --update",
            evidence.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_evidence_replacement_preserves_both_marker_contracts() {
        for (kind, marker, context) in [
            (EvidenceKind::Predictive, "PREDICTIVE", "evidence"),
            (EvidenceKind::Rolling, "ROLLING", "rolling evidence"),
        ] {
            let start = format!("<!-- BEGIN GENERATED {marker} EVIDENCE -->");
            let end = format!("<!-- END GENERATED {marker} EVIDENCE -->");
            let readme = format!("before\r\n{start}\nold\n{end}\r\nafter\r\n");
            let generated = format!("{start}\nnew\n{end}\n\n");
            assert_eq!(
                replace_generated_section(&readme, &generated, kind).unwrap(),
                format!("before\r\n{start}\nnew\n{end}\r\nafter\r\n")
            );
            assert_eq!(
                replace_generated_section("no markers", &generated, kind)
                    .unwrap_err()
                    .to_string(),
                format!("README is missing the generated {context} start marker")
            );
            assert_eq!(
                replace_generated_section(&start, &generated, kind)
                    .unwrap_err()
                    .to_string(),
                format!("README is missing the generated {context} end marker")
            );
        }
    }

    #[test]
    fn documentation_verification_ignores_platform_line_endings() {
        assert_eq!(
            canonical_newlines("alpha\r\nbeta\r\n"),
            canonical_newlines("alpha\nbeta\n")
        );
        assert_ne!(
            canonical_newlines("alpha\r\nbeta\r\n"),
            canonical_newlines("alpha\ngamma\n")
        );
    }

    #[test]
    fn predictive_docs_verification_accepts_crlf_checkout() {
        assert!(
            super::verify_predictive_evidence_docs(
                "score 3.1944\r\n",
                "score 3.1944\n",
                "before\r\nscore 3.1944\r\n",
                "before\nscore 3.1944\n",
                Path::new("evidence.json"),
            )
            .is_ok()
        );
    }

    #[test]
    fn predictive_docs_verification_rejects_stale_content() {
        let fragment_error = super::verify_predictive_evidence_docs(
            "score 3.0000\r\n",
            "score 3.1944\n",
            "before\r\nscore 3.1944\r\n",
            "before\nscore 3.1944\n",
            Path::new("evidence.json"),
        )
        .expect_err("stale fragment");
        assert!(
            fragment_error
                .to_string()
                .contains("generated evidence fragment is stale")
        );

        let readme_error = super::verify_predictive_evidence_docs(
            "score 3.1944\r\n",
            "score 3.1944\n",
            "before\r\nscore 3.0000\r\n",
            "before\nscore 3.1944\n",
            Path::new("evidence.json"),
        )
        .expect_err("stale README");
        assert!(
            readme_error
                .to_string()
                .contains("README evidence fragment is stale")
        );
    }
}
