//! The frozen item set: `Item` (JSONL rows), its content hash, and the optional
//! manifest that pins that hash.
//!
//! An item's `question` field is a [`jev_core::Question`] — the real M0/M1
//! contract enum — so an item cannot drift into a lookalike shape that the server
//! would never accept. `render_question` is then used to hash the *rendered
//! prompts* of the whole set, which pins the prompt shape the evaluation was run
//! against (that is the "eval hashes the prompt" requirement in `specs/M0.md`).
//!
//! Row shape (one JSON object per line; blank lines and `#` comments are skipped):
//!
//! ```json
//! {"id":"zh-ev-0001","category":"evidence_judgment","source":"authored-zh-v1",
//!  "state":"…","question":{"type":"choice","instructions":"…","criteria":{"支持":null,"不支持":null}},
//!  "gold":"支持","positive":"支持","provenance":"…"}
//! ```
//!
//! `gold` is typed per question kind: a **label string** for `choice` (a key of
//! `criteria`), a **non-negative integer level index** for `score`, and a
//! **boolean** for `noul`. Scoring always happens on slot *keys*, never on the
//! option letters the model sees, so the letter assignment (which follows the
//! `BTreeMap` order of `criteria`) cannot silently change a result.

use crate::error::EvalError;
use crate::sha::sha256_hex;
use jev_core::Question;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The four categories the M1 starter set must cover. `specs/M1.md` §6 names them
/// (证据判断 / 规则应用 / 候选选择 / 缺失证据). Extra categories are allowed; these
/// four are required.
pub const KNOWN_CATEGORIES: [&str; 4] = [
    "evidence_judgment",
    "rule_application",
    "candidate_selection",
    "missing_evidence",
];

/// One frozen evaluation item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub id: String,
    /// One of [`KNOWN_CATEGORIES`] (at least, for the M1 set).
    pub category: String,
    /// Which frozen set / provenance group this item belongs to. Reported as a
    /// stratum; never merged with the others.
    pub source: String,
    /// The shared context the question is asked against. Per item, so an item
    /// file is self-contained.
    pub state: Value,
    /// The question, in the M0/M1 contract shape.
    pub question: Question,
    /// Gold answer: label string / level index / boolean — see the module docs.
    pub gold: Value,
    /// Slot key used as the positive class for `brier_binary`. Optional: for
    /// `noul` it defaults to `"true"`; for other kinds a missing value simply
    /// excludes the item from that one metric rather than guessing a
    /// binarisation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positive: Option<String>,
    /// Free-text note on where this item came from and how it was constructed.
    /// Required by `specs/M1.md` §6 ("来源/构造说明").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<String>,
}

impl Item {
    /// The item's declared slot keys, in contract order.
    ///
    /// * `choice` — the `criteria` keys (the option labels), in `BTreeMap` order.
    /// * `score` — `"0".."n-1"` (the level indices as decimal strings).
    /// * `noul` — `["true", "false"]`.
    pub fn slot_labels(&self) -> Vec<String> {
        match &self.question {
            Question::Choice(q) => q.criteria.keys().cloned().collect(),
            Question::Score(q) => (0..q.criteria.len()).map(|i| i.to_string()).collect(),
            Question::Noul(_) => vec!["true".to_string(), "false".to_string()],
        }
    }

    /// The gold answer expressed as a slot key.
    pub fn gold_label(&self) -> Result<String, String> {
        match &self.question {
            Question::Choice(_) => self
                .gold
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "gold must be a string option label for `choice`".to_string()),
            Question::Score(_) => self
                .gold
                .as_u64()
                .map(|index| index.to_string())
                .ok_or_else(|| {
                    "gold must be a non-negative integer level index for `score`".to_string()
                }),
            Question::Noul(_) => match self.gold.as_bool() {
                Some(true) => Ok("true".to_string()),
                Some(false) => Ok("false".to_string()),
                None => Err("gold must be a boolean for `noul`".to_string()),
            },
        }
    }

    /// The positive class for `brier_binary`, if one is defined for this item.
    pub fn positive_label(&self) -> Option<String> {
        match &self.question {
            // `noul` has a canonical positive class (yes == true).
            Question::Noul(_) => Some(
                self.positive
                    .clone()
                    .unwrap_or_else(|| "true".to_string()),
            ),
            _ => self.positive.clone(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("id must not be empty".to_string());
        }
        if self.category.trim().is_empty() {
            return Err("category must not be empty".to_string());
        }
        if self.source.trim().is_empty() {
            return Err("source must not be empty".to_string());
        }
        let slots = self.slot_labels();
        match &self.question {
            Question::Choice(q) => {
                if q.criteria.len() < 2 {
                    return Err(format!(
                        "`choice` needs at least 2 options, found {}",
                        q.criteria.len()
                    ));
                }
                if q.criteria.len() > jev_core::MAX_CHOICE_OPTIONS {
                    return Err(format!(
                        "`choice` allows at most {} options, found {}",
                        jev_core::MAX_CHOICE_OPTIONS,
                        q.criteria.len()
                    ));
                }
            }
            Question::Score(q) => {
                if q.criteria.len() < 2 {
                    return Err(format!(
                        "`score` needs at least 2 levels, found {}",
                        q.criteria.len()
                    ));
                }
            }
            Question::Noul(q) => {
                if let Some(criteria) = &q.criteria {
                    if criteria.r#true.is_none() && criteria.r#false.is_none() {
                        return Err(
                            "`noul.criteria`, when present, must define `true` and/or `false`"
                                .to_string(),
                        );
                    }
                }
            }
        }
        let gold = self.gold_label()?;
        if !slots.contains(&gold) {
            return Err(format!(
                "gold {gold:?} is not one of this item's slots {slots:?}"
            ));
        }
        if let Some(positive) = self.positive_label() {
            if !slots.contains(&positive) {
                return Err(format!(
                    "positive {positive:?} is not one of this item's slots {slots:?}"
                ));
            }
        }
        Ok(())
    }
}

/// A loaded, hash-pinned item set.
#[derive(Debug, Clone, PartialEq)]
pub struct ItemSet {
    pub path: PathBuf,
    /// SHA-256 of the file's raw bytes. This is the frozen content hash.
    pub sha256: String,
    /// SHA-256 over `(id, rendered prompt)` of every item, in file order. Pins the
    /// prompt shape the evaluation was run against.
    pub prompt_set_sha256: String,
    pub items: Vec<Item>,
}

impl ItemSet {
    /// How many items carry each category, plus the count for each of the four
    /// required categories (0 if absent).
    pub fn category_counts(&self) -> std::collections::BTreeMap<String, usize> {
        let mut counts = std::collections::BTreeMap::new();
        for item in &self.items {
            *counts.entry(item.category.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// How many items carry each source.
    pub fn source_counts(&self) -> std::collections::BTreeMap<String, usize> {
        let mut counts = std::collections::BTreeMap::new();
        for item in &self.items {
            *counts.entry(item.source.clone()).or_insert(0) += 1;
        }
        counts
    }
}

/// Read + validate an item set and hash it.
pub fn load_item_set(path: &Path) -> Result<ItemSet, EvalError> {
    let display = path.display().to_string();
    let bytes = std::fs::read(path).map_err(|source| EvalError::Io {
        path: display.clone(),
        source,
    })?;
    let sha256 = sha256_hex(&bytes);
    let text = String::from_utf8(bytes).map_err(|_| EvalError::NotUtf8 {
        path: display.clone(),
    })?;

    let mut items: Vec<Item> = Vec::new();
    let mut first_line: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line_no = index + 1;
        let item: Item = serde_json::from_str(line).map_err(|err| EvalError::Json {
            path: display.clone(),
            line: line_no,
            message: err.to_string(),
        })?;
        item.validate().map_err(|message| EvalError::Item {
            path: display.clone(),
            line: line_no,
            id: item.id.clone(),
            message,
        })?;
        if let Some(previous) = first_line.insert(item.id.clone(), line_no) {
            return Err(EvalError::DuplicateItemId {
                path: display.clone(),
                id: item.id,
                first_line: previous,
            });
        }
        items.push(item);
    }

    if items.is_empty() {
        return Err(EvalError::EmptyItemSet { path: display });
    }

    let present: BTreeSet<&str> = items.iter().map(|item| item.category.as_str()).collect();
    let missing: Vec<String> = KNOWN_CATEGORIES
        .iter()
        .filter(|category| !present.contains(**category))
        .map(|category| (*category).to_string())
        .collect();
    if !missing.is_empty() {
        return Err(EvalError::MissingCategories {
            path: display,
            missing,
            required: KNOWN_CATEGORIES.iter().map(|c| (*c).to_string()).collect(),
        });
    }

    let prompt_set_sha256 = render_set_hash(&items)?;

    Ok(ItemSet {
        path: path.to_path_buf(),
        sha256,
        prompt_set_sha256,
        items,
    })
}

/// Hash `(id, rendered prompt)` for every item, in file order.
///
/// The item id is included so that reordering or relabelling the file is
/// detectable, even though ids themselves never enter a prompt (`specs/M0.md` §0.3).
fn render_set_hash(items: &[Item]) -> Result<String, EvalError> {
    let mut buffer = String::new();
    for item in items {
        let rendered = jev_core::render_question(&item.state, &item.question, &item.id).map_err(
            |source| EvalError::Render {
                id: item.id.clone(),
                source,
            },
        )?;
        buffer.push_str("--- item ");
        buffer.push_str(&item.id);
        buffer.push('\n');
        buffer.push_str(&rendered.prompt);
        buffer.push('\n');
    }
    Ok(sha256_hex(buffer.as_bytes()))
}

/// `eval/items/manifest.json`: the deliberate act of freezing a set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: String,
    pub sets: Vec<ManifestSet>,
}

/// One frozen set inside a [`Manifest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestSet {
    /// Repo-relative path, for humans.
    pub path: String,
    /// The recorded content hash. Any deviation makes `run` refuse.
    pub sha256: String,
    pub n_items: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Free-form lifecycle note, e.g. `"starter"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// The scale the design calls for, when this set is knowingly below it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_n_items: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self, EvalError> {
        let display = path.display().to_string();
        let bytes = std::fs::read(path).map_err(|source| EvalError::Io {
            path: display.clone(),
            source,
        })?;
        serde_json::from_slice(&bytes).map_err(|err| EvalError::FileJson {
            path: display,
            message: err.to_string(),
        })
    }

    /// Find the entry that pins `items_path`, matching on file name so that the
    /// check survives being run from a different working directory.
    pub fn entry_for(&self, items_path: &Path) -> Option<&ManifestSet> {
        let wanted = items_path.file_name()?;
        self.sets
            .iter()
            .find(|set| Path::new(&set.path).file_name() == Some(wanted))
    }
}

/// Refuse to compute when the item file no longer hashes to what the manifest
/// recorded.
pub fn check_manifest(
    manifest_path: &Path,
    items_path: &Path,
    actual_hash: &str,
) -> Result<(), EvalError> {
    let manifest = Manifest::load(manifest_path)?;
    let entry = manifest
        .entry_for(items_path)
        .ok_or_else(|| EvalError::ManifestEntryMissing {
            path: manifest_path.display().to_string(),
            items: items_path.display().to_string(),
        })?;
    if entry.sha256 != actual_hash {
        return Err(EvalError::HashMismatch {
            path: items_path.display().to_string(),
            expected: entry.sha256.clone(),
            actual: actual_hash.to_string(),
        });
    }
    Ok(())
}

/// Compare a hash against an explicit `--expect-sha256` value.
pub fn check_expected_hash(expected: &str, actual: &str, items_path: &Path) -> Result<(), EvalError> {
    if expected.eq_ignore_ascii_case(actual) {
        Ok(())
    } else {
        Err(EvalError::HashMismatch {
            path: items_path.display().to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn item(gold: Value, question: Value) -> Item {
        Item {
            id: "t-1".into(),
            category: "evidence_judgment".into(),
            source: "unit".into(),
            state: json!("state"),
            question: serde_json::from_value(question).expect("question parses"),
            gold,
            positive: None,
            provenance: None,
        }
    }

    #[test]
    fn choice_slots_are_the_sorted_criteria_keys() {
        let it = item(
            json!("yes"),
            json!({"type":"choice","instructions":"pick","criteria":{"yes":null,"no":null}}),
        );
        // BTreeMap order: "no" < "yes".
        assert_eq!(it.slot_labels(), vec!["no".to_string(), "yes".to_string()]);
        assert_eq!(it.gold_label().unwrap(), "yes");
        assert!(it.validate().is_ok());
    }

    #[test]
    fn score_slots_are_the_level_indices_and_gold_is_an_integer() {
        let it = item(
            json!(2),
            json!({"type":"score","instructions":"rate","criteria":["bad","ok","good"]}),
        );
        assert_eq!(
            it.slot_labels(),
            vec!["0".to_string(), "1".to_string(), "2".to_string()]
        );
        assert_eq!(it.gold_label().unwrap(), "2");
        assert!(it.validate().is_ok());
    }

    #[test]
    fn noul_slots_are_true_and_false_and_gold_is_a_boolean() {
        let it = item(
            json!(true),
            json!({"type":"noul","instructions":"does it hold?"}),
        );
        assert_eq!(
            it.slot_labels(),
            vec!["true".to_string(), "false".to_string()]
        );
        assert_eq!(it.gold_label().unwrap(), "true");
        assert_eq!(it.positive_label().unwrap(), "true");
        assert!(it.validate().is_ok());
    }

    #[test]
    fn validation_rejects_typed_mistakes_in_gold() {
        let choice = item(
            json!(1),
            json!({"type":"choice","instructions":"pick","criteria":{"a":null,"b":null}}),
        );
        assert!(choice.validate().unwrap_err().contains("string option label"));

        let score = item(
            json!("2"),
            json!({"type":"score","instructions":"rate","criteria":["x","y"]}),
        );
        assert!(score.validate().unwrap_err().contains("level index"));

        let noul = item(json!("true"), json!({"type":"noul","instructions":"x"}));
        assert!(noul.validate().unwrap_err().contains("boolean"));
    }

    #[test]
    fn validation_rejects_a_gold_that_is_not_a_declared_slot() {
        let it = item(
            json!("maybe"),
            json!({"type":"choice","instructions":"pick","criteria":{"yes":null,"no":null}}),
        );
        assert!(it.validate().unwrap_err().contains("not one of this item's slots"));
    }

    #[test]
    fn validation_rejects_a_positive_class_outside_the_slots() {
        let mut it = item(
            json!("yes"),
            json!({"type":"choice","instructions":"pick","criteria":{"yes":null,"no":null}}),
        );
        it.positive = Some("perhaps".into());
        assert!(it.validate().unwrap_err().contains("positive"));
    }

    #[test]
    fn validation_rejects_degenerate_option_counts() {
        let one = item(
            json!("a"),
            json!({"type":"choice","instructions":"pick","criteria":{"a":null}}),
        );
        assert!(one.validate().unwrap_err().contains("at least 2 options"));

        let flat = item(
            json!(0),
            json!({"type":"score","instructions":"rate","criteria":["only"]}),
        );
        assert!(flat.validate().unwrap_err().contains("at least 2 levels"));
    }

    #[test]
    fn a_positive_class_is_never_guessed_for_non_noul_items() {
        let it = item(
            json!("a"),
            json!({"type":"choice","instructions":"pick","criteria":{"a":null,"b":null}}),
        );
        assert_eq!(it.positive_label(), None);
    }
}
