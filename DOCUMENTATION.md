# Project context
Uncertainty-aware AI triage for teledermatology (final-year project). Classifies skin photos into urgency tiers (Urgent, Soon, Routine); low-confidence cases are escalated. Research prototype, not clinical.

# Rules
- Python, PyTorch. Pin versions in requirements.txt. Fix random seeds.
- Never commit raw images or datasets. data/ is gitignored.
- Splits must be by patient/lesion so no case appears in both train and test.
- DDI is an external test set only: never used for training or tuning.
- The taxonomy is a CSV: source_label, condition_group, urgency_tier, rationale, source. It is frozen before training. Do not invent clinical rationales or sources; leave them blank for me to fill in.
- Explain what you did after each step. Ask before large changes.

# Taxonomy tiering
Source guideline: NICE NG12, Suspected cancer: recognition and referral (last updated 15 April 2026), section 1.7 Skin cancers.

## Tier definitions
- Urgent: NG12 suspected cancer pathway referral.
- Soon: NG12 non-urgent referral.
- Routine: no referral recommended, or clearly benign.

## Default rule for conditions NG12 does not cover
- Clearly benign conditions -> Routine.
- Uncovered conditions that could be serious, or are unclear -> Soon, with confidence = low.

## Confidence
Rows not yet frozen are marked `provisional; <level>` in the confidence column, where level is clear, judgement call or low.

## Sensitivity analyses
Results will also be reported under two alternative mappings:
- Cautious: bcc and akiec -> Urgent.
- Relaxed: akiec -> Routine.

# Fitzpatrick17k near-duplicates
Fitzpatrick17k has no patient or lesion IDs, so near-duplicate groups are the only protection against the same image or lesion appearing in more than one split. Code: src/find_duplicates.py. Outputs: data/fitzpatrick/duplicates/.

## Signals
- Perceptual hash (64-bit DCT pHash), compared by Hamming distance: finds the same photo resized, recompressed or lightly edited.
- ImageNet ResNet-50 embeddings, compared by cosine similarity: finds crops, zooms, watermarked copies and other shots of the same lesion.
- File hash: no byte-identical files were found.

## Rule D
A pair of images is a duplicate if:
- cosine >= 0.92, or
- pHash distance <= 10 and cosine >= 0.80.

Duplicates are linked into groups (if A~B and B~C, then A, B and C share a group). The group is used as group_id in src/make_splits.py, so each group stays in one split. No images are removed and no labels are changed.

Result: 2,326 duplicate pairs; 3,725 images (22.5%) in 1,689 groups; largest group 80 images.

## Reasoning
- Chosen after visual review of sampled pairs in each score band. About 7 in 8 pairs were true duplicates at cosine 0.92-0.96, against about half at 0.90-0.92. A stricter cut (0.94) misses many real duplicates in the 0.92-0.94 band.
- pHash on its own gives false matches from distance 6, but every true pHash match reviewed had cosine above about 0.83, so pHash matches need cosine >= 0.80.
- Over-grouping is safe for leakage (unrelated images only share a split); under-grouping is not (the same lesion would be in train and test). The rule therefore errs towards grouping.
- Chaining creates a few large groups. The largest (80 images) is mostly hairy close-ups with mixed labels, not one lesion; it holds about 0.5% of the data and only affects split balance. A cosine cut of 0.90 produced a 295-image group, which is one reason for not going lower.
- Known gap: 61 pairs with pHash distance <= 4 but cosine < 0.80 are not joined by rule D. Visual review showed 60 of them are the same photo (mostly resized or with a different aspect ratio), and 22 of them fall in different splits. Not yet fixed; decision pending.

## Label conflicts found (not changed)
Among 298 near-certain duplicate pairs (pHash distance <= 4 or cosine >= 0.98):
- 24 have different condition labels (data/fitzpatrick/duplicates/condition_label_conflicts.csv). Many are subtype pairs, e.g. rhinophyma / rosacea and lentigo maligna / melanoma.
- 163 have different Fitzpatrick skin types (data/fitzpatrick/duplicates/skin_type_conflicts.csv): 121 differ by one type, 31 by two or more, and 11 involve an unknown type (-1). This indicates noise in the skin-type labels, which matters for fairness analysis.

# Uncertainty and calibration
- Escalation rule: a case whose uncertainty (softmax entropy, or 1 - max softmax) is above a threshold goes to the Urgent tier for human review, whatever tier was predicted.
- Temperature scaling uses a single scalar temperature fitted on the validation split.
- Fitting the temperature and evaluating calibration on the same validation split is optimistic. Validation results after scaling must be reported as such.
- The final temperature and escalation threshold must be chosen on the validation split and then applied unchanged to the test split. They must not be refitted or re-tuned on the test split.
