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

# Uncertainty and calibration
- Escalation rule: a case whose uncertainty (softmax entropy, or 1 - max softmax) is above a threshold goes to the Urgent tier for human review, whatever tier was predicted.
- Temperature scaling uses a single scalar temperature fitted on the validation split.
- Fitting the temperature and evaluating calibration on the same validation split is optimistic. Validation results after scaling must be reported as such.
- The final temperature and escalation threshold must be chosen on the validation split and then applied unchanged to the test split. They must not be refitted or re-tuned on the test split.
