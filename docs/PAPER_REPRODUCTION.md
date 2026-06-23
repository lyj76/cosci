# Paper Reproduction Guide

This guide distinguishes a system reproduction from the small `paper-aml`
gold-set smoke test.

## What the AML experiment used

Supplementary Note 5.1 describes a constrained search over roughly 2300
approved drugs curated with the Open Targets Platform and 34 cancer types. The
pipeline generated a candidate-specific hypothesis, review, and mechanism of
action. It then combined:

1. Co-Scientist review score.
2. DepMap Q2 2024 target-dependency probability.
3. Expert oncologist review.
4. Wet-lab validation of selected candidates.

The supplement does not provide the exact 2300-drug input file or enough detail
to reconstruct how that list was serialized into each model request. Therefore,
an exact historical reproduction requires reconstructing the Open Targets input
and the matching DepMap snapshot.

## What this repository now supports

`@session` runs the Supervisor-driven Generation, Reflection, Ranking,
Evolution, Proximity, and Meta-review loop. `@pipeline` runs Generation only.

`--candidate-universe` accepts:

- A text file with one candidate per line.
- A CSV or TSV whose first column contains the candidate name.

Candidates are assigned to separate Generation tasks. The full list is not
inserted into one prompt. Constrained tasks do not semantically deduplicate
different candidates. Evolution is limited to candidate-preserving
feasibility/simplification passes so the session cannot leave the supplied
search space.

Reflection now verifies cited excerpts against fetched pages. Verification
status is carried into Ranking, Evolution, Meta-review, and the final report.
Ranking uses real alternating debate turns and repeats each verdict with the
hypothesis positions swapped; inconsistent verdicts are recorded but do not
update Elo.

To reconstruct the approved-drug universe from Open Targets, run:

```bash
python scripts/build_open_targets_aml_universe.py \
  --drug-molecule-dir /path/to/open-targets/platform/25.03/output/drug_molecule \
  --known-drug-dir /path/to/open-targets/platform/25.03/output/known_drug \
  --out data/inputs/approved_drugs_25.03.txt \
  --format txt
```

If you want the script to emit CSV with metadata for auditing, use
`--format csv`.

For `@session`, gold recall is measured only over the top-ranked `k` hypotheses
by internal Elo. Merely processing a gold drug somewhere in the input universe
does not count as a hit.

## Recommended validation sequence

Start with a small, blinded pilot list that includes known positives and
distractors:

```bash
co-scientist --config ./co-scientist.toml bench \
  --preset paper-aml \
  -c deepseek-session=openai_compatible:deepseek-v4-pro@session \
  --judge openai_compatible:deepseek-v4-flash \
  --candidate-universe data/inputs/aml-approved-drugs-pilot.txt \
  --n 20 \
  --matches 0 \
  --budget-per-candidate 10
```

After checking task completion, citations, review quality, ranking stability,
and cost, increase `--n` toward the full universe. A 2300-candidate run is not a
small benchmark and should not use the README's historical `$3` budget.

To test whether the agent system improves over the same underlying DeepSeek
model, run the same-model ablation:

```bash
co-scientist --config ./co-scientist.toml bench \
  --preset deepseek-aml-uplift \
  --judge openai_compatible:deepseek-v4-flash \
  --candidate-universe data/inputs/aml-approved-drugs-pilot.txt \
  --n 20 \
  --matches 4 \
  --budget-per-candidate 10 \
  --judge-budget 5
```

This compares `deepseek-v4-pro@session`, `deepseek-v4-pro@pipeline`, and
`deepseek-v4-pro@direct`. Use several runs or a larger match count; a single
two-match Elo result is not evidence of an agent uplift.

## Remaining gaps for an exact reproduction

- Exact Open Targets approved-drug list used by the paper.
- Mapping from each drug to known target genes and approved indications.
- DepMap Q2 2024 dependency probabilities and the paper's aggregation logic.
- The paper's full set of Generation strategies, including focus-area,
  conditional-hop, and raw-idea strategies.
- Expert review and wet-lab validation.

Until those inputs and stages are supplied, results should be described as a
paper-derived method reproduction, not an exact reproduction of the reported
AML experiment.
