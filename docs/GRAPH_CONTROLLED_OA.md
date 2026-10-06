# OA graph-controlled experiment, v1

## Foreground commands

Use the project Python environment, from the repository root:

```bash
python -u scripts/run_controlled_oa.py --output-dir data/runs/controlled_local/oa_graph_01
```

Defaults: the same three biomedical PDFs specified in the runner, one local
evidence-gap query, at most one revision round, parallelism 2, no Elo,
600-second phase limit and 1800-second limit for the current invocation's timed
phases. Progress prints every 30 seconds while a phase is waiting.

After interruption:

```bash
python -u scripts/run_controlled_oa.py --output-dir data/runs/controlled_local/oa_graph_01 --resume
```

To check resumability without any model call or manifest change:

```bash
python -u scripts/run_controlled_oa.py --output-dir data/runs/controlled_local/oa_graph_01 --check-resume
```

Structured responses are validated before graph mutation. Empty content, truncated
output, malformed JSON and schema mismatch get at most one explicit retry;
refusals and request failures stop. The output budget is initially 8192 and rises
to 12288 only for empty/truncated responses. SDK automatic retries are disabled
for these calls. Per-call timeout remains 180s, inside the enclosing phase budget.
`checkpoints/structured_diagnostics_*.json` preserves visible content, finish reason,
token usage (when supplied), validation errors and attempt status; reasoning is
recorded only as character count. Exhausted retries do not generate fake evidence
or close a causal gap. A cached retrieval can be reused after the error.

The known pre-fix `oa_graph_01` run can be resumed across this structured-output
repair: compatibility is checked against the exact previous two source hashes,
unchanged corpus/parameters/other agent code, and only checkpoints preceding
attribution. The original manifest is backed up and the migration is recorded.
This does not allow arbitrary code changes or reuse of completed downstream
graph results from another implementation.

If a phase needs longer, keep the same inputs and extend time limits:

```bash
python -u scripts/run_controlled_oa.py --output-dir data/runs/controlled_local/oa_graph_01 --resume --phase-timeout 900 --overall-timeout 2700
```

Completed extraction, generation, initial review, graph-query results,
attributions, post-retrieval reviews, revision rounds, experiment proposal and
individual PI branches are cached. An unfinished operation is retried on
restart; this is not automatic offline waiting. The PaperQA index is rebuilt
lazily only if another uncached query is needed. Extraction subqueries and
individual branches within an incomplete revision round are not yet cached
separately. Reporting and session export follow the timed phases.

The manifest fingerprints PDF bytes, objective, controller parameters and
relevant local code. Changed inputs/code require a new output directory. Old
`oa_run_02` artifacts are useful audit material, not valid v1 resume inputs.
Checkpoints currently include local Python pickle objects as well as readable
JSON; resume only this program's own local run directories.

## What actually controls execution

1. Generate mechanisms and obtain the initial Skeptic review.
2. Build strategy/step-specific causal edges. Generator citation strings alone
   do not establish support. Include extracted caveats as evidence nodes.
3. Prioritize unresolved steps flagged as contradicted/unsupported by Skeptic,
   then unsupported-core and fatal reviews. This priority is a heuristic,
   **not measured expected information gain**.
4. Query the same in-memory PaperQA corpus. Cache the raw result before the
   attribution call. Identical queries can reuse a cached result.
5. Attribute each retrieved passage against the full causal claim and research
   context: supports / contradicts / entity_only / context_only / unknown.
   Support and contradiction need a verbatim quote, a citation with page data
   and an affirmative scope judgement. Conflicting evidence keeps the edge
   unresolved. These are model judgements, not verified scientific truth.
6. Send the audit and passages to the relevant branch, and re-review only
   branches that received passages. Exhausted query budgets leave gaps open.
7. Skip revision for passing reviews. Otherwise revise with the audit context,
   re-review, and retain the parent unless the review-burden vector has no
   regression and at least one improvement. This is a review proxy; it does
   not establish a biological improvement.
8. Rebuild the selected-version graph; only exactly unchanged edge identities
   retain the old attribution. Propose one experiment comparing actual
   candidate IDs with different predictions and outcome-dependent decisions.
9. PI receives branch evidence/caveats, audit, uncertainty and the experiment
   proposal. Final PI prose has not yet undergone a separate final audit.

This is a bounded graph-controlled episode in the **OA entry point**, not a
replacement of the legacy SQLite Supervisor. It uses the existing OA model
client for attribution/experiment calls; unified provider-cost accounting is
not yet wired into these calls. PDFs are the only retrieval corpus; model and
embedding calls can still use remote APIs.

## Artifacts to inspect

- `checkpoints/manifest.json`: corpus hashes and experiment signature.
- `checkpoints/graph_episode.json`: incremental graph plus completed query trace.
- `checkpoints/retrieval-*.json`: full retrieved passages and generated answer.
- `checkpoints/graph_control_trace.json`: chosen claims, quote checks, scope
  decisions, unresolved counts and why retrieval stopped.
- `checkpoints/evolution_round_*.json`: proposed revisions and reviews, including
  revisions that were rejected; accepted state is in `evolved_state.json`.
- `checkpoints/selected_version_graph.json`: selected mechanism version, including
  proposed experiment edges; no experimental results are implied.
- `checkpoints/discriminating_experiment.json`: intervention, controls,
  per-candidate predictions and next decisions.
- `oa_discovery_report.json`: graph traces, revision acceptance decisions and
  final proposals. Ranking scores are heuristic, not Elo or truth probabilities.

## Comparison protocol

For a current-code ablation with graph interventions disabled:

```bash
python -u scripts/run_controlled_oa.py --output-dir data/runs/controlled_local/oa_graph_off_01 --graph-queries 0
```

This disables graph retrieval and the dedicated discriminating-experiment call;
it is not an exact replay of the historical code. Compare identical corpus
hashes/objective/model and repeat runs before claiming a capability increase.
Inspect quote validity, disease/species/cell/compartment scope, preservation of
negative results, unsupported-core findings, rejected revisions, experimental
discriminability, latency and model-call cost. Do not optimize merely for fewer
unresolved edges: a correctly retained uncertainty is better than false support.

Verification in this implementation uses offline model/retriever stubs and a
socket-blocking fixture, including a PI interruption/resume scenario. A real
v1 model run is still required to measure scientific quality.
