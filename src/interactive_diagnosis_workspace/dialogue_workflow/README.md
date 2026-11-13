# Dialogue Workflow — Setting 1

This directory contains the clean two-LLM workflow. It is intentionally separate
from `clean_workflow/` and the older controller-based implementation.

## Research setting

- **LLM A — Clinician:** sees the initial presentation and prior report replies,
  asks focused clinical questions, maintains a ranked DDx, and submits `FINAL`.
- **LLM B — Hospital/Report:** sees all source-extracted longitudinal PMC evidence,
  decides the current clinical phase and pending/available status, and selects the
  single record that answers the clinician's request.
- **Thin runner:** counts turns, keeps A and B contexts separate, validates the ID
  selected by B, records B's phase/pending decisions, and returns stored source text
  verbatim. It does not make medical or timeline decisions itself.

The initial presentation starts at the first encounter where diagnostic workup begins
for the still-undiagnosed condition. Earlier relevant events remain available as prior
history, but an incidental antecedent event is not used as the starting encounter merely
because it appears first chronologically.

The final diagnosis and hidden reference must not be included in either model's
prompt. They are used only by an offline evaluator after the episode.

## Actions

```text
ASK_HISTORY
REQUEST_EXAM
ORDER_TEST
REQUEST_RESULT
ORDER_PROCEDURE
WAIT_REASSESS
FINAL
```

`REQUEST_RESULT` covers laboratories, imaging reports, pathology, molecular tests,
microbiology, and text findings extracted from article images. Setting 1
does not provide raw image pixels.

`WAIT_REASSESS` covers longitudinal clinical updates that a clinician would normally
learn during care, including hospitalization, ICU transfer, procedures, new symptoms,
complications, thrombosis, treatment response, and treatment failure. It is not limited
to pending test results. The Clinician is prompted to use `WAIT_REASSESS` after repeated
unavailable-result replies or after a local/intermediate diagnosis when the longitudinal
course remains unknown.

## Verbatim report responses

LLM B returns a status and `evidence_id`; it does not compose or paraphrase the answer.
Normally the runner sends the stored `source_text` unchanged to LLM A. When one source
record mixes a legitimate clinical result with retrospective diagnostic narration, B
may select a strict contiguous `verbatim_excerpt`, which the runner validates against
the stored source before returning it. A diagnosis-revealing record cannot be returned
in full.

Article titles and retrospective discussion/conclusion text remain unavailable as
answers. Contemporaneous measurements, positive/negative test results, mutations,
morphology, and formal report findings remain requestable even when their source
sentence ends with an author-written diagnostic conclusion.

## Timeline rule

Timeline and phase metadata guide LLM B. There is no deterministic medical phase
controller: B decides which source record is responsive and available. The runner only
enforces state consistency. Placing an order or marking it pending does not advance the
phase; returning a later source record does.

If an order has no result in the source, B returns `ordered_but_not_documented` rather
than creating a pending result that can never resolve. This status means the article
does not document the result; it is not a delayed or pending result.

- If A vaguely asks for a repeated measurement, B selects the earliest relevant,
  undisclosed result available in the current phase.
- Later results are returned after B decides that the dialogue has reached that phase.
- An explicit future-timepoint request can be marked `pending` once rather than leaked,
  but a later matching result request or reassessment must release the source-backed
  result.
- If no result is pending, `WAIT_REASSESS` releases the earliest undisclosed
  source-grounded clinical course update at or after the current phase.

Example:

```text
A: What did the blood tests show?
B: [verbatim pre-admission blood-test evidence]

A: Were blood tests repeated later?
B: [verbatim later/ICU blood-test evidence]
```

Every B response logs its phase and pending orders. B must be fixed across compared
clinician models because its phase decisions are part of the evaluation environment.

The default and intended benchmark budget is 25 total clinician turns. A voluntary
`FINAL` ends the episode early. On the final turn, the clinician is prompted again if needed
and must submit its current most likely diagnosis with diagnostic reasoning.

## Minimal usage

```python
from interactive_diagnosis_workspace.dialogue_workflow import DialogueWorkflow

workflow = DialogueWorkflow(
    case_data,
    clinician_client=clinician_client,
    report_client=report_client,
    clinician_model="clinician-deployment",
    report_model="report-deployment",
    max_turns=25,
)
episode = workflow.run_episode()
```

Use separate model deployments when possible. Only LLM A is the evaluated model;
LLM B should remain fixed across experiments.

## GPT-4.1 extraction

`extraction.py` builds the case JSON consumed by this workflow from PMC NXML or
plain article text. It uses the Azure credentials already loaded from `.env` and
selects the extraction deployment in this order:

```text
explicit model argument
AZURE_OPENAI_EXTRACTION_DEPLOYMENT
AZURE_FOUNDRY_EXTRACTOR_DEPLOYMENT
AZURE_OPENAI_DEPLOYMENT_GPT4
gpt-4.1
```

Run it on an NXML file:

```bash
python -m interactive_diagnosis_workspace.dialogue_workflow.extraction \
  path/to/article.nxml \
  --case-id PMC12345 \
  --output interactive_diagnosis_workspace/dialogue_workflow/data/PMC12345.json
```

The extractor requires every evidence `source_text` to match a contiguous article
span after whitespace normalization. It rejects non-verbatim model output and an
initial presentation containing the final diagnosis. Repeated measurements receive
a shared `repeat_key`; textual figure findings are marked `image_finding=true`, and
no image pixels or paths are included.

## End-to-end batch runner

`runner.py` connects extraction, dialogue, evaluation, result persistence, and
batch summaries. It writes an atomic checkpoint after every completed turn and can
resume an interrupted episode. Independent cases can run concurrently.

```bash
python -m interactive_diagnosis_workspace.dialogue_workflow.runner \
  --case-id PMC10362754 \
  --case-id PMC10367410 \
  --case-id PMC10548549 \
  --input-root internalmed \
  --run-dir interactive_diagnosis_workspace/dialogue_workflow/results/pilot3_v2 \
  --clinician-model o4-mini \
  --report-model gpt-4.1 \
  --max-turns 25 \
  --workers 3
```

Each run contains:

```text
cases/          GPT-4.1 extractions
partials/       per-turn checkpoints, removed after successful completion
episodes/       complete A/B dialogues and token usage
evaluations/    per-case diagnosis and strategy metrics
errors/         structured failures
_manifest.json  models, protocol, case status, and paths
_summary.json   aggregate metrics
```

`--rerun-episodes` reuses extracted cases but reruns model dialogues.
`--case-cache-dir` imports extractions from another run. Resume rejects existing
episodes whose clinician model, report model, or turn budget differs from the
requested configuration.

The deterministic evaluator reports diagnosis/alias match, confidence, Brier score,
turns, action counts, Report status counts, evidence yield, phase trajectory,
supporting-evidence validity, and token use. Diagnostic reasoning is retained and
checked for presence but is not assigned a subjective LLM-judge score.

## Three-case pilot

The consistent pilot with GPT-4.1 extraction, o4-mini Clinician, GPT-4.1 Report,
and a 20-turn budget is stored under `results/pilot3_v2/`. It completed all three
cases. The small pilot is diagnostic rather than a benchmark result: Report selection
errors and initial-presentation choices materially affected the clinician trajectory,
so Report accuracy and judge sensitivity must be measured before scaling.
