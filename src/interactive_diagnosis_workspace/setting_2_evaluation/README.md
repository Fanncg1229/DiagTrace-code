# Setting 2 Trajectory Evaluation

Deterministic source-derived trajectory acquisition metrics for the diagnosis-aware
Setting 2 workflow.

The evaluator compares:

- gold: source-derived evidence categories from extracted case JSON files;
- prediction: LLM A's active query categories in Setting 2 episodes.

By default, `WAIT_REASSESS`, `FINAL_TRAJECTORY`, and `FINAL` are ignored. The
initial presentation is prepended as the first collapsed stage in both the gold
and prediction trajectories because Setting 2 gives it to LLM A before active
question asking begins.

Metrics:

- `weighted_tversky_alignment`: coverage with overuse/underuse penalties
  (`alpha=2`, `beta=1` by default);
- `weighted_lcs_recall`: how much of the source-derived category chain appears
  in the same order;
- `weighted_pairwise_order_accuracy`: how often matched category pairs preserve
  the source-derived order.

The primary taxonomy contains only categories that correspond to the local
workflow's clinical evidence acquisition actions. Nature/MIRA-style categories
such as `Plan` and `Search` are not primary Setting 2 actions here; requests of
that kind are mapped to `other_or_off_protocol` with low weight.

Example:

```bash
python -m interactive_diagnosis_workspace.setting_2_evaluation.run \
  --case-list-manifest interactive_diagnosis_workspace/dialogue_workflow/results/pilot15_v12_ddx_judge_alignment/_manifest.json \
  --episode-path interactive_diagnosis_workspace/trajectory_workflow/results/pilot3_setting2_v2 \
  --episode-path interactive_diagnosis_workspace/trajectory_workflow/results/pilot10_setting2_v1_merged \
  --case-dir interactive_diagnosis_workspace/dialogue_workflow/results/pilot3_v10_wait_after_stall/cases \
  --case-dir interactive_diagnosis_workspace/dialogue_workflow/results/pilot3_v10_extra7/cases \
  --case-dir interactive_diagnosis_workspace/dialogue_workflow/results/pilot5_v11_setting1_extra/cases \
  --output-dir interactive_diagnosis_workspace/setting_2_evaluation/results/pilot15_available_setting2
```
