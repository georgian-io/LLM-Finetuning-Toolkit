---
sidebar_position: 8
---

# Ablation

The `ablation` section controls the settings for ablation studies. It includes:

## Parameters

- `use_ablate`: Whether to perform ablation studies.
- `study_name`: The name of the ablation study.

:::tip
When `use_ablate` is set to true, the toolkit will generate multiple configurations by permuting the specified parameters. This allows you to easily compare different settings and their impact on the model's performance.
:::

## Example

```yaml
ablation:
  use_ablate: true
  study_name: "ablation_study_1"
```
