# datafun-06-ml

[![Workflow Guide](https://img.shields.io/badge/Pro--Guide-pro--analytics--02-green)](https://denisecase.github.io/pro-analytics-02/workflow-b-apply-example-project/)

[![Python 3.14](https://img.shields.io/badge/python-3.14%2B-blue?logo=python)](./pyproject.toml)

[![uv managed](https://img.shields.io/badge/uv-managed-DE5FE9)](https://docs.astral.sh/uv/)

[![ty type checked](https://img.shields.io/badge/ty-type_checked-2F80ED)](https://docs.astral.sh/ty/)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)

[![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

[![marimo](https://img.shields.io/badge/marimo-reactive_notebook-FF6B6B)](https://docs.marimo.io/)

[![Zensical docs](https://img.shields.io/badge/Zensical-docs-purple)](https://zensical.org/)

[![MIT](https://img.shields.io/badge/license-see%20LICENSE-yellow.svg)](./LICENSE)

> Professional Python project: linear regression and predictive analytics.

## Project Goal

This project introduces **linear regression**, the process of fitting a model to data and using it to make predictions.

Think about two variables that might be related:

- Does study time predict exam scores?
- Does temperature predict energy usage?
- Does advertising spend predict revenue?

The original project used the Palmer Penguins dataset to explore whether **bill length could be used to predict body mass**.

For this project modification, I changed the model to use **two features instead of one**:

- `bill_length_mm`
- `flipper_length_mm`

The target being predicted is:

- `body_mass_g`

My new question is:

> **Does using both bill length and flipper length improve the prediction of a penguin's body mass compared with using bill length alone?**

This modification uses multiple linear regression to determine whether combining two penguin measurements provides useful information for predicting body mass.

## My Project Modification

The original model used **bill length** as the only feature for predicting penguin body mass.

I modified the project to use both **bill length and flipper length** as features.

The model now uses:

```text
Features:
- Bill Length (mm)
- Flipper Length (mm)

Target:
- Body Mass (g)
