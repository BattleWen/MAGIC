---
license: mit
pretty_name: HumanEvalPack
language_creators:
- expert-generated
multilinguality:
- multilingual
language:
- code
tags:
- code
dataset_info:
- config_name: cpp
  features:
  - name: task_id
    dtype: string
  - name: prompt
    dtype: string
  - name: declaration
    dtype: string
  - name: canonical_solution
    dtype: string
  - name: buggy_solution
    dtype: string
  - name: bug_type
    dtype: string
  - name: failure_symptoms
    dtype: string
  - name: entry_point
    dtype: string
  - name: import
    dtype: string
  - name: test_setup
    dtype: string
  - name: test
    dtype: string
  - name: example_test
    dtype: string
  - name: signature
    dtype: string
  - name: docstring
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: test
    num_bytes: 469111
    num_examples: 164
  download_size: 193981
  dataset_size: 469111
- config_name: go
  features:
  - name: task_id
    dtype: string
  - name: prompt
    dtype: string
  - name: declaration
    dtype: string
  - name: canonical_solution
    dtype: string
  - name: buggy_solution
    dtype: string
  - name: bug_type
    dtype: string
  - name: failure_symptoms
    dtype: string
  - name: entry_point
    dtype: string
  - name: import
    dtype: string
  - name: test_setup
    dtype: string
  - name: test
    dtype: string
  - name: example_test
    dtype: string
  - name: signature
    dtype: string
  - name: docstring
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: test
    num_bytes: 463234
    num_examples: 164
  download_size: 198394
  dataset_size: 463234
- config_name: java
  features:
  - name: task_id
    dtype: string
  - name: prompt
    dtype: string
  - name: declaration
    dtype: string
  - name: canonical_solution
    dtype: string
  - name: buggy_solution
    dtype: string
  - name: bug_type
    dtype: string
  - name: failure_symptoms
    dtype: string
  - name: entry_point
    dtype: string
  - name: import
    dtype: string
  - name: test_setup
    dtype: string
  - name: test
    dtype: string
  - name: example_test
    dtype: string
  - name: signature
    dtype: string
  - name: docstring
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: test
    num_bytes: 589440
    num_examples: 164
  download_size: 210440
  dataset_size: 589440
- config_name: js
  features:
  - name: task_id
    dtype: string
  - name: prompt
    dtype: string
  - name: declaration
    dtype: string
  - name: canonical_solution
    dtype: string
  - name: buggy_solution
    dtype: string
  - name: bug_type
    dtype: string
  - name: failure_symptoms
    dtype: string
  - name: entry_point
    dtype: string
  - name: import
    dtype: string
  - name: test_setup
    dtype: string
  - name: test
    dtype: string
  - name: example_test
    dtype: string
  - name: signature
    dtype: string
  - name: docstring
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: test
    num_bytes: 435189
    num_examples: 164
  download_size: 194044
  dataset_size: 435189
- config_name: python
  features:
  - name: task_id
    dtype: string
  - name: prompt
    dtype: string
  - name: declaration
    dtype: string
  - name: canonical_solution
    dtype: string
  - name: buggy_solution
    dtype: string
  - name: bug_type
    dtype: string
  - name: failure_symptoms
    dtype: string
  - name: entry_point
    dtype: string
  - name: import
    dtype: string
  - name: test_setup
    dtype: string
  - name: test
    dtype: string
  - name: example_test
    dtype: string
  - name: signature
    dtype: string
  - name: docstring
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: test
    num_bytes: 423013
    num_examples: 164
  download_size: 191279
  dataset_size: 423013
- config_name: rust
  features:
  - name: task_id
    dtype: string
  - name: prompt
    dtype: string
  - name: declaration
    dtype: string
  - name: canonical_solution
    dtype: string
  - name: buggy_solution
    dtype: string
  - name: bug_type
    dtype: string
  - name: failure_symptoms
    dtype: string
  - name: entry_point
    dtype: string
  - name: import
    dtype: string
  - name: test_setup
    dtype: string
  - name: test
    dtype: string
  - name: example_test
    dtype: string
  - name: signature
    dtype: string
  - name: docstring
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: test
    num_bytes: 450539
    num_examples: 164
  download_size: 168464
  dataset_size: 450539
configs:
- config_name: cpp
  data_files:
  - split: test
    path: cpp/test-*
- config_name: go
  data_files:
  - split: test
    path: go/test-*
- config_name: java
  data_files:
  - split: test
    path: java/test-*
- config_name: js
  data_files:
  - split: test
    path: js/test-*
- config_name: python
  data_files:
  - split: test
    path: python/test-*
  default: true
- config_name: rust
  data_files:
  - split: test
    path: rust/test-*
---

![Octopack](https://github.com/bigcode-project/octopack/blob/31f3320f098703c7910e43492c39366eeea68d83/banner.png?raw=true)

# Dataset Card for HumanEvalPack

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
- [Additional Information](#additional-information)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Repository:** https://github.com/bigcode-project/octopack
- **Paper:** [OctoPack: Instruction Tuning Code Large Language Models](https://arxiv.org/abs/2308.07124)
- **Point of Contact:** [Niklas Muennighoff](mailto:n.muennighoff@gmail.com)

### Dataset Summary

> HumanEvalPack is an extension of OpenAI's HumanEval to cover 6 total languages across 3 tasks. The Python split is exactly the same as OpenAI's Python HumanEval. The other splits are translated by humans (similar to HumanEval-X but with additional cleaning, see [here](https://github.com/bigcode-project/octopack/tree/main/evaluation/create/humaneval-x#modifications-muennighoff)). Refer to the [OctoPack paper](https://arxiv.org/abs/2308.07124) for more details.
> 
- **Languages:** Python, JavaScript, Java, Go, C++, Rust
- **OctoPack🐙🎒:**

<table>
<tr>
<th>Data</t> 
<td><a href=https://huggingface.co/datasets/bigcode/commitpack>CommitPack</a></td>
<td>4TB of GitHub commits across 350 programming languages</td>
</tr>
<tr>
<th></t> 
<td><a href=https://huggingface.co/datasets/bigcode/commitpackft>CommitPackFT</a></td>
<td>Filtered version of CommitPack for high-quality commit messages that resemble instructions</td>
</tr>
<tr>
<th>Model</t> 
<td><a href=https://huggingface.co/bigcode/octocoder>OctoCoder</a></td>
<td>StarCoder (16B parameters) instruction tuned on CommitPackFT + OASST</td>
</tr>
<tr>
<th></t> 
<td><a href=https://huggingface.co/bigcode/octogeex>OctoGeeX</a></td>
<td>CodeGeeX2 (6B parameters) instruction tuned on CommitPackFT + OASST</td>
</tr>
<tr>
<th>Evaluation</t> 
<td><a href=https://huggingface.co/datasets/bigcode/humanevalpack>HumanEvalPack</a></td>
<td>Extension of OpenAI's HumanEval to cover 3 scenarios across 6 languages</td>
</tr>
</table>

## Usage

```python
# pip install -q datasets
from datasets import load_dataset
# Languages: "python", "js", "java", "go", "cpp", "rust"
ds = load_dataset("bigcode/humanevalpack", "python")["test"]
ds[0]
```

## Dataset Structure


### Data Instances


An example looks as follows:

```json
{
  "task_id": "Python/0",
  "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
  "declaration": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n",
  "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
  "buggy_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = elem - elem2\n                if distance < threshold:\n                    return True\n\n    return False\n",
  "bug_type": "missing logic",
  "failure_symptoms": "incorrect output",
  "entry_point": "has_close_elements",
  "import": ""
  "test_setup": ""
  "test": "\n\n\n\n\ndef check(has_close_elements):\n    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\ncheck(has_close_elements)",
  "example_test": "def check(has_close_elements):\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\ncheck(has_close_elements)\n",
  "signature": "has_close_elements(numbers: List[float], threshold: float) -> bool",
  "docstring": "Check if in given list of numbers, are any two numbers closer to each other than\ngiven threshold.\n>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\nFalse\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\nTrue",
  "instruction": "Write a Python function `has_close_elements(numbers: List[float], threshold: float) -> bool` to solve the following problem:\nCheck if in given list of numbers, are any two numbers closer to each other than\ngiven threshold.\n>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\nFalse\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\nTrue"
}
```

### Data Fields

The data fields are the same among all splits:
- `task_id`: Indicates the language (Python/JavaScript/Java/Go/C++/Rust) and task id (from 0 to 163) of the problem
- `prompt`: the prompt for models relying on code continuation
- `declaration`: the declaration of the function (same as prompt but without the docstring)
- `canonical_solution`: the correct solution passing all unit tests for the problem
- `buggy_solution`: same as `canonical_solution` but with a subtle human-written bug causing the unit tests to fail
- `bug_type`: the type of the bug in `buggy_solution` (one of [`missing logic`, `excess logic`, `value misuse`, `operator misuse`, `variable misuse`, `function misuse`])
- `failure_symptoms`: the problem the bug causes (one of [`incorrect output`, `stackoverflow`, `infinite loop`])
- `entry_point`: the name of the function
- `import`: imports necessary for the solution (only present for Go)
- `test_setup`: imports necessary for the test execution (only present for Go)
- `test`: the unit tests for the problem
- `example_test`: additional unit tests different from `test` that could be e.g. provided to the model (these are not used in the paper)
- `signature`: the signature of the function
- `docstring`: the docstring describing the problem
- `instruction`: an instruction for HumanEvalSynthesize in the form `Write a {language_name} function {signature} to solve the following problem:\n{docstring}`

## Citation Information

```bibtex
@article{muennighoff2023octopack,
      title={OctoPack: Instruction Tuning Code Large Language Models}, 
      author={Niklas Muennighoff and Qian Liu and Armel Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro von Werra and Shayne Longpre},
      journal={arXiv preprint arXiv:2308.07124},
      year={2023}
}
```