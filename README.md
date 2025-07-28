![ICX logo wide](docs/assets/icx360_logos/png/darkmode_icx_wide.png#gh-dark-mode-only)
![ICX logo wide](docs/assets/icx360_logos/png/lightmode_icx_wide.png#gh-light-mode-only)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0)  [![](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) [![Discussions](https://img.shields.io/badge/Discussions-Join%20the%20Conversation-blue)](https://github.com/IBM/ICX360/discussions) [![Issues](https://img.shields.io/github/issues/IBM/ICX360)](https://github.com/IBM/ICX360/issues) [![Docs](https://img.shields.io/badge/Docs-View%20Documentation-blue)](https://ibm.github.io/ICX360/)

## Overview

This toolkit provides in-context explanations for LLMs - explanations of the output of an LLM in terms of parts of the input context given to the LLM. It will be useful for both researchers and practitioners who want to understand the reason for a particular generation with respect to the context.

The toolkit features explanation methods, quick start and elaborate example notebooks, tests, and documentation. The quick start notebooks can also be run in Google Colab.


## Methods

1. **Multi-Level Explanations for Generative Language Models**: Explains generated text by attributing to parts of the input context and quantifying the importance of these parts to the generation. [![Read Paper](https://img.shields.io/badge/Read%20Paper-PDF-yellow)](https://arxiv.org/pdf/2403.14459) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IBM/ICX360/blob/main/examples/mexgen/quick_start.ipynb)

2. **CELL your Model: Contrastive Explanations for Large Language Models**: Explains text generation by generating contrastive prompts (i.e., edited version of the input prompt) that elicit responses that differ from the original response according to a pre-defined score. [![Read Paper](https://img.shields.io/badge/Read%20Paper-PDF-yellow)](https://arxiv.org/pdf/2406.11785) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IBM/ICX360/blob/main/examples/cell/quick_start.ipynb)

3. **Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for Large Language Models**: Explains potential jailbreak threats by highlighting important prompt tokens based on model gradients. [![Read Paper](https://img.shields.io/badge/Read%20Paper-PDF-yellow)](https://arxiv.org/pdf/2412.18171) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IBM/ICX360/blob/main/examples/th/quick_start.ipynb)



## Prerequisites

The toolkit can be installed locally using the instructions below. Please ensure sufficient resources (such as GPUs) are available for running the methods.

The toolkit uses [uv](https://docs.astral.sh/uv/) as the package manager (Python 3.11). Make sure that `uv` is installed via either:

```curl -Ls https://astral.sh/uv/install.sh | sh```

or using [Homebrew](https://brew.sh):

```brew install astral-sh/uv/uv```

or using pip (use this if in Windows):

```pip install uv```

## Installation

Once `uv` is installed, in Linux or Mac, clone the repo:

```commandline
git clone git@github.com:IBM/ICX360.git icx360
cd icx360
```

Ensure that you are inside the `icx360` directory (where `README.md` is located) and run:
```commandline
uv venv --python 3.12
source .venv/bin/activate
uv pip install .
```

Or in Windows, run:

```commandline
uv venv --python 3.12
.venv/bin/activate
uv pip install .
```

The package has been tested on `Red Hat Enterprise Linux 9`.


##  Quickstart Examples
1. [MExGen Quick Start](examples/mexgen/quick_start.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IBM/ICX360/blob/main/examples/mexgen/quick_start.ipynb)

2. [CELL Quick Start](examples/cell/quick_start.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IBM/ICX360/blob/main/examples/cell/quick_start.ipynb)

3. [Token Highlighter Jailbreak Inspector Quick Start](examples/th/quick_start.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IBM/ICX360/blob/main/examples/th/quick_start.ipynb)

Find many more examples [here](examples/index.md).

## Tests

We have included a collection of tests that can be used for checking if the installation worked properly. This can be achieved by running the "non-slow" tests:
```
pytest -m "not slow"
```
Note that these only test the plumbing and not any realistic functionality.

Those interested in testing the realistic functionality can run the notebooks in the `./examples/` folder or run the "slow and not vllm" tests. These use Hugging Face models that are large enough to meaningfully perform their tasks. Please ensure you have sufficient infrastructure (e.g. GPUs) before running these. Run the "slow and not vllm" tests using:
```
pytest -m "slow and not vllm"
```

Finally, if you also have a VLLM model that you would like to test, first enter its parameters in `model_catalog` in `tests/conftest.py` and then run:
```
pytest -m "vllm"
```

## License

ICX360 is provided under [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

## Contributing

- Get started by checking our [contribution guidelines](CONTRIBUTING.md).
- If you have any questions, just ask!

## Get involved

Lets form a community around this toolkit! Ask a question, raise an issue, or express interest to contribute in the [discussions page.](https://github.com/IBM/ICX360/discussions)

## IBM ❤️ Open Source AI

The first release of ICX360 has been brought to you by IBM in the hope of building a larger community around this topic.
