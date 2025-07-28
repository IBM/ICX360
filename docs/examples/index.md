# Examples

## Quickstart
- [MExGen Quick Start](mexgen/quick_start.ipynb): MExGen is used to attribute an importance score to each sentence in the input document that is summarized by an LLM.
- [CELL Quick Start](cell/quick_start.ipynb) - CELL is used to generate a modified input prompt for an LLM that will result in a response that is contrasted to the original response.
- [Token Highlighter Jailbreak Inspector Quick Start](th/quick_start.ipynb) - Token Highlighter is used to compute the importance of sentences in the input prompt to an LLM contributing to affirmative responses, with demonstrations to inspect jailbreak prompts.

## MExGen
- [MExGen for Question Answering](mexgen/question_answering.ipynb) - MExGen is used  to explain an LLM's response to a question in terms of a document provided as context to the LLM. The explanation consists of sentence-level and mixed word- and sentence-level attributions.
- [MExGen for Retrieval Augmented Generation](mexgen/RAG.ipynb) - MExGen is used to explain an LLM's response to a question in retrieval-augmented generation (RAG). The explanation shows which retrieved documents and which parts thereof are most important to the LLM.
- [MExGen for Summarization](mexgen/summarization.ipynb) - MExGen is used to explain an LLM's summarization of a document using both sentence-level and mixed word-, sentence-level attributions.

## CELL
- [CELL for Natural Language Generation](cell/natural_language_generation.ipynb) - CELL and mCELL (a myopic version of CELL that is more expensive) are used to generate a modified input prompt for LLMs that will result in a response that is contrasted to the original response. This is demonstrated with a small and a larger LLM.

## Token Highlighter
- [Token Highlighter Jailbreak Inspector](th/LLM_jailbreak.ipynb) - Token Highlighter is used to identify important segments in the input prompt (tokens/words/sentences) contributing to affirmative responses, with demonstrations to inspect jailbreak prompts
