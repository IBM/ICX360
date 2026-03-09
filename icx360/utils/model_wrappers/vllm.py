"""
Wrapper for VLLM models.
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.

from tqdm import tqdm

from icx360.utils.model_wrappers import GeneratedOutput, Model


class VLLMModel(Model):
    """
    Wrapper for VLLM models.

    Attributes:
        _model (OpenAI model object):
            Underlying model object.
        _model_name (str):
            Name of the model.
        _tokenizer (transformers tokenizer or None):
            HuggingFace tokenizer corresponding to the model (for applying chat template).
    """
    def __init__(self, model, model_name, tokenizer=None):
        """
        Initialize VLLMModel wrapper.

        Args:
            model (OpenAI model object):
                Underlying model object.
            model_name (str):
                Name of the model.
            tokenizer (transformers tokenizer or None):
                HuggingFace tokenizer corresponding to the model (for applying chat template).
        """
        super().__init__(model)
        self._model_name = model_name
        self._tokenizer = tokenizer

    def convert_input(self, inputs, chat_template=False, system_prompt=None, unit_ranges=None, **kwargs):
        """
        Convert input(s) into a list of strings.

        Args:
            inputs (str or List[str] or List[List[str]]):
                A single input text, a list of input texts, or a list of segmented texts.
            chat_template (bool):
                Whether to apply chat template.
            system_prompt (str or None):
                System prompt to include in chat template.
            unit_ranges (dict or None):
                Mapping from chat template parts to ranges of input units.

        Returns:
            inputs (List[str]):
                Converted input(s) as a list of strings.
        """
        if isinstance(inputs, str):
            # Single input text, convert to list
            inputs = [inputs]
        elif not isinstance(inputs, list):
            raise TypeError("Inputs must be a string or list for VLLMModel")

        if chat_template:
            if self._tokenizer is None:
                raise TypeError("HuggingFace tokenizer must be provided to apply chat template")

            if isinstance(inputs, list) and isinstance(inputs[0], list) and unit_ranges is not None:
                # Inputs are segmented into units and a mapping from chat template parts to units is given
                inputs = self._construct_chat_template_from_mapping(inputs, unit_ranges)
            else:
                if isinstance(inputs, list) and isinstance(inputs[0], list):
                    # Inputs are segmented into units but no mapping given, just join units
                    inputs = ["".join(inp) for inp in inputs]

                # Construct chat messages, placing each input into a single user message
                if system_prompt is not None:
                    messages = [[{"role": "system", "content": system_prompt},
                                {"role": "user", "content": inp}] for inp in inputs]
                else:
                    messages = [[{"role": "user", "content": inp}] for inp in inputs]

                # Apply chat template
                inputs = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            if isinstance(inputs, list) and isinstance(inputs[0], list):
                # Join segmented units
                inputs = ["".join(inp) for inp in inputs]

        return inputs

    def _construct_chat_template_from_mapping(self, inputs, unit_ranges):
        """
        Construct chat template given mapping from parts of the chat template to input units.

        Args:
            inputs (List[List[str]]):
                A list of input texts segmented into units.
            unit_ranges (dict):
                Mapping from chat template parts to ranges of input units.

        Returns:
            inputs_formatted (List[str]):
                List of inputs formatted according to chat template.
        """
        inputs_formatted = []
        # Iterate over inputs
        for inp in inputs:

            # Construct conversation turn by turn
            conversation = []
            for turn_ranges in unit_ranges["conversation"]:
                turn = {}
                for key, rng in turn_ranges.items():
                    # There should be only one range per turn
                    turn["role"] = key
                    turn["content"] = "".join(inp[rng[0] : rng[1]])
                conversation.append(turn)

            if "documents" in unit_ranges:
                # Construct documents
                documents = []
                for doc_id, doc_ranges in enumerate(unit_ranges["documents"]):
                    document = {"doc_id": doc_id + 1}
                    for key, rng in doc_ranges.items():
                        # Document text and possibly a title
                        document[key] = "".join(inp[rng[0] : rng[1]])
                    documents.append(document)
            else:
                documents = None

            # Construct chat template from conversation and documents
            input_formatted = self._tokenizer.apply_chat_template(conversation,
                                                                  documents=documents,
                                                                  add_generation_prompt=True,
                                                                  tokenize=False)
            inputs_formatted.append(input_formatted)

        return inputs_formatted

    def generate(self, inputs, chat_template=False, system_prompt=None, text_only=True, unit_ranges=None, **kwargs):
        """
        Generate response from model.

        Args:
            inputs (str or List[str] or List[List[str]]):
                A single input text, a list of input texts, or a list of segmented texts.
            chat_template (bool):
                Whether to apply chat template.
            system_prompt (str or None):
                System prompt to include in chat template.
            text_only (bool):
                Return only generated text (default) or an object containing additional outputs.
            unit_ranges (dict or None):
                Mapping from chat template parts to ranges of input units.
            **kwargs (dict):
                Additional keyword arguments for VLLM model.

        Returns:
            output_obj (List[str] or icx360.utils.model_wrappers.GeneratedOutput):
                If text_only == True, a list of generated texts corresponding to inputs.
                If text_only == False, a GeneratedOutput object containing the following:
                    output_text: List of generated texts.
        """
        # Convert input into list of strings if needed
        inputs = self.convert_input(inputs, chat_template, system_prompt, unit_ranges)

        # Generate output
        output_text = []
        # Iterate over generated outputs
        for result in tqdm(self._model.completions.create(model=self._model_name, prompt=inputs, **kwargs).choices, total=len(inputs)):
            output_text.append(result.text)

        if text_only:
            return output_text
        else:
            return GeneratedOutput(output_text=output_text)
