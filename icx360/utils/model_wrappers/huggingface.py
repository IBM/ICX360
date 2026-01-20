"""
Wrapper for HuggingFace models.
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.

from math import ceil, log2

import torch

from icx360.utils.model_wrappers import GeneratedOutput, Model
from icx360.utils.toma import toma_generate


class HFModel(Model):
    """
    Wrapper for HuggingFace models.

    Attributes:
        _model (transformers model object):
            Underlying model object.
        _tokenizer (transformers tokenizer):
            Tokenizer corresponding to model.
        _device (str):
            Device on which the model resides.
    """
    def __init__(self, model, tokenizer):
        """
        Initialize HFModel wrapper.

        Args:
            model (transformers model object):
                Underlying model object.
            tokenizer (transformers tokenizer):
                Tokenizer corresponding to model.
        """
        super().__init__(model)
        self._tokenizer = tokenizer
        self._device = model.device

    def convert_input(self, inputs, chat_template=False, system_prompt=None, **kwargs):
        """
        Encode input text as token IDs for HuggingFace model.

        Args:
            inputs (str or List[str] or List[List[str]]):
                A single input text, a list of input texts, or a list of segmented texts.
            chat_template (bool):
                Whether to apply chat template.
            system_prompt (str or None):
                System prompt to include in chat template.
            **kwargs (dict):
                Additional keyword arguments for tokenizer.

        Returns:
            input_encoding (transformers.BatchEncoding):
                Object produced by tokenizer.
        """
        kwargs["return_tensors"] = "pt"
        if isinstance(inputs, list):
            # Batch of strings, enable padding and truncation
            kwargs["padding"] = True
            kwargs["truncation"] = True
            if isinstance(inputs[0], list):
                if chat_template:
                    # Join segmented strings
                    inputs = ["".join(inp) for inp in inputs]
                else:
                    # Indicate to tokenizer that strings are segmented
                    kwargs["is_split_into_words"] = True

        if chat_template:
            # Construct chat messages
            if isinstance(inputs, list):
                if system_prompt is not None:
                    messages = [[{"role": "system", "content": system_prompt},
                                 {"role": "user", "content": inp}] for inp in inputs]
                else:
                    messages = [[{"role": "user", "content": inp}] for inp in inputs]
            else:
                if system_prompt is not None:
                    messages = [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": inputs}]
                else:
                    messages = [{"role": "user", "content": inputs}]
            # Encode chat
            input_encoding = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, **kwargs).to(self._device)
        else:
            # Encode text
            input_encoding = self._tokenizer(inputs, **kwargs).to(self._device)

        return input_encoding

    def generate(self, inputs, chat_template=False, system_prompt=None, tokenizer_kwargs={}, text_only=True, **kwargs):
        """
        Generate response from model.

        Args:
            inputs (str or List[str] or List[List[str]]):
                A single input text, a list of input texts, or a list of segmented texts.
            chat_template (bool):
                Whether to apply chat template.
            system_prompt (str or None):
                System prompt to include in chat template.
            tokenizer_kwargs (dict):
                Additional keyword arguments for tokenizer.
            text_only (bool):
                Return only generated text (default) or an object containing additional outputs.
            **kwargs (dict):
                Additional keyword arguments for HuggingFace model.

        Returns:
            output_obj (List[str] or icx360.utils.model_wrappers.GeneratedOutput):
                If text_only == True, a list of generated texts corresponding to inputs.
                If text_only == False, a GeneratedOutput object containing the following:
                    output_ids: (num_inputs, output_token_count) torch.Tensor of generated token IDs.
                    output_text: List of generated texts.
                    output_token_count: Maximum number of generated tokens.
        """
        # Encode input text as token IDs
        inputs = self.convert_input(inputs, chat_template, system_prompt, **tokenizer_kwargs)
        num_inputs, input_length = inputs["input_ids"].shape

        if num_inputs == 1 or not torch.cuda.is_available():
            # Just use model.generate
            output_dict = self._model.generate(**inputs, return_dict_in_generate=True, **kwargs)
            output_ids = output_dict.sequences

        else:
            # Generate using toma
            # Set max_new_tokens if needed
            if "max_new_tokens" not in kwargs:
                kwargs["max_new_tokens"] = 32
            # Position where generated output starts (encoder-decoder output always begins with one special token?)
            gen_start = 1 if self._model.config.is_encoder_decoder else input_length
            # Pre-allocate output_ids Tensor
            output_ids = torch.full((num_inputs, kwargs["max_new_tokens"] + gen_start), self._tokenizer.pad_token_id, device=self._device)

            # Generate using toma
            batch_size_init = 2 ** ceil(log2(num_inputs))
            toma_generate(0, num_inputs, self._model, inputs, output_ids, toma_initial_step=batch_size_init, **kwargs)

        if not self._model.config.is_encoder_decoder:
            # For decoder-only models, truncate output tokens to response only
            output_ids = output_ids[:, input_length:]

        # Output text
        output_text = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        if text_only:
            return output_text
        else:
            # Output object
            output_token_count = output_ids.shape[1]
            output_obj = GeneratedOutput(output_ids=output_ids, output_text=output_text, output_token_count=output_token_count)

            return output_obj


class PipelineHFModel(HFModel):
    """
    HFModel-compatible wrapper around a SteeringPipeline.

    Attributes:
        _pipeline (aisteer360.algorithms.core.SteeringPipeline):
            AISteer360 SteeringPipeline object.
        _model (transformers model object):
            Underlying model object.
        _tokenizer (transformers tokenizer):
            Tokenizer corresponding to model.
        _device (str):
            Device on which the model resides.
        _runtime_kwargs (dict or None):
            Optional per-call parameters for controls at runtime.
    """

    def __init__(self, pipeline, tokenizer, runtime_kwargs: dict | None = None):
        """
        Initialize PipelineHFModel wrapper.

        Args:
            pipeline (aisteer360.algorithms.core.SteeringPipeline):
                AISteer360 SteeringPipeline object.
            tokenizer (transformers tokenizer):
                Tokenizer corresponding to model.
            runtime_kwargs (dict or None):
                Optional per-call parameters for controls at runtime.
        """
        super().__init__(pipeline.model, tokenizer)

        self._pipeline = pipeline
        self._runtime_kwargs = runtime_kwargs

    def generate(
        self,
        inputs,
        chat_template: bool = False,
        system_prompt: str | None = None,
        tokenizer_kwargs: dict = {},
        text_only: bool = True,
        **kwargs,
    ):
        """
        Generate response from SteeringPipeline.

        Args:
            inputs (str or List[str] or List[List[str]]):
                A single input text, a list of input texts, or a list of segmented texts.
            chat_template (bool):
                Whether to apply chat template.
            system_prompt (str or None):
                System prompt to include in chat template.
            tokenizer_kwargs (dict):
                Additional keyword arguments for tokenizer.
            text_only (bool):
                Return only generated text (default) or an object containing additional outputs.
            **kwargs (dict):
                Additional keyword arguments for pipeline.

        Returns:
            output_obj (List[str] or icx360.utils.model_wrappers.GeneratedOutput):
                If text_only == True, a list of generated texts corresponding to inputs.
                If text_only == False, a GeneratedOutput object containing the following:
                    output_ids: (num_inputs, output_token_count) torch.Tensor of generated token IDs.
                    output_text: List of generated texts.
                    output_token_count: Maximum number of generated tokens.
        """

        encoding = self.convert_input(
            inputs,
            chat_template=chat_template,
            system_prompt=system_prompt,
            **tokenizer_kwargs,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        input_length = input_ids.shape[1]

        runtime_kwargs = self._runtime_kwargs

        with torch.no_grad():
            output_ids = self._pipeline.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                runtime_kwargs=runtime_kwargs,
                **kwargs,
            )
        # SteeringPipeline already truncates output tokens to generated tokens only, don't truncate again

        output_text = self._tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if text_only:
            return output_text
        else:
            return GeneratedOutput(
                output_ids=output_ids,
                output_text=output_text,
                output_token_count=output_ids.shape[1],
            )
