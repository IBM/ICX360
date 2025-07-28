"""
Base class for model wrappers and class for model-generated outputs.
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.


from abc import ABC, abstractmethod


class Model(ABC):
    """
    Base class for wrappers of different types of models.

    Attributes:
        _model: Underlying model object.
    """
    def __init__(self, model):
        """
        Initialize Model wrapper.

        Args:
            model: Underlying model object.
        """
        self._model = model

    def convert_input(self, inputs):
        """
        Convert input(s) as needed for the model type.

        Args:
            inputs (str or List[str] or List[List[str]]):
                A single input text, a list of input texts, or a list of segmented texts.

        Returns:
            inputs (type required by model):
                Converted inputs.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, inputs, text_only=True, **kwargs):
        """
        Generate response from model.

        Args:
            inputs (str or List[str] or List[List[str]]):
                A single input text, a list of input texts, or a list of segmented texts.
            text_only (bool):
                Return only generated text (default) or an object containing additional outputs.
            **kwargs (dict):
                Additional keyword arguments for model.

        Returns:
            output_obj (List[str] or icx360.utils.model_wrappers.GeneratedOutput):
                If text_only == True, a list of generated texts corresponding to inputs.
                If text_only == False, a GeneratedOutput object to hold outputs.
        """
        raise NotImplementedError


class GeneratedOutput:
    """
    Holds outputs of generate() method.

    Attributes:
        output_ids (torch.Tensor or None):
            Generated token IDs for each input.
        output_text (List[str] or None):
            Generated text for each input.
        output_token_count (int or None):
            Maximum number of generated tokens.
        logits (torch.Tensor or None):
            Output logits for each input.
    """
    def __init__(self, output_ids=None, output_text=None, output_token_count=None, logits=None):
        """
        Initialize GeneratedOutput.

        Args:
            output_ids (torch.Tensor or None):
                Generated token IDs for each input.
            output_text (List[str] or None):
                Generated text for each input.
            output_token_count (int or None):
                Maximum number of generated tokens.
            logits (torch.Tensor or None):
                Output logits for each input.
        """
        self.output_ids = output_ids
        self.output_text = output_text
        self.output_token_count = output_token_count
        self.logits = logits
