"""
Base class for scalarizers.

Scalarizers compute real-valued scalar outputs for text inputs or outputs of LLMs, for example by
comparing the inputs to a reference input or the corresponding outputs to a reference output.
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.

from abc import ABC, abstractmethod


class Scalarizer(ABC):
    """
    Base class for scalarizers.

    Attributes:
        model (icx360.utils.model_wrappers.Model or None):
            Generative model, wrapped in an icx360.utils.model_wrappers.Model object (optional, default None).
    """
    def __init__(self, model=None):
        """
        Initialize Scalarizer.

        Args:
            model (icx360.utils.model_wrappers.Model or None):
                Generative model, wrapped in an icx360.utils.model_wrappers.Model object (optional, default None).
        """
        self.model = model

    @abstractmethod
    def scalarize_output(self, inputs=None, outputs=None, ref_input=None, ref_output=None, **kwargs):
        """
        Compute scalar outputs.

        Args:
            inputs (str or List[str] or List[List[str]] or None):
                Inputs to compute scalar outputs for:
                A single input text, a list of input texts, or a list of segmented texts.
            outputs (str or List[str] or None):
                Outputs to scalarize (corresponding to inputs).
            ref_input (str or None):
                Reference input used to scalarize.
            ref_output (str or icx360.utils.model_wrappers.GeneratedOutput or None):
                Reference output (text or GeneratedOutput object) used to scalarize.
            **kwargs (dict):
                Additional keyword arguments.

        Returns:
            scalar_outputs ((num_inputs,) torch.Tensor):
                Scalar output for each input.
        """
        raise NotImplementedError
