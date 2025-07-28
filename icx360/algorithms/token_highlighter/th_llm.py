"""
    Class for TokenHilighter explainer (TH-LLM).
    Interpreting LLMs based on the importance analysis of input text units.
"""
from typing import List, Tuple

import torch

from icx360.algorithms.lwbe import LocalWBExplainer
from icx360.utils.segmenters import SpaCySegmenter, exclude_non_alphanumeric


class TokenHighlighter(LocalWBExplainer):
    """
        Class for TokenHilighter explainer (TH-LLM).
        Interpreting LLMs based on the importance analysis of input text units.
    """

    def __init__(self, model, tokenizer, segmenter, **kwargs):
        """
        Initialize the TH-LLM explainer.

        Args:
            model: The large language model object.
            tokenizer: The tokenizer object.
            segmenter: The segmenter object.
            affirmation: The affirmation sentence template.
            pooling: The aggregation method ("norm_mean", "mean_norm", or "matrix").
        """

        self.m = model
        self.tok = tokenizer
        self.segmenter = SpaCySegmenter(segmenter)
        affirmation: str = kwargs.get("affirmation", "Sure, I'd like to help you with this.")
        self.pooling: str = kwargs.get("pooling", "mean_norm")

        input_slot = "<slot_for_user_input_design_by_xm>"
        affirmation_slot = "<slot_for_model_affirmation_design_by_xm>"

        sample_input = self.tok.apply_chat_template(
            [
                {"role": "user", "content": input_slot},
                {"role": "assistant", "content": affirmation_slot}
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        input_start_id = sample_input.find(input_slot)
        affirmation_start_id = sample_input.find(affirmation_slot)

        prefix = sample_input[:input_start_id]
        infix = sample_input[input_start_id + len(input_slot):affirmation_start_id]  # infix
        suffix = sample_input[affirmation_start_id + len(affirmation_slot):]  # suffix

        self.token_ids=self._get_token_ids(prefix,infix,affirmation,suffix)

    def _get_token_ids(self,prefix,infix,affirmation,suffix):
        prefix_ids = self.tok(
            prefix, add_special_tokens=True, return_tensors="pt"
        ).input_ids[0]

        infix_ids = self.tok(
            infix, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]

        affirmation_ids = self.tok(
            affirmation, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]

        suffix_ids = self.tok(
            suffix, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]

        return {
            "prefix_ids":prefix_ids,
            "infix_ids":infix_ids,
            "affirmation_ids":affirmation_ids,
            "suffix_ids":suffix_ids
        }

    def explain_instance(
        self, input_orig, unit_types, ind_segment, segment_type, **kwargs
    ):
        """
        Compute importance scores for each text unit.

        Args:
            input_orig (str): Original input text.
            unit_types (Union[str, List[str]]): Type(s) of each text unit.
            ind_segment (Union[bool, List[bool]]): Whether to segment.
            segment_type (str): Type of segmentation to apply.
            max_phrase_length (int): Max length allowed for a phrase.

        Returns:
            Dict[str, Any]: Attribution information dictionary.
        """
        max_phrase_length: int = kwargs.get("max_phrase_length", 10)
        if isinstance(ind_segment, bool):
            ind_segment = [ind_segment]

        if isinstance(input_orig, str) or any(ind_segment):
            input_orig, unit_types, _ = self.segmenter.segment_units(
                input_orig, ind_segment, unit_types,
                segment_type=segment_type,
                max_phrase_length=max_phrase_length
            )

            unit_types = exclude_non_alphanumeric(unit_types, input_orig)

        num_units = len(input_orig)
        if isinstance(unit_types, str):
            unit_types = [unit_types] * num_units

        if self.pooling == "norm_mean":
            units, unit_scores = self.explain_instance_norm_mean(input_orig)
        elif self.pooling == "mean_norm":
            units, unit_scores = self.explain_instance_mean_norm(input_orig)
        elif self.pooling == "matrix":
            units, unit_scores = self.explain_instance_matrix(input_orig)
        else:
            units, unit_scores = self.explain_instance_mean_norm(input_orig)

        coef = {
            "unit_types": unit_types,
            "units": units,
            "scores": unit_scores
        }

        return {"attributions": coef}

    def _compute_gradient(self,full_id,prompt_ids):
        full_embedding = self.m.get_input_embeddings()(full_id.to(self.m.device))
        prompt_embedding = self.m.get_input_embeddings()(prompt_ids.to(self.m.device))

        prompt_embeds = prompt_embedding.detach().unsqueeze(0)
        prompt_embeds.requires_grad_()
        embeds = full_embedding.detach()

        full_embeds = torch.cat([
            embeds[:, :len(self.token_ids["prefix_ids"]), :],
            prompt_embeds,
            embeds[:, len(self.token_ids["prefix_ids"]) + len(prompt_ids):, :]
        ], dim=1)

        logits = self.m(inputs_embeds=full_embeds).logits
        targets_start = sum(
            len(self.token_ids[key]) for key in ["prefix_ids","infix_ids"]
        )
        targets_start += len(prompt_ids)
        targets_end = targets_start + len(self.token_ids["affirmation_ids"])
        targets = full_id[0, targets_start:targets_end].to(self.m.device)

        loss = torch.nn.CrossEntropyLoss()(
            logits[0, targets_start - 1:targets_end - 1, :], targets
        )
        loss.backward()

        return prompt_embeds.grad
        #[0].norm(dim=1)

    def explain_instance_norm_mean(self, units: List[str]) -> Tuple[List[str], List[float]]:
        """
        Use the L2 norm of the average of token gradients as the importance score for each unit.

        Args:
            units (List[str]): A list of text units (e.g., phrases or words) that form the prompt.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - The list of units.
                - The unit scores based on the norm_mean method.
        """
        unit_token_ids = []
        unit_to_token_mapping = []
        unit_token_counts = [0] * len(units)

        for unit_idx, unit in enumerate(units):
            token_ids = self.tok(unit, add_special_tokens=False, return_tensors="pt").input_ids[0]
            unit_token_ids.append(token_ids)
            unit_to_token_mapping.extend([unit_idx] * len(token_ids))
            unit_token_counts[unit_idx] += len(token_ids)

        prompt_ids = torch.cat(unit_token_ids, dim=0)
        full_id = torch.stack([torch.cat(
            (self.token_ids["prefix_ids"], prompt_ids, self.token_ids["infix_ids"],
            self.token_ids["affirmation_ids"], self.token_ids["suffix_ids"]),
            dim=0
        )])

        grad_norm=self._compute_gradient(full_id,prompt_ids)[0].norm(dim=1)

        unit_scores = [0.0] * len(units)
        for token_idx, unit_idx in enumerate(unit_to_token_mapping):
            unit_scores[unit_idx] += grad_norm[token_idx].item()

        for unit_idx in range(len(units)):
            if unit_token_counts[unit_idx] > 0:
                unit_scores[unit_idx] /= unit_token_counts[unit_idx]

        return units, unit_scores

    def explain_instance_matrix(self, units: List[str]) -> Tuple[List[str], List[float]]:
        """
        Use the Frobenius norm of the token gradient matrix as the importance score for each unit.

        Args:
            units (List[str]): A list of text units (e.g., phrases or words) that form the prompt.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - The list of units.
                - The unit scores based on Frobenius norms of token gradients.
        """
        unit_token_ids = []
        unit_to_token_mapping = []

        for unit_idx, unit in enumerate(units):
            token_ids = self.tok(unit, add_special_tokens=False, return_tensors="pt").input_ids[0]
            unit_token_ids.append(token_ids)
            unit_to_token_mapping.extend([unit_idx] * len(token_ids))

        prompt_ids = torch.cat(unit_token_ids, dim=0)
        full_id = torch.stack([torch.cat(
            (self.token_ids["prefix_ids"], prompt_ids, self.token_ids["infix_ids"],
            self.token_ids["affirmation_ids"], self.token_ids["suffix_ids"]),
            dim=0
        )])

        grad_matrix = self._compute_gradient(full_id,prompt_ids)[0]
        unit_scores = [0.0] * len(units)
        current_token_idx = 0

        for unit_idx, unit in enumerate(units):
            num_tokens_in_unit = unit_to_token_mapping.count(unit_idx)
            if num_tokens_in_unit > 0:
                unit_grad_matrix = grad_matrix[
                    current_token_idx:current_token_idx + num_tokens_in_unit, :
                ]
                unit_scores[unit_idx] = torch.norm(unit_grad_matrix, p='fro').item()
                current_token_idx += num_tokens_in_unit

        return units, unit_scores

    def explain_instance_mean_norm(self, units: List[str]) -> Tuple[List[str], List[float]]:
        """
        Use the average of the L2 norms of token gradients as the importance score for each unit.

        Args:
            units (List[str]): A list of text units (e.g., phrases or words) that form the prompt.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - The list of units.
                - The unit scores based on the mean_norm method.
        """
        unit_token_ids = []
        unit_to_token_mapping = []

        for unit_idx, unit in enumerate(units):
            token_ids = self.tok(unit, add_special_tokens=False, return_tensors="pt").input_ids[0]
            unit_token_ids.append(token_ids)
            unit_to_token_mapping.extend([unit_idx] * len(token_ids))

        prompt_ids = torch.cat(unit_token_ids, dim=0)
        full_id = torch.stack([torch.cat(
            (self.token_ids["prefix_ids"], prompt_ids, self.token_ids["infix_ids"],
            self.token_ids["affirmation_ids"], self.token_ids["suffix_ids"]),
            dim=0
        )])

        grad_matrix = self._compute_gradient(full_id,prompt_ids)[0]
        unit_scores = [0.0] * len(units)
        current_token_idx = 0

        for unit_idx, unit in enumerate(units):
            num_tokens_in_unit = unit_to_token_mapping.count(unit_idx)
            if num_tokens_in_unit > 0:
                unit_grad_matrix = grad_matrix[
                    current_token_idx:current_token_idx + num_tokens_in_unit, :
                ]
                avg_vector = unit_grad_matrix.mean(dim=0)
                unit_scores[unit_idx] = torch.norm(avg_vector, p=2).item()
                current_token_idx += num_tokens_in_unit

        return units, unit_scores
