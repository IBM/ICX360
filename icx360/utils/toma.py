"""
Model inference utilities that use the toma package to avoid running out of CUDA memory.
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.

import torch
from toma import toma


@toma.range()
def toma_generate(start, end, model, input_dict, output_ids, output_hidden_states=False, hidden_states=None, **kwargs):
    """
    Generate outputs using the toma package to adapt to CUDA memory constraints.

    This function passes a batch of inputs to a transformers generative model.
    It generates token IDs and, optionally, hidden states, and stores them in pre-allocated Tensors.

    Args:
        start (int):
            Index of the first input in the batch.
        end (int):
            Index of the last input in the batch.
        model (transformers model):
            Generative model.
        input_dict (dict-like):
            Dict-like object produced by a HuggingFace tokenizer, containing input data.
        output_ids ((num_inputs, gen_start + max_new_tokens) torch.Tensor):
            Pre-allocated Tensor to store generated token IDs.
        output_hidden_states (bool):
            Whether to also output model's hidden states/representations.
        hidden_states (tuple(torch.Tensor) or None):
            If output_hidden_states == True, then for each layer of the encoder,
            a pre-allocated (num_inputs, input_length, hidden_dim) Tensor of hidden states/representations,
            to be populated by toma_generate. Otherwise, None.
        **kwargs (dict):
            Additional keyword arguments for the HuggingFace model.

    Returns:
        None.

    This function modifies the provided output_ids Tensor in-place with generated token IDs and, if requested,
    the hidden_states tuple with corresponding hidden states.
    """
    # input_dict for batch
    input_dict_batch = input_dict.copy()
    for key in input_dict_batch:
        if isinstance(input_dict_batch[key], torch.Tensor):
            # Slice tensor in input_dict_batch
            input_dict_batch[key] = input_dict_batch[key][start:end]
    print("toma_generate batch size =", end - start)

    # Generate outputs
    output_dict = model.generate(**input_dict_batch, return_dict_in_generate=True, output_hidden_states=output_hidden_states, **kwargs)
    gen_length = output_dict.sequences.shape[1]
    # Save output batch
    output_ids[start:end, :gen_length] = output_dict.sequences

    if output_hidden_states:
        # Save hidden states
        for layer in range(len(hidden_states)):
            hidden_states[layer][start:end] = output_dict.encoder_hidden_states[layer]

    return

@toma.range()
def toma_call(start, end, model, input_dict, logits, output_hidden_states=False, hidden_states=None):
    """
    Call model using the toma package to adapt to CUDA memory constraints.

    This function passes a batch of inputs to a transformers classification model.
    It produces logits and, optionally, hidden states, and stores them in pre-allocated Tensors.

    Args:
        start (int):
            Index of the first input in the batch.
        end (int):
            Index of the last input in the batch.
        model (transformers model):
            Classification model.
        input_dict (dict-like):
            Dict-like object produced by a HuggingFace tokenizer, containing input data.
        logits ((num_inputs, num_labels) torch.Tensor):
            Pre-allocated Tensor to store logits.
        output_hidden_states (bool):
            Whether to also output model's hidden states/representations.
        hidden_states (tuple(torch.Tensor) or None):
            If output_hidden_states == True, then for each layer of the model,
            a pre-allocated (num_inputs, input_length, hidden_dim) Tensor of hidden states/representations,
            to be populated by toma_call. Otherwise, None.

    Returns:
        None.

    This function modifies the provided logits Tensor in-place with predicted logits and, if requested,
    the hidden_states tuple with corresponding hidden states.
    """
    # input_dict for batch
    input_dict_batch = input_dict.copy()
    for key in input_dict_batch:
        if isinstance(input_dict_batch[key], torch.Tensor):
            # Slice tensor in input_dict_batch
            input_dict_batch[key] = input_dict_batch[key][start:end]
    print("toma_call batch size =", end - start)

    # Call model
    with torch.no_grad():
        output_dict = model(**input_dict_batch, output_hidden_states=output_hidden_states)
    # Save batch of logits
    logits[start:end] = output_dict.logits

    if output_hidden_states:
        # Save hidden states
        for layer in range(len(hidden_states)):
            hidden_states[layer][start:end] = output_dict.hidden_states[layer]

    return

@toma.range()
def toma_get_probs(start, end, model, input_dict, ref_output, log_probs, output_hidden_states=False, hidden_states=None):
    """
    Compute log probabilities of tokens in a given reference output using the toma package to adapt to CUDA memory.

    This function passes a batch of inputs to a transformers generative model.
    It computes log probabilities of reference output tokens conditioned on these outputs
    and, optionally, hidden states, and stores them in pre-allocated Tensors.

    Args:
        start (int):
            Index of the first input in the batch.
        end (int):
            Index of the last input in the batch.
        model (transformers model):
            Generative model.
        input_dict (dict-like):
            Dict-like object produced by a HuggingFace tokenizer, containing input data.
        ref_output ((1, num_output_tokens) torch.Tensor):
            Token IDs of reference output to compute log probabilities for.
        log_probs ((num_inputs, gen_length) torch.Tensor):
            Pre-allocated Tensor to store log probabilities.
        output_hidden_states (bool):
            Whether to also output model's hidden states/representations.
        hidden_states (tuple(torch.Tensor) or None):
            If output_hidden_states == True, then for each layer of the model,
            a pre-allocated (num_inputs, input_length, hidden_dim) Tensor of hidden states/representations,
            to be populated by toma_get_probs. Otherwise, None.

    Returns:
        None.

    This function modifies the provided log_probs Tensor in-place with predicted log probabilities and, if requested,
    the hidden_states tuple with corresponding hidden states.
    """
    batch_size = end - start
    print("toma_get_probs batch size =", batch_size)

    # input_dict for batch
    input_dict_batch = input_dict.copy()
    for key in input_dict_batch:
        if isinstance(input_dict_batch[key], torch.Tensor):
            # Slice tensor in input_dict_batch
            input_dict_batch[key] = input_dict_batch[key][start:end]

    # Call model on given input and output sequences
    ref_output_expanded = ref_output.expand(batch_size, -1)
    with torch.no_grad():
        if model.config.is_encoder_decoder:
            # Encoder-decoder model: pass inputs and reference output as separate arguments
            output_dict = model(**input_dict_batch, decoder_input_ids=ref_output_expanded, output_hidden_states=output_hidden_states)
        else:
            # Decoder-only model: concatenate inputs with reference output
            combined_input_output = torch.cat([input_dict_batch["input_ids"], ref_output_expanded], dim=1)
            output_dict = model(combined_input_output, output_hidden_states=output_hidden_states)

    # Number of generated tokens in output
    # encoder-decoder output always begins with a fixed special token e.g. <pad>,
    # while decoder-only output has been truncated to only the generated response
    gen_length = ref_output.shape[1] - model.config.is_encoder_decoder
    # Position where generated output starts (in concatenated input-output for decoder-only)
    gen_start = 1 if model.config.is_encoder_decoder else input_dict_batch["input_ids"].shape[1]

    # Convert logits into tuple
    # logits indices are off by one because logits at position i-1 are for predicting token at position i
    scores = tuple(output_dict.logits[:, pos, :] for pos in range(gen_start - 1, gen_start + gen_length - 1))

    # Get probabilities of tokens in the given output
    # NOTE: although ref_output_expanded and scores have different token lengths,
    # compute_transition_scores() seems to align their last positions
    log_probs[start:end] = model.compute_transition_scores(ref_output_expanded, scores, normalize_logits=True)

    if output_hidden_states:
        # Save hidden states
        for layer in range(len(hidden_states)):
            hidden_states[layer][start:end] = output_dict.encoder_hidden_states[layer]

    return
