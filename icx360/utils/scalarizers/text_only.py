"""
Scalarized model that computes similarity scores between generated texts and a reference output text.

This "scalarized model" is a generative model that can also compute similarity scores
between the texts it generates and a reference output text.
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.

from math import ceil, log2

import evaluate
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from icx360.utils.general_utils import select_device
from icx360.utils.scalarizers import BARTScorer, Scalarizer
from icx360.utils.toma import toma_call, toma_get_probs


class TextScalarizedModel(Scalarizer):
    """
    Generative model that also computes similarity scores between its generated texts and a reference text.

    Attributes:
        model (icx360.utils.model_wrappers.Model):
            Generative model, wrapped in an icx360.utils.model_wrappers.Model object.
        sim_scores (List[str]):
            List of similarity scores to compute.
                "nli_logit"/"nli": Logit/probability of entailment label from natural language inference model.
                "bert": BERTScore.
                "st": Cosine similarity between SentenceTransformer embeddings.
                "summ": Generation probability of a summarization model (similar to BARTScore).
                "bart": BARTScore.
        model_nli (transformers.AutoModelForSequenceClassification):
            Natural language inference model.
        tokenizer_nli (transformers.AutoTokenizer):
            Tokenizer for natural language inference model.
        idx_entail (int):
            Index corresponding to entailment label.
        bertscore (evaluate.EvaluationModule):
            BERTScore evaluation module.
        model_bert (str):
            Name of BERT-like model for computing BERTScore.
        model_st (SentenceTransformer model):
            SentenceTransformer embedding model.
        model_summ (transformers.AutoModelForSeq2SeqLM):
            Summarization model.
        tokenizer_summ (transformers.AutoTokenizer):
            Tokenizer for summarization model.
        bart_scorer (BARTScorer):
            Object for computing BARTScore.
        device (torch.device or str or None):
            Device for the above models.
    """
    def __init__(self, model=None, sim_scores=["nli_logit", "bert", "st", "summ", "bart"],
                 model_nli=None, model_bert=None, model_st="all-MiniLM-L6-v2",
                 model_summ=None, model_bart="facebook/bart-large-cnn", device=None):
        """
        Initialize TextScalarizedModel.

        Args:
            model (icx360.utils.model_wrappers.Model):
                Generative model, wrapped in an icx360.utils.model_wrappers.Model object.
            sim_scores (List[str]):
                List of similarity scores to compute.
                    "nli_logit"/"nli": Logit/probability of entailment label from natural language inference model.
                    "bert": BERTScore.
                    "st": Cosine similarity between SentenceTransformer embeddings.
                    "summ": Generation probability of a summarization model (similar to BARTScore).
                    "bart": BARTScore.
            model_nli (str):
                Name of natural language inference model.
            model_bert (str):
                Name of BERT-like model for computing BERTScore.
            model_st (str):
                Name of SentenceTransformer embedding model.
            model_summ (str):
                Name of summarization model.
            model_bart (str):
                Name of BART-like model for computing BARTScore.
            device (torch.device or str or None):
                Device for the above models.
        """
        super().__init__(model)
        self.sim_scores = sim_scores
        self.device = select_device() if device is None else device

        if "nli" in sim_scores or "nli_logit" in sim_scores:
            self.model_nli = AutoModelForSequenceClassification.from_pretrained(model_nli).to(self.device)
            self.tokenizer_nli = AutoTokenizer.from_pretrained(model_nli)

            # Index for entailment class
            for key in self.model_nli.config.label2id:
                if key.lower().startswith("entail"):
                    self.idx_entail = self.model_nli.config.label2id[key]

        if "bert" in sim_scores or "bert_log" in sim_scores:
            self.bertscore = evaluate.load("bertscore")
            self.model_bert = model_bert

        if "st" in sim_scores or "st_log" in sim_scores:
            self.model_st = SentenceTransformer(model_st, device=self.device)

        if "summ" in sim_scores:
            self.model_summ = AutoModelForSeq2SeqLM.from_pretrained(model_summ).to(self.device)
            self.tokenizer_summ = AutoTokenizer.from_pretrained(model_summ)

        if "bart" in sim_scores:
            self.bart_scorer = BARTScorer(device=self.device, checkpoint=model_bart)

    def scalarize_output(self, inputs=None, outputs=None, ref_input=None, ref_output=None,
                         max_new_tokens_factor=1.5, symmetric=True, idf=False, transformation="log_prob_mean",
                         **kwargs):
        """
        Compute similarity scores between generated texts and reference text.

        Args:
            inputs (str or List[str] or List[List[str]] or None):
                Inputs to compute similarity scores for:
                A single input text, a list of input texts, or a list of segmented texts.
            outputs (List[str] or None):
                Generated texts to compute similarity scores for.
                If None, then will be generated by calling self.model.generate().
            ref_input (str or None):
                Reference input used to scalarize - not used.
            ref_output (icx360.utils.model_wrappers.GeneratedOutput):
                Reference output object containing reference text (ref_output.output_text).
            max_new_tokens_factor (float):
                Multiplicative factor for setting max_new_tokens for generation.
            symmetric (bool):
                Make NLI entailment score symmetric
                (geometric mean of reference -> generated and generated -> reference).
            idf (bool):
                Use idf weighting for BERTScore.
            transformation (str, optional):
                Transformation to apply to output token probabilities of summarization model.
                    "log_prob_mean": arithmetic mean of log probabilities (default).
                    "log_prob_sum": sum of log probabilities.
                    "prob_geo_mean": geometric mean of probabilities.
                    "prob_prod": product of probabilities.
            **kwargs (dict):
                Additional keyword arguments for model.

        Returns:
            scores (dict of (num_inputs,) torch.Tensor):
                For each label in self.sim_scores, a Tensor of corresponding similarity scores
                between generated texts and the reference text.
        """
        if ref_output is None:
            raise ValueError("ref_output must be provided for TextScalarizedModel.scalarize_output()")

        if outputs is None:
            # Need to generate output texts
            if inputs is not None and self.model is not None:
                # Set max_new_tokens based on length of reference output
                ref_length = ref_output.output_token_count
                if ref_length is not None and "max_new_tokens" not in kwargs:
                    kwargs["max_new_tokens"] = round(max_new_tokens_factor * ref_length)

                # Generate output texts given inputs
                outputs = self.model.generate(inputs, **kwargs)
            else:
                raise ValueError("Inputs and generative model must be provided if outputs is None")
        num_outputs = len(outputs)

        # Extract reference text
        ref_text = ref_output.output_text
        if isinstance(ref_text, str):
            ref_text = [ref_text]
        print(ref_text)
        print(outputs[:5])

        # Compute each similarity score in self.sim_scores
        scores = {}
        if "nli" in self.sim_scores or "nli_logit" in self.sim_scores:
            # Compute NLI entailment scores
            print("NLI scalarizer ref->gen")
            # First encode using NLI tokenizer
            input_dict_nli = self.tokenizer_nli(ref_text * num_outputs, outputs, padding=True, truncation=True, return_tensors="pt").to(self.device)
            # Initialize NLI logits
            logits_nli = torch.empty((num_outputs, 3), device=self.device)
            # Call NLI model using toma
            batch_size_init = 2 ** ceil(log2(num_outputs))
            toma_call(0, num_outputs, self.model_nli, input_dict_nli, logits_nli, toma_initial_step=batch_size_init)
            # Compute scores (probabilities) for entailment
            scores_nli = logits_nli.softmax(dim=1)[:, self.idx_entail]

            if symmetric:
                # Repeat with reference and generated outputs swapped
                print("NLI scalarizer gen->ref")
                input_dict_nli = self.tokenizer_nli(outputs, ref_text * num_outputs, padding=True, truncation=True, return_tensors="pt").to(self.device)
                logits_nli = torch.empty((num_outputs, 3), device=self.device)
                toma_call(0, num_outputs, self.model_nli, input_dict_nli, logits_nli, toma_initial_step=batch_size_init)
                # Take geometric mean of scores
                scores_nli *= logits_nli.softmax(dim=1)[:, self.idx_entail]
                scores_nli = torch.sqrt(scores_nli)

            if "nli" in self.sim_scores:
                scores["nli"] = scores_nli
            if "nli_logit" in self.sim_scores:
                # Convert probabilities to log-odds
                scores["nli_logit"] = torch.logit(scores_nli.cpu())

        if "bert" in self.sim_scores or "bert_log" in self.sim_scores:
            # Compute BERTScores (F1)
            if self.model_bert is None:
                scores_bert = self.bertscore.compute(predictions=outputs, references=ref_text * num_outputs, lang="en", idf=idf, device=self.device)
            else:
                scores_bert = self.bertscore.compute(predictions=outputs, references=ref_text * num_outputs, model_type=self.model_bert, idf=idf, device=self.device)
            scores_bert = torch.tensor(scores_bert["f1"])

            if "bert" in self.sim_scores:
                scores["bert"] = scores_bert
            if "bert_log" in self.sim_scores:
                # Log scores
                scores["bert_log"] = torch.log(scores_bert)

        if "st" in self.sim_scores or "st_log" in self.sim_scores:
            # Compute SentenceTransformer similarities
            # Encode reference output and generated outputs using SentenceTransformer
            ref_embedding = self.model_st.encode(ref_text, convert_to_tensor=True)
            gen_embedding = self.model_st.encode(outputs, convert_to_tensor=True)

            # Compute cosine similarities
            scores_st = util.cos_sim(ref_embedding, gen_embedding)[0]

            if "st" in self.sim_scores:
                scores["st"] = scores_st
            if "st_log" in self.sim_scores:
                # Log scores
                scores["st_log"] = torch.log(scores_st)

        if "summ" in self.sim_scores:
            print("summ scalarizer")
            # Encode generated and reference texts using summarization tokenizer
            gen_encoding_summ = self.tokenizer_summ(outputs, padding=True, truncation=True, return_tensors="pt").to(self.device)
            ref_encoding_summ = self.tokenizer_summ(ref_text, return_tensors="pt").to(self.device).input_ids

            # Initialize probabilities
            ref_gen_length = ref_encoding_summ.shape[1] - 1 # encoder-decoder output always begins with a special token? e.g. <pad>
            log_probs = torch.empty((num_outputs, ref_gen_length), device=self.device)

            # Get probabilities of tokens in reference text using toma
            batch_size_init = 2 ** ceil(log2(num_outputs))
            toma_get_probs(0, num_outputs, self.model_summ, gen_encoding_summ, ref_encoding_summ, log_probs, toma_initial_step=batch_size_init)

            # Transform probabilities
            if transformation in ("log_prob_mean", "prob_geo_mean"):
                # Mean of log probabilities
                scores["summ"] = log_probs.mean(dim=1)
            elif transformation in ("log_prob_sum", "prob_prod"):
                # Sum of log probabilities
                scores["summ"] = log_probs.sum(dim=1)
            else:
                raise ValueError("Transformation not recognized")
            if transformation.startswith("prob"):
                # Convert log probabilities to probabilities
                scores["summ"] = scores["summ"].exp()

        if "bart" in self.sim_scores:
            scores["bart"] = self.bart_scorer.score(outputs, ref_text * num_outputs)
            scores["bart"] = torch.tensor(scores["bart"])

        return scores
