#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing scalarizers, which compute scalar output values based on the outputs or inputs of an LLM.
"""

from .bart_score import BARTScorer
from .base_scalarizer import Scalarizer
from .bleu_scalarizer import BleuScalarizer
from .contradiction_scalarizer import ContradictionScalarizer
from .nli_scalarizer import NLIScalarizer
from .preference_scalarizer import PreferenceScalarizer
from .prob import ProbScalarizedModel
from .text_only import TextScalarizedModel
