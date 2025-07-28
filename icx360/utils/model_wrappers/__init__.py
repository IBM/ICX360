#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing wrappers for different types of models (used by MExGen and CELL).
"""

from .base_model_wrapper import GeneratedOutput, Model
from .huggingface import HFModel
from .vllm import VLLMModel
