import inspect 
import os
import random
import textwrap
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

import pandas as pd
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset, IterableDataset
from packaging import version
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_comet_available,
    is_wandb_available,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_peft_available, is_torch_xpu_available
from transformers.utils.deprecation import deprecate_kwarg

from ..data_utils import maybe_apply_chat_template, maybe_extract_prompt
from ..models import PreTrainedModelWrapper, create_reference_model
from .callbacks import SyncRefModelCallback
from .dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType
from .utils import (
    RunningMoments,
    cap_exp,
    disable_dropout_in_model,
    empty_cache,
    flush_left,
    generate_model_card,
    get_comet_experiment_url,
    log_table_to_comet_experiment,
    pad,
    pad_to_length,
    peft_module_casting_to_bf16,
)

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    """
    """