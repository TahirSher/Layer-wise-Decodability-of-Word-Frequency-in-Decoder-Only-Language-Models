import os
import sys
import gc
import logging
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

TARGET_GPU = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(TARGET_GPU)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, r2_score, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, Ridge
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('multi_model_ldt_v6.log')
    ]
)
logger = logging.getLogger(__name__)
SEED = 42

# ── GPU ──────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    _gp = torch.cuda.get_device_properties(0)
    logger.info(f"GPU: {_gp.name}  ({_gp.total_memory/1024**3:.1f} GB)")
else:
    DEVICE = torch.device("cpu")
    logger.warning("CUDA not available — using CPU")

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


# ════════════════════════════════════════════════════════════════════════════
# ENUMS & CONFIG
# ════════════════════════════════════════════════════════════════════════════

class ArchitectureType(Enum):
    ENCODER_ONLY    = "encoder-only"
    DECODER_ONLY    = "decoder-only"
    ENCODER_DECODER = "encoder-decoder"


class InputMode(Enum):
    RAW_TEXT = "raw_text"
    CHAT     = "chat"


class PaddingSide(Enum):
    LEFT  = "left"
    RIGHT = "right"


class PoolingStrategy(Enum):
    """Pooling strategies for hidden state extraction."""
    LAST_TOKEN = "last_token"
    MEAN_POOL  = "mean_pool"


@dataclass
class ModelConfig:
    name:              str
    model_id:          str
    architecture_type: str
    input_mode:        str  = "raw_text"
    padding_side:      str  = "left"
    batch_size:        int  = 32
    use_8bit:          bool = False
    use_4bit:          bool = False

    def __post_init__(self):
        self.architecture_type = ArchitectureType(self.architecture_type)
        self.input_mode        = InputMode(self.input_mode)
        self.padding_side      = PaddingSide(self.padding_side)


@dataclass
class Config:
    WORDS_PATH:    str = "Items.csv"
    NONWORDS_PATH: str = "NonWord.csv"
    OUTPUT_DIR:    str = "LDT_results"

    MODELS: List[ModelConfig] = field(default_factory=lambda: [


        ModelConfig(
            name="SmolLM2-360M",
            model_id="HuggingFaceTB/SmolLM2-360M",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="Qwen2.5-1.5B",
            model_id="Qwen/Qwen2.5-1.5B",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="Llama-3.2-1B",
            model_id="meta-llama/Llama-3.2-1B",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="Qwen2.5-3B",
            model_id="Qwen/Qwen2.5-3B",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="SmolLM3-3B-Base",
            model_id="HuggingFaceTB/SmolLM3-3B-Base",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="LLaMA-3.2-3B",
            model_id="meta-llama/Llama-3.2-3B",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="Qwen3-4B-Base",
            model_id="Qwen/Qwen3-4B-Base",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),



        ModelConfig(
            name="Qwen3-8B-Base",
            model_id="Qwen/Qwen3-8B-Base",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="Llama-3.1-8B",
            model_id="meta-llama/Llama-3.1-8B",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),
    ])

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_SAMPLES: Optional[int] = None

    CLEAR_CACHE_BETWEEN_MODELS: bool = True

    HIGH_FREQ_PERCENTILE: float = 66.0
    LOW_FREQ_PERCENTILE:  float = 33.0

    CLASSIFIER_ARCHITECTURE: list  = field(default_factory=lambda: [512, 256])
    CLASSIFIER_USE_RESIDUAL:  bool  = False
    CLASSIFIER_DROPOUT:       float = 0.3
    CLASSIFIER_LR:            float = 1e-3
    CLASSIFIER_EPOCHS:        int   = 50
    CLASSIFIER_PATIENCE:      int   = 10
    CLASSIFIER_WEIGHT_DECAY:  float = 1e-2
    CLASSIFIER_BATCH_SIZE:    int   = 64
    CLASSIFIER_FOCAL_GAMMA:   float = 0.0
    CLASSIFIER_BN_MOMENTUM:   float = 0.1
    CLASSIFIER_NOISE_STD:     float = 0.0

    VAL_SIZE:  float = 0.15
    TEST_SIZE: float = 0.15

    MIN_BATCH_SIZE:        int   = 8
    MIN_TEST_SAMPLES:      int   = 10
    MIN_SAMPLES_PER_GROUP: int   = 50
    ALPHA:                 float = 0.05

    # ── Analysis 4: Multi-seed stability ──────────────────────────────
    N_SEEDS: int = 5  # Number of random seeds for stability analysis

    # ── Analysis 5: Pooling ablation ──────────────────────────────────
    POOLING_STRATEGIES: List[str] = field(default_factory=lambda: [
        "last_token", "mean_pool"
    ])

    FIGURES_DIR:    str = None
    RESULTS_DIR:    str = None
    COMPARISON_DIR: str = None

    def __post_init__(self):
        self.FIGURES_DIR    = os.path.join(self.OUTPUT_DIR, "figures")
        self.RESULTS_DIR    = os.path.join(self.OUTPUT_DIR, "results")
        self.COMPARISON_DIR = os.path.join(self.OUTPUT_DIR, "comparisons")
        for d in [self.OUTPUT_DIR, self.FIGURES_DIR,
                  self.RESULTS_DIR, self.COMPARISON_DIR]:
            os.makedirs(d, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════════════

class LDTDataset(Dataset):
    """Balanced word / non-word dataset with frequency stratification."""

    def __init__(self, words_df, nonwords_df, tokenizer, config: Config):
        self.tokenizer = tokenizer
        self.config    = config

        wp  = self._process_words(words_df)
        nwp = self._process_nonwords(nonwords_df)

        if config.MAX_SAMPLES is not None:
            tgt = config.MAX_SAMPLES // 2
            wp  = wp.sample(n=min(tgt, len(wp)),   random_state=SEED)
            nwp = nwp.sample(n=min(tgt, len(nwp)), random_state=SEED)
        else:
            n   = min(len(wp), len(nwp))
            wp  = wp.sample(n=n,  random_state=SEED)
            nwp = nwp.sample(n=n, random_state=SEED)

        self.data = (pd.concat([wp, nwp], ignore_index=True)
                       .sample(frac=1, random_state=SEED)
                       .reset_index(drop=True))
        self._compute_token_counts(tokenizer)
        self._stratify_frequency()
        self._print_summary()

    def _process_words(self, df):
        p = pd.DataFrame()
        p['stimulus']      = df['Word'].astype(str).str.lower()
        p['is_word']       = 1
        p['length']        = pd.to_numeric(df['Length'],           errors='coerce')
        p['accuracy']      = pd.to_numeric(
            df['I_Mean_Accuracy'].replace('#', np.nan),            errors='coerce')
        p['log_frequency'] = pd.to_numeric(
            df['Log_Freq_HAL'].replace('#', np.nan),               errors='coerce')
        p['ortho_n']       = pd.to_numeric(df['Ortho_N'].replace('#', np.nan), errors='coerce')
        p['is_high_freq']  = 0
        p['is_low_freq']   = 0
        p['freq_group']    = 'mid'
        return p.dropna(subset=['stimulus'])

    def _process_nonwords(self, df):
        p = pd.DataFrame()
        p['stimulus']      = df['Word'].astype(str).str.lower()
        p['is_word']       = 0
        p['length']        = pd.to_numeric(df['Length'],           errors='coerce')
        p['accuracy']      = pd.to_numeric(
            df['NWI_Mean_Accuracy'].replace('#', np.nan),          errors='coerce')
        p['log_frequency'] = np.nan
        p['ortho_n']       = pd.to_numeric(df['Ortho_N'].replace('#', np.nan), errors='coerce')
        p['is_high_freq']  = 0
        p['is_low_freq']   = 0
        p['freq_group']    = 'nonword'
        return p.dropna(subset=['stimulus'])

    # ── Analysis 2: Tokenization control ──────────────────────────────
    def _compute_token_counts(self, tokenizer):
        """
        For each stimulus, compute the number of subword tokens and
        whether it is a single-token item under this tokenizer.
        This is essential for controlling tokenization confounds:
        frequency effects observed in probing could be artefacts of
        high-frequency words being single-token while low-frequency
        words are multi-token (and thus have a fundamentally different
        representation structure).
        """
        token_counts = []
        for stim in self.data['stimulus']:
            ids = tokenizer.encode(stim, add_special_tokens=False)
            token_counts.append(len(ids))
        self.data['token_count'] = token_counts
        self.data['is_single_token'] = (self.data['token_count'] == 1).astype(int)
        logger.info(
            f"Tokenization: single-token={self.data['is_single_token'].sum()}, "
            f"multi-token={(self.data['is_single_token'] == 0).sum()}, "
            f"mean tokens={self.data['token_count'].mean():.2f}"
        )

    def _stratify_frequency(self):
        wm = self.data['is_word'] == 1
        fm = wm & self.data['log_frequency'].notna()
        if fm.sum() == 0:
            logger.warning("No frequency data — stratification skipped"); return
        fv  = self.data.loc[fm, 'log_frequency']
        hi  = np.percentile(fv, self.config.HIGH_FREQ_PERCENTILE)
        lo  = np.percentile(fv, self.config.LOW_FREQ_PERCENTILE)
        hm  = fm & (self.data['log_frequency'] >= hi)
        lm  = fm & (self.data['log_frequency'] <= lo)
        mm  = fm & ~hm & ~lm
        self.data.loc[hm, ['is_high_freq','freq_group']] = [1, 'high']
        self.data.loc[lm, ['is_low_freq', 'freq_group']] = [1, 'low']
        self.data.loc[mm,  'freq_group']                 = 'mid'
        logger.info(f"Freq stratification: hi>={hi:.3f}  lo<={lo:.3f}")

    def _print_summary(self):
        wm = self.data['is_word'] == 1
        print(f"\n{'='*64}\nDATASET SUMMARY\n{'='*64}")
        print(f"  Total         : {len(self.data):,}")
        print(f"  Words  (1)    : {wm.sum():,}")
        print(f"  Non-words (0) : {(~wm).sum():,}")
        print(f"  High-freq     : {(self.data['is_high_freq']==1).sum():,}")
        print(f"  Low-freq      : {(self.data['is_low_freq'] ==1).sum():,}")
        print(f"  Single-token  : {self.data['is_single_token'].sum():,}")
        print(f"  Multi-token   : {(self.data['is_single_token']==0).sum():,}")
        print(f"{'='*64}\n")

    def __len__(self):  return len(self.data)

    def __getitem__(self, idx):
        r = self.data.iloc[idx]
        def ft(v): return torch.tensor(v if pd.notna(v) else float('nan'),
                                       dtype=torch.float32)
        return {
            'stimulus':        r['stimulus'],
            'is_word':         torch.tensor(r['is_word'],      dtype=torch.long),
            'accuracy':        ft(r['accuracy']),
            'log_frequency':   ft(r['log_frequency']),
            'length':          ft(r['length']),
            'ortho_n':         ft(r['ortho_n']),
            'is_high_freq':    torch.tensor(r['is_high_freq'], dtype=torch.long),
            'is_low_freq':     torch.tensor(r['is_low_freq'],  dtype=torch.long),
            'freq_group':      r['freq_group'],
            'token_count':     torch.tensor(r['token_count'],  dtype=torch.long),
            'is_single_token': torch.tensor(r['is_single_token'], dtype=torch.long),
            'idx':             idx,
        }


# ════════════════════════════════════════════════════════════════════════════
# SINGLE-PASS ALL-LAYER EXTRACTOR
# ════════════════════════════════════════════════════════════════════════════

class AllLayerExtractor:

    def __init__(self, model_config: ModelConfig, device: str = 'cuda'):
        print(f"\n{'='*64}\nLOADING: {model_config.name}")
        print(f"  model_id     : {model_config.model_id}")
        print(f"  arch         : {model_config.architecture_type.value}")
        print(f"  input_mode   : {model_config.input_mode.value}")
        print(f"  padding_side : {model_config.padding_side.value}")
        print(f"  extraction   : SINGLE PASS — all layers cached in one forward pass")
        print(f"{'='*64}")

        self.model_config      = model_config
        self.device            = device
        self.architecture_type = model_config.architecture_type
        self.input_mode        = model_config.input_mode

        # ── tokeniser ─────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_id,
            trust_remote_code=True
        )

        if model_config.architecture_type == ArchitectureType.DECODER_ONLY:
            self.tokenizer.padding_side = model_config.padding_side.value

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token    = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ── quantisation ──────────────────────────────────────────────────
        qcfg = None
        if model_config.use_4bit:
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif model_config.use_8bit:
            qcfg = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                llm_int8_enable_fp32_cpu_offload=True
            )

        dmap    = 'auto' if device == 'cuda' else None
        max_mem = None
        if device == 'cuda' and torch.cuda.is_available():
            tot     = torch.cuda.get_device_properties(0).total_memory / 1024**3
            max_mem = {0: f"{int(tot*0.85)}GB", "cpu": "30GB"}

        kw = dict(
            output_hidden_states=True,
            device_map=dmap,
            quantization_config=qcfg,
            low_cpu_mem_usage=True,
            max_memory=max_mem,
            trust_remote_code=True
        )

        # ── dtype selection ───────────────────────────────────────────
        # bfloat16 has the same exponent range as float32 (max ~3.4e38)
        # so it eliminates the fp16 overflow→NaN problem in deep layers
        # while using the same 2 bytes per parameter as fp16.
        # Requires compute capability ≥ 8.0 (Ampere: A100, A6000, etc.)
        if device == 'cuda' and torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc[0] >= 8:
                kw['torch_dtype'] = torch.bfloat16
                logger.info(f"  Using bfloat16 (GPU CC {cc[0]}.{cc[1]} ≥ 8.0)")
            else:
                kw['torch_dtype'] = torch.float16
                logger.info(f"  Using float16 (GPU CC {cc[0]}.{cc[1]} < 8.0 — "
                            f"bf16 not supported, watch for overflow in deep layers)")
        else:
            kw['torch_dtype'] = torch.float32

        try:
            self.model = AutoModel.from_pretrained(model_config.model_id, **kw)
            self._model_type = "AutoModel"
        except Exception as e:
            logger.warning(
                f"AutoModel failed for {model_config.model_id} ({e}); "
                f"falling back to AutoModelForCausalLM"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id, **kw
            )
            self._model_type = "AutoModelForCausalLM"

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        print(f"✓ Loaded ({self._model_type}) | layers={self.num_layers} "
              f"| hidden_dim={self.hidden_dim}")

        print(f"  pad_token    : '{self.tokenizer.pad_token}' "
              f"(id={self.tokenizer.pad_token_id})")
        print(f"  padding_side : {self.tokenizer.padding_side}")

        if self.input_mode == InputMode.CHAT:
            self._validate_chat_template()

        print()

    def _validate_chat_template(self):
        test_msg = [{"role": "user", "content": "test"}]
        try:
            if (hasattr(self.tokenizer, 'apply_chat_template') and
                    self.tokenizer.chat_template is not None):
                formatted = self.tokenizer.apply_chat_template(
                    test_msg, tokenize=False, add_generation_prompt=True
                )
                print(f"  chat_template test: {repr(formatted[:80])}...")
        except Exception as e:
            logger.warning(f"Chat template validation failed: {e}")

    def build_input_text(self, stimulus: str) -> str:
        if self.input_mode == InputMode.RAW_TEXT:
            return stimulus

        if (hasattr(self.tokenizer, 'apply_chat_template') and
                self.tokenizer.chat_template is not None):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": stimulus}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(
                    f"apply_chat_template failed ({e}); using model-specific fallback"
                )

        model_id_lower = self.model_config.model_id.lower()
        if 'phi-3' in model_id_lower or 'phi3' in model_id_lower:
            return f"<|user|>\n{stimulus}<|end|>\n<|assistant|>\n"
        if 'qwen' in model_id_lower:
            return (f"<|im_start|>user\n{stimulus}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
        if 'llama' in model_id_lower:
            return (f"<|begin_of_text|><|start_header_id|>user"
                    f"<|end_header_id|>\n\n{stimulus}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")

        logger.warning(
            f"No known template for {self.model_config.model_id}; "
            f"using raw stimulus"
        )
        return stimulus

    # ── Analysis 5: Pooling with explicit strategy selection ──────────
    def _pool(self, layer_hidden_b: torch.Tensor,
              attention_mask_b: torch.Tensor,
              strategy: str = "last_token") -> np.ndarray:
        """
        Pool a single sample's hidden states to a fixed-size vector.

        Parameters
        ----------
        layer_hidden_b   : (T, H) — one sample's hidden states for one layer
        attention_mask_b : (T,) — attention mask (1=real, 0=pad)
        strategy         : 'last_token' or 'mean_pool'

        Returns
        -------
        float32 numpy vector of shape (H,)

        NaN handling
        ────────────
        Deep transformer layers in fp16 can overflow for certain inputs,
        producing NaN/Inf in hidden states. We upcast to float32 BEFORE
        pooling to rescue values that are merely large (> 65504) but
        within fp32 range. For values that are already NaN from the
        model's internal fp16 computation (e.g. Inf * 0 in LayerNorm),
        we replace them with 0.0 post-pooling so they do not propagate
        into downstream probe training (StandardScaler, gradient
        computation, etc.). Affected samples are counted and logged
        by _sanity_check.

        Pooling logic
        ─────────────
        last_token:
            Encoder-only: CLS token at position 0.
            Decoder-only + left-pad: position -1 (always real).
            Decoder-only + right-pad: last non-pad position.

        mean_pool:
            Mean of all non-padding token representations.
            Muennighoff et al. (2022) show mean pooling can be
            competitive with last-token pooling for representation
            extraction. Masking out padding tokens is essential for
            unbiased mean computation — padding tokens carry no
            semantic information and would dilute the representation.
        """
        # ── Upcast to fp32 BEFORE any arithmetic ──────────────────────
        # This is the primary defence against fp16 overflow NaNs.
        # In fp16, values > 65504 become Inf, and Inf in LayerNorm
        # produces NaN.  Upcasting here cannot recover values that
        # the model already turned into NaN internally, but it
        # prevents the pooling arithmetic itself from introducing
        # additional precision loss.
        layer_hidden_b = layer_hidden_b.float()
        attention_mask_b = attention_mask_b.float()

        if strategy == "mean_pool":
            # Mean over non-pad tokens only
            mask = attention_mask_b.unsqueeze(-1)  # (T, 1)
            masked = layer_hidden_b * mask  # zero out pad positions
            summed = masked.sum(dim=0)      # (H,)
            count  = mask.sum().clamp(min=1)
            vec = summed / count
        else:
            # last_token (default)
            if self.architecture_type == ArchitectureType.ENCODER_ONLY:
                vec = layer_hidden_b[0, :]  # CLS token
            else:
                if self.tokenizer.padding_side == 'left':
                    vec = layer_hidden_b[-1, :]
                else:
                    actual_len = int(attention_mask_b.sum().item())
                    vec = layer_hidden_b[max(actual_len - 1, 0), :]

        result = vec.cpu().numpy()

        # ── Replace residual NaN/Inf with 0.0 ────────────────────────
        # These are values the model itself produced as NaN in fp16
        # (irrecoverable). Replacing with 0.0 is conservative: it
        # prevents downstream StandardScaler from producing all-NaN
        # columns and keeps the sample in the dataset (dropping it
        # would change N and break stratification). The sanity check
        # counts and logs these so the user knows which layers and
        # how many samples are affected.
        nan_mask = ~np.isfinite(result)
        if nan_mask.any():
            result[nan_mask] = 0.0

        return result

    @torch.no_grad()
    def extract_all_layers(self, dataloader,
                           pooling_strategy: str = "last_token"
                           ) -> Tuple[Dict[int, np.ndarray], Dict]:
        """
        Run ONE forward pass per batch, cache all layers.

        Returns
        ───────
        all_hidden : dict  {layer_idx (0 … num_layers-1) : np.ndarray (N, H)}
        targets    : dict  {metadata arrays for the full dataset}
        """
        layer_accum: Dict[int, List[np.ndarray]] = {
            li: [] for li in range(self.num_layers)
        }
        targets = {k: [] for k in
                   ['is_word','is_high_freq','is_low_freq',
                    'freq_group','stimulus','idx',
                    'token_count','is_single_token',
                    'log_frequency','length','ortho_n']}

        n_batches = len(dataloader)
        print(f"  Extracting all {self.num_layers} layers in a single pass "
              f"({n_batches} batches, pooling={pooling_strategy}) …")

        for batch in tqdm(dataloader,
                          desc=f"Extraction [{pooling_strategy}] | {self.model_config.name}"):

            formatted = [self.build_input_text(s) for s in batch['stimulus']]
            enc = self.tokenizer(
                formatted, return_tensors='pt', padding=True,
                truncation=True, max_length=64, add_special_tokens=True
            )

            input_ids      = enc['input_ids'].to(self.device)
            attention_mask = enc['attention_mask'].to(self.device)

            fwd_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            try:
                outputs = self.model(**fwd_kwargs, use_cache=False)
            except TypeError:
                outputs = self.model(**fwd_kwargs)

            B = input_ids.size(0)

            for li in range(self.num_layers):
                lh = outputs.hidden_states[li + 1]  # skip embedding layer

                batch_pool = []
                for b in range(B):
                    batch_pool.append(
                        self._pool(lh[b], attention_mask[b],
                                   strategy=pooling_strategy)
                    )

                layer_accum[li].append(np.stack(batch_pool))
                del lh

            del outputs
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            # ── accumulate metadata ───────────────────────────────────
            targets['is_word'].append(batch['is_word'].cpu().numpy())
            targets['is_high_freq'].append(batch['is_high_freq'].cpu().numpy())
            targets['is_low_freq'].append(batch['is_low_freq'].cpu().numpy())
            targets['freq_group'].extend(batch['freq_group'])
            targets['stimulus'].extend(batch['stimulus'])
            targets['idx'].extend(batch['idx'].cpu().numpy())
            targets['token_count'].append(batch['token_count'].cpu().numpy())
            targets['is_single_token'].append(batch['is_single_token'].cpu().numpy())
            targets['log_frequency'].append(batch['log_frequency'].cpu().numpy())
            targets['length'].append(batch['length'].cpu().numpy())
            targets['ortho_n'].append(batch['ortho_n'].cpu().numpy())

        # ── concatenate batches into full-dataset arrays ──────────────
        all_hidden: Dict[int, np.ndarray] = {}
        for li in range(self.num_layers):
            all_hidden[li] = np.concatenate(layer_accum[li], axis=0)
            del layer_accum[li]
        gc.collect()

        targets = {
            k: (np.concatenate(v) if isinstance(v[0], np.ndarray) else v)
            for k, v in targets.items()
        }

        total_bytes = sum(a.nbytes for a in all_hidden.values())
        print(f"  ✓ Extraction complete — "
              f"cached {self.num_layers} layers  "
              f"({total_bytes/1024**3:.2f} GB RAM)")

        self._sanity_check(all_hidden)

        return all_hidden, targets

    def _sanity_check(self, all_hidden: Dict[int, np.ndarray]):
        """
        Verify extracted representations are not degenerate.

        Checks every layer (not just sampled ones) because NaN
        issues are layer-specific — they typically appear only in
        the deepest layers where the residual stream magnitude is
        highest and most likely to overflow fp16.

        Logs a clear FAIL/WARN/OK status per layer. Previously
        this method could log 'sanity OK' even when NaN was present
        because the OK message was gated only on the variance check.
        """
        total_nan_samples = 0
        total_inf_samples = 0
        total_zero_samples = 0
        nan_layers = []

        for li in range(self.num_layers):
            h = all_hidden[li]
            n_zero = int(np.sum(np.all(h == 0, axis=1)))
            n_nan  = int(np.sum(np.any(np.isnan(h), axis=1)))
            n_inf  = int(np.sum(np.any(np.isinf(h), axis=1)))

            total_nan_samples += n_nan
            total_inf_samples += n_inf
            total_zero_samples += n_zero

            # Compute variance excluding NaN rows for meaningful stat
            finite_mask = np.all(np.isfinite(h), axis=1)
            if finite_mask.sum() > 0:
                var = np.var(h[finite_mask], axis=0).mean()
            else:
                var = np.nan

            has_problem = (n_zero > 0 or n_nan > 0 or n_inf > 0
                           or (not np.isnan(var) and var < 1e-6))

            # Only log details for first, middle, last, and problem layers
            # to avoid flooding the log for 32+ layer models
            is_sampled = li in [0, self.num_layers // 2, self.num_layers - 1]

            if has_problem:
                nan_layers.append(li)
                if n_nan > 0:
                    logger.error(
                        f"  Layer {li}: FAIL — {n_nan} samples contained NaN "
                        f"(replaced with 0.0 in pooled vectors). "
                        f"Cause: fp16 overflow in deep transformer layers."
                    )
                if n_inf > 0:
                    logger.error(
                        f"  Layer {li}: FAIL — {n_inf} samples contain Inf!"
                    )
                if n_zero > 0:
                    logger.warning(
                        f"  Layer {li}: WARN — {n_zero} all-zero vectors. "
                        f"Possible causes: padding token pooled, or NaN→0 replacement."
                    )
                if not np.isnan(var) and var < 1e-6:
                    logger.warning(
                        f"  Layer {li}: WARN — mean feature variance = {var:.2e} "
                        f"(suspiciously low — representations may be collapsed)"
                    )
            elif is_sampled:
                logger.info(
                    f"  Layer {li}: OK (var={var:.4f}, "
                    f"zeros={n_zero}, nan={n_nan})"
                )

        # ── Summary ───────────────────────────────────────────────────
        if total_nan_samples > 0 or total_inf_samples > 0:
            logger.warning(
                f"  SANITY SUMMARY: {total_nan_samples} NaN samples, "
                f"{total_inf_samples} Inf samples across "
                f"{len(nan_layers)} layers {nan_layers}. "
                f"Affected vectors were zeroed to prevent downstream "
                f"corruption. Consider: (1) using bfloat16 instead of "
                f"fp16 if your GPU supports it, or (2) inspecting the "
                f"specific stimuli that trigger overflow."
            )
        else:
            logger.info("  SANITY SUMMARY: All layers clean — no NaN/Inf/zero issues.")


# ════════════════════════════════════════════════════════════════════════════
# PROBE CLASSIFIERS
# ════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Pre-activation residual block. He et al. (2016)."""
    def __init__(self, dim, dropout=0.3, bn_mom=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim, eps=1e-5, momentum=bn_mom)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim, eps=1e-5, momentum=bn_mom)
        self.fc2 = nn.Linear(dim, dim)
        self.dp  = nn.Dropout(dropout)
        for fc in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(fc.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(fc.bias, 0)

    def forward(self, x):
        h = F.relu(self.bn1(x));  h = self.fc1(h)
        h = self.dp(F.relu(self.bn2(h)));  h = self.fc2(h)
        return h + x


class LexicalDecisionClassifier(nn.Module):
    """
    Binary probe: Word (1) vs Non-Word (0).
    When hidden_dims=[], reduces to a linear probe (Hewitt & Liang, 2019).
    """
    def __init__(self, input_dim, hidden_dims=[512, 256],
                 dropout=0.3, use_residual=False, bn_mom=0.1):
        super().__init__()
        self.use_residual = use_residual

        if not hidden_dims:
            # Linear probe
            self.input_proj = nn.Identity()
            self.layers     = nn.ModuleList()
            self.out        = nn.Linear(input_dim, 2)
            nn.init.xavier_normal_(self.out.weight, gain=1.0)
            nn.init.constant_(self.out.bias, 0)
            return

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0], eps=1e-5, momentum=bn_mom),
            nn.ReLU(), nn.Dropout(dropout * 0.5)
        )
        nn.init.kaiming_normal_(self.input_proj[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.input_proj[0].bias, 0)

        if use_residual and hidden_dims:
            self.res1 = ResidualBlock(hidden_dims[0], dropout*0.5, bn_mom)
            self.res2 = ResidualBlock(hidden_dims[0], dropout*0.5, bn_mom)

        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            blk = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1], eps=1e-5, momentum=bn_mom),
                nn.ReLU(), nn.Dropout(dropout)
            )
            nn.init.kaiming_normal_(blk[0].weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(blk[0].bias, 0)
            self.layers.append(blk)

        self.out = nn.Linear(hidden_dims[-1], 2)
        nn.init.xavier_normal_(self.out.weight, gain=1.0)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x):
        x = self.input_proj(x)
        if self.use_residual and hasattr(self, 'res1'):
            x = self.res1(x); x = self.res2(x)
        for l in self.layers: x = l(x)
        return self.out(x)


class FocalLoss(nn.Module):
    """
    Lin et al. (2017). When gamma=0, reduces to weighted cross-entropy.
    """
    def __init__(self, alpha=None, gamma=0.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma

    def forward(self, logits, targets):
        if self.gamma == 0.0:
            weight = self.alpha if self.alpha is not None else None
            return F.cross_entropy(logits, targets, weight=weight)

        ce     = F.cross_entropy(logits, targets, reduction='none')
        log_pt = (F.log_softmax(logits, 1) * F.one_hot(targets, logits.size(1))).sum(1)
        pt     = torch.clamp(log_pt.exp(), 1e-7, 1.0)
        f      = ((1-pt)**self.gamma) * ce
        if self.alpha is not None: f = self.alpha[targets] * f
        return f.mean()


# ════════════════════════════════════════════════════════════════════════════
# METRICS HELPER
# ════════════════════════════════════════════════════════════════════════════

def _full_metrics(y_true, y_pred, y_prob, group_name) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    return {
        'group_name': group_name, 'n_samples': len(y_true),
        'accuracy': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'auc': auc,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'predictions': y_pred, 'probabilities': y_prob,
    }


# ════════════════════════════════════════════════════════════════════════════
# FREQUENCY ANALYZER  (probe training + evaluation, memory-only)
# ════════════════════════════════════════════════════════════════════════════

class FrequencyAnalyzer:
    """
    Trains a probe classifier on the cached hidden states of each layer.
    No transformer forward pass happens here — pure numpy/pytorch training
    on CPU-RAM cached arrays.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE

    def train_classifier(self, X_tr, y_tr, X_val, y_val, layer_idx, model_name,
                         hidden_dims=None, seed=SEED):
        """Train a single probe classifier with specified architecture and seed."""
        # Reproducibility per seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        sc     = StandardScaler()
        Xtr_s  = sc.fit_transform(X_tr)
        Xval_s = sc.transform(X_val)
        Xtr_t  = torch.FloatTensor(Xtr_s); ytr_t  = torch.LongTensor(y_tr)
        Xval_t = torch.FloatTensor(Xval_s); yval_t = torch.LongTensor(y_val)

        ds   = TensorDataset(Xtr_t, ytr_t)
        bs   = min(self.config.CLASSIFIER_BATCH_SIZE,
                   max(self.config.MIN_BATCH_SIZE, len(ds)//10))
        cc   = np.bincount(y_tr)
        sw   = (1.0/cc)[y_tr]
        samp = WeightedRandomSampler(sw, len(sw), replacement=True,
                                     generator=torch.Generator().manual_seed(seed))
        dl   = DataLoader(ds, batch_size=bs, sampler=samp, num_workers=0,
                          pin_memory=(self.device=='cuda'), drop_last=True)

        if hidden_dims is None:
            hidden_dims = self.config.CLASSIFIER_ARCHITECTURE

        model = LexicalDecisionClassifier(
            input_dim=X_tr.shape[1],
            hidden_dims=hidden_dims,
            dropout=self.config.CLASSIFIER_DROPOUT,
            use_residual=self.config.CLASSIFIER_USE_RESIDUAL,
            bn_mom=self.config.CLASSIFIER_BN_MOMENTUM
        ).to(self.device)

        cw   = torch.FloatTensor([len(y_tr)/(len(cc)*c) for c in cc]).to(self.device)
        crit = FocalLoss(alpha=cw, gamma=self.config.CLASSIFIER_FOCAL_GAMMA)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.CLASSIFIER_LR,
            weight_decay=self.config.CLASSIFIER_WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=3, min_lr=1e-6)

        best_vl, pat, best_st = float('inf'), 0, None

        for ep in range(self.config.CLASSIFIER_EPOCHS):
            model.train()
            tl = []
            for bX, by in dl:
                bX, by = bX.to(self.device), by.to(self.device)
                if self.config.CLASSIFIER_NOISE_STD > 0:
                    bX = bX + torch.randn_like(bX) * self.config.CLASSIFIER_NOISE_STD
                opt.zero_grad()
                lg = model(bX); loss = crit(lg, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl.append(loss.item())

            model.eval()
            with torch.no_grad():
                vl    = model(Xval_t.to(self.device))
                vl_loss = F.cross_entropy(vl, yval_t.to(self.device))

            sch.step(vl_loss.item())

            if vl_loss.item() < best_vl:
                best_vl = vl_loss.item(); pat = 0
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
            if pat >= self.config.CLASSIFIER_PATIENCE:
                break

        model.load_state_dict(best_st)
        model.scaler = sc
        return model

    def evaluate_group(self, clf, X, y, group_name) -> Dict:
        Xs = clf.scaler.transform(X)
        clf.eval()
        with torch.no_grad():
            probs = F.softmax(clf(torch.FloatTensor(Xs).to(self.device)), 1).cpu().numpy()
        return _full_metrics(y, probs.argmax(1), probs[:,1], group_name)

    def analyze_layer(self, hidden_states, targets, layer_idx, model_name,
                      seed=SEED) -> Dict:
        """Probe one layer's cached representations."""
        is_word      = targets['is_word']
        is_high_freq = targets['is_high_freq']
        is_low_freq  = targets['is_low_freq']
        freq_groups  = targets['freq_group']

        wm  = (is_word == 1)
        hfm = wm & (is_high_freq == 1)
        lfm = wm & (is_low_freq  == 1)

        if hfm.sum() < self.config.MIN_SAMPLES_PER_GROUP or \
           lfm.sum() < self.config.MIN_SAMPLES_PER_GROUP:
            logger.warning(f"Layer {layer_idx}: insufficient samples "
                           f"(high={hfm.sum()}, low={lfm.sum()})")
            return self._empty_result(layer_idx)

        strat   = ['nonword' if is_word[i]==0 else f"word_{freq_groups[i]}"
                   for i in range(len(is_word))]
        idx_all = np.arange(len(is_word))

        try:
            Xtv,Xte,ytv,yte,itv,ite,stv,_ = train_test_split(
                hidden_states, is_word, idx_all, strat,
                test_size=self.config.TEST_SIZE, stratify=strat,
                random_state=seed)
        except ValueError:
            Xtv,Xte,ytv,yte,itv,ite = train_test_split(
                hidden_states, is_word, idx_all,
                test_size=self.config.TEST_SIZE, random_state=seed); stv=None

        vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
        try:
            Xtr,Xval,ytr,yval,_,_ = train_test_split(
                Xtv,ytv,itv, test_size=vsz,
                stratify=(stv if stv is not None else ytv),
                random_state=seed)
        except ValueError:
            Xtr,Xval,ytr,yval,_,_ = train_test_split(
                Xtv,ytv,itv, test_size=vsz, random_state=seed)

        clf = self.train_classifier(Xtr, ytr, Xval, yval, layer_idx,
                                    model_name, seed=seed)

        overall = self.evaluate_group(clf, Xte, yte, "Test-Overall")

        hf_mask = np.isin(ite, np.where(hfm)[0])
        lf_mask = np.isin(ite, np.where(lfm)[0])

        hf_res = (self.evaluate_group(clf, Xte[hf_mask], yte[hf_mask], "Test-HighFreq")
                  if hf_mask.sum() >= self.config.MIN_TEST_SAMPLES else None)
        lf_res = (self.evaluate_group(clf, Xte[lf_mask], yte[lf_mask], "Test-LowFreq")
                  if lf_mask.sum() >= self.config.MIN_TEST_SAMPLES else None)

        freq_effect = (self._freq_stats(hf_res, lf_res)
                       if hf_res and lf_res else self._empty_freq_effect())

        del clf
        if self.device == 'cuda': torch.cuda.empty_cache()

        return {
            'layer': layer_idx, 'overall': overall,
            'high_frequency': hf_res, 'low_frequency': lf_res,
            'frequency_effect': freq_effect,
            'split_info': {'n_train':len(Xtr),'n_val':len(Xval),'n_test':len(Xte)},
        }

    def _freq_stats(self, hf, lf) -> Dict:
        ah,al   = hf['accuracy'], lf['accuracy']
        nh,nl   = hf['n_samples'], lf['n_samples']
        ch,cl   = int(round(ah*nh)), int(round(al*nl))
        me      = min(ch, nh-ch, cl, nl-cl)
        pp      = (ch+cl)/(nh+nl)
        se      = np.sqrt(pp*(1-pp)*(1/nh+1/nl))
        if se > 0:
            cc = (0.5/nh+0.5/nl) if me < 10 else 0.0
            z  = max(0.0, abs(ah-al)-cc)/se
            p  = 2*(1-stats.norm.cdf(z))
        else:
            z = p = np.nan
        coh_h = (2*(np.arcsin(np.sqrt(np.clip(ah,0,1))) -
                    np.arcsin(np.sqrt(np.clip(al,0,1))))
                 if not (np.isnan(ah) or np.isnan(al)) else np.nan)
        return {
            'accuracy_difference': ah-al,
            'high_freq_accuracy': ah, 'low_freq_accuracy': al,
            'z_statistic': z, 'p_value': p, 'cohens_h': coh_h,
            'n_high': nh, 'n_low': nl, 'correct_high': ch, 'correct_low': cl,
            'continuity_correction': me < 10,
        }

    def _empty_freq_effect(self):
        return ({k: np.nan for k in ['accuracy_difference','high_freq_accuracy',
                                      'low_freq_accuracy','z_statistic',
                                      'p_value','cohens_h']} |
                {'n_high':0,'n_low':0,'correct_high':0,'correct_low':0,
                 'continuity_correction':False})

    def _empty_result(self, li):
        return {'layer':li,'overall':None,'high_frequency':None,'low_frequency':None,
                'frequency_effect':self._empty_freq_effect(),'split_info':None}


# ════════════════════════════════════════════════════════════════════════════
# EXTENDED ANALYSES (Document Requirements 1–7)
# ════════════════════════════════════════════════════════════════════════════

class ExtendedAnalyses:
    """
    Implements all seven analyses required to make the frequency claim
    defensible. Each method operates on pre-extracted hidden states (numpy
    arrays cached in CPU RAM) — no additional GPU forward passes needed.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE

    # ────────────────────────────────────────────────────────────────────
    # Analysis 1: Direct Frequency Probe
    # ────────────────────────────────────────────────────────────────────
    def direct_frequency_probe(self, all_hidden, targets, model_name) -> Dict:
        """
        Train a WORDS-ONLY probe: high-freq vs low-freq (binary).
        This is the single most important analysis because the original
        pipeline infers frequency sensitivity indirectly from accuracy
        gaps in a word-vs-nonword probe. Here we directly classify
        frequency from hidden states.

        Unlike the LDT probe (word vs nonword), this probe trains only
        on word stimuli and the label is the frequency group (high=1,
        low=0). This provides direct evidence that the representation
        encodes frequency information, not just lexical status.

        Also runs an optional 3-way classification (high/mid/low) as
        secondary evidence.
        """
        logger.info(f"  [Analysis 1] Direct frequency probe — {model_name}")

        is_word      = targets['is_word']
        is_high_freq = targets['is_high_freq']
        is_low_freq  = targets['is_low_freq']
        freq_groups  = targets['freq_group']

        # Binary: high vs low words only
        wm  = (is_word == 1)
        hfm = wm & (is_high_freq == 1)
        lfm = wm & (is_low_freq  == 1)
        binary_mask = hfm | lfm

        if binary_mask.sum() < 2 * self.config.MIN_SAMPLES_PER_GROUP:
            logger.warning("  Insufficient samples for direct frequency probe")
            return {'binary': [], 'three_way': []}

        # Labels: 1=high, 0=low
        freq_labels = np.zeros(len(is_word), dtype=int)
        freq_labels[hfm] = 1

        # 3-way: high=2, mid=1, low=0 (words only with frequency data)
        mid_mask = wm & (~hfm) & (~lfm) & (np.array(
            [g == 'mid' for g in freq_groups]))
        three_way_mask = hfm | lfm | mid_mask
        three_way_labels = np.zeros(len(is_word), dtype=int)
        three_way_labels[hfm] = 2
        three_way_labels[mid_mask] = 1
        # low stays 0

        binary_results = []
        three_way_results = []
        analyzer = FrequencyAnalyzer(self.config)

        for li in tqdm(range(len(all_hidden)), desc="Direct freq probe"):
            if li not in all_hidden:
                continue

            # ── Binary probe ──────────────────────────────────────────
            X_bin = all_hidden[li][binary_mask]
            y_bin = freq_labels[binary_mask]

            if len(np.unique(y_bin)) < 2:
                binary_results.append({
                    'layer': li, 'accuracy': np.nan, 'f1': np.nan,
                    'auc': np.nan, 'n_high': hfm.sum(), 'n_low': lfm.sum()
                })
                continue

            try:
                strat_bin = y_bin
                Xtv, Xte, ytv, yte = train_test_split(
                    X_bin, y_bin, test_size=self.config.TEST_SIZE,
                    stratify=strat_bin, random_state=SEED)
                vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
                Xtr, Xval, ytr, yval = train_test_split(
                    Xtv, ytv, test_size=vsz, stratify=ytv, random_state=SEED)

                clf = analyzer.train_classifier(
                    Xtr, ytr, Xval, yval, li, model_name)
                res = analyzer.evaluate_group(clf, Xte, yte, "FreqBinary")

                binary_results.append({
                    'layer': li, 'accuracy': res['accuracy'],
                    'precision': res['precision'], 'recall': res['recall'],
                    'f1': res['f1'], 'auc': res['auc'],
                    'n_high': hfm.sum(), 'n_low': lfm.sum(),
                })
                del clf
            except Exception as e:
                logger.warning(f"  Binary freq probe layer {li} failed: {e}")
                binary_results.append({
                    'layer': li, 'accuracy': np.nan, 'f1': np.nan,
                    'auc': np.nan, 'n_high': hfm.sum(), 'n_low': lfm.sum()
                })

            # ── 3-way probe ───────────────────────────────────────────
            X_3w = all_hidden[li][three_way_mask]
            y_3w = three_way_labels[three_way_mask]

            if len(np.unique(y_3w)) < 3:
                three_way_results.append({
                    'layer': li, 'accuracy': np.nan, 'f1_macro': np.nan
                })
                continue

            try:
                sc = StandardScaler()
                Xtv3, Xte3, ytv3, yte3 = train_test_split(
                    X_3w, y_3w, test_size=self.config.TEST_SIZE,
                    stratify=y_3w, random_state=SEED)
                vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
                Xtr3, Xval3, ytr3, yval3 = train_test_split(
                    Xtv3, ytv3, test_size=vsz, stratify=ytv3, random_state=SEED)

                Xtr3_s = sc.fit_transform(Xtr3)
                Xte3_s = sc.transform(Xte3)

                lr3 = LogisticRegression(
                    max_iter=1000, multi_class='multinomial',
                    solver='lbfgs', random_state=SEED, C=1.0)
                lr3.fit(Xtr3_s, ytr3)
                y3_pred = lr3.predict(Xte3_s)
                acc3 = accuracy_score(yte3, y3_pred)
                _, _, f1_3, _ = precision_recall_fscore_support(
                    yte3, y3_pred, average='macro', zero_division=0)

                three_way_results.append({
                    'layer': li, 'accuracy': acc3, 'f1_macro': f1_3,
                    'chance': 1.0 / 3.0,
                })
            except Exception as e:
                logger.warning(f"  3-way freq probe layer {li} failed: {e}")
                three_way_results.append({
                    'layer': li, 'accuracy': np.nan, 'f1_macro': np.nan
                })

        return {'binary': binary_results, 'three_way': three_way_results}

    # ────────────────────────────────────────────────────────────────────
    # Analysis 2: Tokenization-Controlled Rerun
    # ────────────────────────────────────────────────────────────────────
    def tokenization_controlled_analysis(self, all_hidden, targets,
                                          model_name) -> Dict:
        """
        Control for tokenization confounds by:
        1. Running the LDT probe on SINGLE-TOKEN stimuli only.
        2. Running on token-count-matched high vs low subsets.

        Rationale: High-frequency words tend to be single subword tokens
        while low-frequency words are split into multiple subwords. For
        last-token pooling, multi-token words have their representation
        at a token that may not capture the full word meaning, creating
        a systematic confound between tokenization and frequency.
        """
        logger.info(f"  [Analysis 2] Tokenization-controlled — {model_name}")

        is_word        = targets['is_word']
        is_high_freq   = targets['is_high_freq']
        is_low_freq    = targets['is_low_freq']
        freq_groups    = targets['freq_group']
        is_single_tok  = targets['is_single_token']
        token_counts   = targets['token_count']

        wm  = (is_word == 1)
        hfm = wm & (is_high_freq == 1)
        lfm = wm & (is_low_freq  == 1)

        # ── Stats table ───────────────────────────────────────────────
        hf_single = (hfm & (is_single_tok == 1)).sum()
        lf_single = (lfm & (is_single_tok == 1)).sum()
        hf_tc_mean = token_counts[hfm].mean() if hfm.sum() > 0 else np.nan
        lf_tc_mean = token_counts[lfm].mean() if lfm.sum() > 0 else np.nan

        token_stats = {
            'high_freq_total': int(hfm.sum()),
            'low_freq_total': int(lfm.sum()),
            'high_freq_single_token': int(hf_single),
            'low_freq_single_token': int(lf_single),
            'high_freq_pct_single': float(hf_single / max(hfm.sum(), 1) * 100),
            'low_freq_pct_single': float(lf_single / max(lfm.sum(), 1) * 100),
            'high_freq_mean_tokens': float(hf_tc_mean),
            'low_freq_mean_tokens': float(lf_tc_mean),
        }
        logger.info(f"  Token stats: HF single={hf_single}/{hfm.sum()}, "
                    f"LF single={lf_single}/{lfm.sum()}")

        analyzer = FrequencyAnalyzer(self.config)

        # ── Single-token-only subset ──────────────────────────────────
        single_mask = (is_single_tok == 1)
        single_ldt_results = []

        if single_mask.sum() >= 2 * self.config.MIN_SAMPLES_PER_GROUP:
            for li in tqdm(range(len(all_hidden)),
                           desc="Single-token LDT"):
                if li not in all_hidden:
                    continue
                # Build targets subset
                tgt_sub = {k: (v[single_mask] if isinstance(v, np.ndarray) else
                               [v[i] for i in range(len(v)) if single_mask[i]])
                           for k, v in targets.items()}
                try:
                    res = analyzer.analyze_layer(
                        all_hidden[li][single_mask], tgt_sub, li, model_name)
                    single_ldt_results.append(res)
                except Exception as e:
                    logger.warning(f"  Single-token layer {li} failed: {e}")
                    single_ldt_results.append(analyzer._empty_result(li))
        else:
            logger.warning("  Not enough single-token stimuli for subset analysis")

        # ── Token-count-matched high vs low ───────────────────────────
        matched_results = []
        # Match by finding the common token counts between high & low
        hf_indices = np.where(hfm)[0]
        lf_indices = np.where(lfm)[0]
        hf_tcounts = token_counts[hf_indices]
        lf_tcounts = token_counts[lf_indices]

        # For each token count, take min(n_high, n_low) from each group
        matched_hf_idx = []
        matched_lf_idx = []
        for tc in np.unique(np.concatenate([hf_tcounts, lf_tcounts])):
            hf_with_tc = hf_indices[hf_tcounts == tc]
            lf_with_tc = lf_indices[lf_tcounts == tc]
            n_match = min(len(hf_with_tc), len(lf_with_tc))
            if n_match > 0:
                rng = np.random.RandomState(SEED)
                matched_hf_idx.extend(rng.choice(hf_with_tc, n_match, replace=False))
                matched_lf_idx.extend(rng.choice(lf_with_tc, n_match, replace=False))

        if len(matched_hf_idx) >= self.config.MIN_SAMPLES_PER_GROUP:
            matched_word_idx = np.array(matched_hf_idx + matched_lf_idx)
            # Also include nonwords to maintain LDT framing
            nw_idx = np.where(is_word == 0)[0]
            matched_all_idx = np.concatenate([matched_word_idx, nw_idx])

            for li in tqdm(range(len(all_hidden)),
                           desc="Token-matched LDT"):
                if li not in all_hidden:
                    continue
                tgt_sub = {k: (v[matched_all_idx] if isinstance(v, np.ndarray) else
                               [v[i] for i in matched_all_idx])
                           for k, v in targets.items()}
                try:
                    res = analyzer.analyze_layer(
                        all_hidden[li][matched_all_idx], tgt_sub, li, model_name)
                    matched_results.append(res)
                except Exception as e:
                    logger.warning(f"  Token-matched layer {li} failed: {e}")
                    matched_results.append(analyzer._empty_result(li))

            logger.info(f"  Token-matched: {len(matched_hf_idx)} high, "
                        f"{len(matched_lf_idx)} low (matched)")
        else:
            logger.warning("  Not enough token-count-matched samples")

        return {
            'token_stats': token_stats,
            'single_token_results': single_ldt_results,
            'token_matched_results': matched_results,
        }

    # ────────────────────────────────────────────────────────────────────
    # Analysis 3: Lexical-Confound-Matched Frequency Comparisons
    # ────────────────────────────────────────────────────────────────────
    def confound_matched_analysis(self, all_hidden, targets,
                                   model_name) -> Dict:
        """
        Match high- and low-frequency words on character length,
        orthographic neighborhood density (Ortho_N), and token count.
        Then rerun the direct frequency probe on matched subsets.

        Uses nearest-neighbor matching without replacement: for each
        low-frequency word, find the high-frequency word closest in
        the [length, ortho_n, token_count] feature space. This ensures
        any probe accuracy differences cannot be attributed to these
        lexical confounds.

        Also produces a balance table showing group statistics before
        and after matching for the paper's methods section.
        """
        logger.info(f"  [Analysis 3] Confound-matched frequency — {model_name}")

        is_word      = targets['is_word']
        is_high_freq = targets['is_high_freq']
        is_low_freq  = targets['is_low_freq']
        lengths      = targets['length']
        ortho_ns     = targets['ortho_n']
        token_counts = targets['token_count']

        wm  = (is_word == 1)
        hfm = wm & (is_high_freq == 1)
        lfm = wm & (is_low_freq  == 1)

        hf_idx = np.where(hfm)[0]
        lf_idx = np.where(lfm)[0]

        # Build feature matrix for matching
        def _feat(idx):
            l = lengths[idx].astype(float)
            o = ortho_ns[idx].astype(float)
            t = token_counts[idx].astype(float)
            # Replace NaN with median for matching
            l[np.isnan(l)] = np.nanmedian(l) if np.any(~np.isnan(l)) else 0
            o[np.isnan(o)] = np.nanmedian(o) if np.any(~np.isnan(o)) else 0
            return np.column_stack([l, o, t])

        hf_feats = _feat(hf_idx)
        lf_feats = _feat(lf_idx)

        # Standardise features for distance computation
        all_feats = np.vstack([hf_feats, lf_feats])
        feat_mean = all_feats.mean(axis=0)
        feat_std  = all_feats.std(axis=0)
        feat_std[feat_std == 0] = 1.0

        hf_feats_s = (hf_feats - feat_mean) / feat_std
        lf_feats_s = (lf_feats - feat_mean) / feat_std

        # Greedy nearest-neighbour matching (low → high, without replacement)
        from scipy.spatial.distance import cdist
        dists = cdist(lf_feats_s, hf_feats_s, metric='euclidean')

        matched_lf = []
        matched_hf = []
        used_hf = set()

        # Sort low-freq by best available match distance
        for lf_i in range(len(lf_idx)):
            row = dists[lf_i].copy()
            row[list(used_hf)] = np.inf
            best_hf = np.argmin(row)
            if row[best_hf] < np.inf:
                matched_lf.append(lf_idx[lf_i])
                matched_hf.append(hf_idx[best_hf])
                used_hf.add(best_hf)

        matched_lf = np.array(matched_lf)
        matched_hf = np.array(matched_hf)

        # ── Balance table ─────────────────────────────────────────────
        def _group_stats(idx, label):
            return {
                'group': label,
                'n': len(idx),
                'length_mean': float(np.nanmean(lengths[idx])),
                'length_std':  float(np.nanstd(lengths[idx])),
                'ortho_n_mean': float(np.nanmean(ortho_ns[idx])),
                'ortho_n_std':  float(np.nanstd(ortho_ns[idx])),
                'token_count_mean': float(np.mean(token_counts[idx])),
                'token_count_std':  float(np.std(token_counts[idx])),
            }

        balance_before = [
            _group_stats(hf_idx, 'high_freq_before'),
            _group_stats(lf_idx, 'low_freq_before'),
        ]
        balance_after = [
            _group_stats(matched_hf, 'high_freq_after'),
            _group_stats(matched_lf, 'low_freq_after'),
        ]

        logger.info(f"  Matched pairs: {len(matched_hf)}")

        # ── Run direct frequency probe on matched subset ──────────────
        matched_probe_results = []
        if len(matched_hf) >= self.config.MIN_SAMPLES_PER_GROUP:
            matched_all = np.concatenate([matched_hf, matched_lf])
            matched_labels = np.concatenate([
                np.ones(len(matched_hf), dtype=int),
                np.zeros(len(matched_lf), dtype=int)
            ])

            for li in tqdm(range(len(all_hidden)),
                           desc="Confound-matched freq probe"):
                if li not in all_hidden:
                    continue
                X_m = all_hidden[li][matched_all]
                y_m = matched_labels

                try:
                    Xtv, Xte, ytv, yte = train_test_split(
                        X_m, y_m, test_size=self.config.TEST_SIZE,
                        stratify=y_m, random_state=SEED)
                    vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
                    Xtr, Xval, ytr, yval = train_test_split(
                        Xtv, ytv, test_size=vsz, stratify=ytv, random_state=SEED)

                    fa = FrequencyAnalyzer(self.config)
                    clf = fa.train_classifier(Xtr, ytr, Xval, yval, li, model_name)
                    res = fa.evaluate_group(clf, Xte, yte, "ConfoundMatched")

                    matched_probe_results.append({
                        'layer': li, 'accuracy': res['accuracy'],
                        'f1': res['f1'], 'auc': res['auc'],
                        'n_matched_pairs': len(matched_hf),
                    })
                    del clf
                except Exception as e:
                    logger.warning(f"  Confound-matched layer {li} failed: {e}")
                    matched_probe_results.append({
                        'layer': li, 'accuracy': np.nan, 'f1': np.nan,
                        'auc': np.nan
                    })
        else:
            logger.warning("  Not enough matched pairs for confound-matched analysis")

        return {
            'balance_before': balance_before,
            'balance_after': balance_after,
            'n_matched_pairs': len(matched_hf),
            'matched_probe_results': matched_probe_results,
        }

    # ────────────────────────────────────────────────────────────────────
    # Analysis 4: Multi-Seed Stability
    # ────────────────────────────────────────────────────────────────────
    def multi_seed_stability(self, all_hidden, targets, model_name,
                              n_seeds: int = 5) -> Dict:
        """
        Repeat the main LDT probe pipeline across multiple random seeds,
        using different train/val/test splits and weight initialisations.
        Report mean ± std per layer.

        This addresses the concern that results from a single 70/15/15
        split may be unstable. Seeds control:
          1. train_test_split random_state
          2. PyTorch weight initialisation (torch.manual_seed)
          3. WeightedRandomSampler ordering

        The reported confidence interval width directly indicates whether
        the frequency effect is robust to data sampling variance.
        """
        logger.info(f"  [Analysis 4] Multi-seed stability ({n_seeds} seeds) "
                    f"— {model_name}")

        analyzer = FrequencyAnalyzer(self.config)
        seeds = [SEED + i * 7 for i in range(n_seeds)]  # Deterministic seed sequence

        # Collect per-seed, per-layer results
        # We only probe a representative subset of layers to save time:
        # first, middle, last, and every 4th layer
        all_layers = sorted(all_hidden.keys())
        probe_layers = sorted(set(
            [all_layers[0], all_layers[len(all_layers)//4],
             all_layers[len(all_layers)//2],
             all_layers[3*len(all_layers)//4],
             all_layers[-1]]
            + all_layers[::4]
        ))

        seed_results = {s: [] for s in seeds}

        for seed_i, seed in enumerate(seeds):
            logger.info(f"  Seed {seed_i+1}/{n_seeds} (seed={seed})")
            for li in tqdm(probe_layers,
                           desc=f"Seed {seed_i+1}/{n_seeds}"):
                if li not in all_hidden:
                    continue
                res = analyzer.analyze_layer(
                    all_hidden[li], targets, li, model_name, seed=seed)
                seed_results[seed].append(res)

        # Aggregate: mean ± std per layer
        layer_stats = []
        for layer_pos, li in enumerate(probe_layers):
            accs = []
            f1s = []
            freq_diffs = []
            cohens_hs = []

            for seed in seeds:
                if layer_pos < len(seed_results[seed]):
                    r = seed_results[seed][layer_pos]
                    if r.get('overall'):
                        accs.append(r['overall']['accuracy'])
                        f1s.append(r['overall']['f1'])
                    fe = r['frequency_effect']
                    if not np.isnan(fe['accuracy_difference']):
                        freq_diffs.append(fe['accuracy_difference'])
                    if not np.isnan(fe.get('cohens_h', np.nan)):
                        cohens_hs.append(fe['cohens_h'])

            layer_stats.append({
                'layer': li,
                'acc_mean': float(np.mean(accs)) if accs else np.nan,
                'acc_std':  float(np.std(accs))  if accs else np.nan,
                'f1_mean':  float(np.mean(f1s))  if f1s  else np.nan,
                'f1_std':   float(np.std(f1s))   if f1s  else np.nan,
                'freq_diff_mean': float(np.mean(freq_diffs)) if freq_diffs else np.nan,
                'freq_diff_std':  float(np.std(freq_diffs))  if freq_diffs else np.nan,
                'cohens_h_mean':  float(np.mean(cohens_hs))  if cohens_hs  else np.nan,
                'cohens_h_std':   float(np.std(cohens_hs))   if cohens_hs  else np.nan,
                'n_seeds': len(accs),
            })

        return {
            'n_seeds': n_seeds,
            'seeds': seeds,
            'layer_stats': layer_stats,
        }

    # ────────────────────────────────────────────────────────────────────
    # Analysis 6: Probe Selectivity Controls
    # ────────────────────────────────────────────────────────────────────
    def probe_selectivity_controls(self, all_hidden, targets,
                                    model_name) -> Dict:
        """
        Three control conditions to establish probe selectivity
        (Hewitt & Liang, 2019):

        1. Linear probe baseline:
           Replace the MLP probe with logistic regression.
           If a linear probe also finds the effect, the information
           is linearly accessible (stronger claim). If only the MLP
           finds it, the information may be non-linearly encoded
           (weaker claim about explicit representation).

        2. Shuffled-label control:
           Train the MLP probe with randomly permuted labels.
           This establishes the ceiling for memorisation/overfitting.
           If the shuffled probe achieves high accuracy, the probe
           architecture has too much capacity relative to the dataset
           and results are not trustworthy.

        3. Selectivity (accuracy_task − accuracy_shuffled):
           A positive selectivity score indicates the probe is
           extracting genuine linguistic signal rather than just
           memorising arbitrary patterns. Hewitt & Liang recommend
           selectivity > 0 with a comfortable margin.
        """
        logger.info(f"  [Analysis 6] Probe selectivity controls — {model_name}")

        is_word = targets['is_word']
        analyzer = FrequencyAnalyzer(self.config)

        linear_results = []
        shuffled_results = []
        selectivity_results = []

        all_layers = sorted(all_hidden.keys())
        # Probe representative layers
        probe_layers = sorted(set(
            [all_layers[0], all_layers[len(all_layers)//4],
             all_layers[len(all_layers)//2],
             all_layers[3*len(all_layers)//4],
             all_layers[-1]]
            + all_layers[::4]
        ))

        for li in tqdm(probe_layers, desc="Selectivity controls"):
            if li not in all_hidden:
                continue
            X = all_hidden[li]
            y = is_word

            # Split
            try:
                strat = y
                Xtv, Xte, ytv, yte = train_test_split(
                    X, y, test_size=self.config.TEST_SIZE,
                    stratify=strat, random_state=SEED)
                vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
                Xtr, Xval, ytr, yval = train_test_split(
                    Xtv, ytv, test_size=vsz, stratify=ytv, random_state=SEED)
            except ValueError:
                continue

            sc = StandardScaler()
            Xtr_s  = sc.fit_transform(Xtr)
            Xte_s  = sc.transform(Xte)
            Xval_s = sc.transform(Xval)

            # ── (a) Linear probe ──────────────────────────────────────
            try:
                lr = LogisticRegression(
                    max_iter=1000, solver='lbfgs', random_state=SEED, C=1.0)
                lr.fit(Xtr_s, ytr)
                y_pred_lr = lr.predict(Xte_s)
                y_prob_lr = lr.predict_proba(Xte_s)[:, 1]
                acc_linear = accuracy_score(yte, y_pred_lr)
                _, _, f1_linear, _ = precision_recall_fscore_support(
                    yte, y_pred_lr, average='binary', zero_division=0)
                try:
                    auc_linear = roc_auc_score(yte, y_prob_lr)
                except:
                    auc_linear = np.nan

                linear_results.append({
                    'layer': li, 'accuracy': acc_linear,
                    'f1': f1_linear, 'auc': auc_linear
                })
            except Exception as e:
                logger.warning(f"  Linear probe layer {li} failed: {e}")
                linear_results.append({
                    'layer': li, 'accuracy': np.nan, 'f1': np.nan,
                    'auc': np.nan
                })

            # ── (b) Shuffled-label control ────────────────────────────
            try:
                rng = np.random.RandomState(SEED)
                ytr_shuffled = rng.permutation(ytr)
                yval_shuffled = rng.permutation(yval)

                clf_shuf = analyzer.train_classifier(
                    Xtr, ytr_shuffled, Xval, yval_shuffled,
                    li, model_name, seed=SEED)
                res_shuf = analyzer.evaluate_group(
                    clf_shuf, Xte, yte, "Shuffled")

                shuffled_results.append({
                    'layer': li, 'accuracy': res_shuf['accuracy'],
                    'f1': res_shuf['f1'], 'auc': res_shuf['auc'],
                })
                del clf_shuf
            except Exception as e:
                logger.warning(f"  Shuffled probe layer {li} failed: {e}")
                shuffled_results.append({
                    'layer': li, 'accuracy': np.nan, 'f1': np.nan,
                    'auc': np.nan
                })

            # ── (c) Selectivity ───────────────────────────────────────
            # Train normal MLP probe for this layer too
            try:
                clf_task = analyzer.train_classifier(
                    Xtr, ytr, Xval, yval, li, model_name, seed=SEED)
                res_task = analyzer.evaluate_group(clf_task, Xte, yte, "Task")

                acc_task = res_task['accuracy']
                acc_shuf = shuffled_results[-1]['accuracy']
                selectivity = acc_task - acc_shuf if not np.isnan(acc_shuf) else np.nan

                selectivity_results.append({
                    'layer': li,
                    'acc_task': acc_task,
                    'acc_shuffled': acc_shuf,
                    'selectivity': selectivity,
                })
                del clf_task
            except Exception as e:
                selectivity_results.append({
                    'layer': li, 'acc_task': np.nan,
                    'acc_shuffled': np.nan, 'selectivity': np.nan,
                })

        return {
            'linear_probe': linear_results,
            'shuffled_label': shuffled_results,
            'selectivity': selectivity_results,
        }

    # ────────────────────────────────────────────────────────────────────
    # Analysis 7: Continuous Frequency Regression
    # ────────────────────────────────────────────────────────────────────
    def continuous_frequency_regression(self, all_hidden, targets,
                                         model_name) -> Dict:
        """
        Predict continuous log_HAL frequency from hidden states using
        ridge regression. This provides stronger psycholinguistic evidence
        than tertile separation because it treats frequency as a continuous
        variable and avoids information loss from discretisation.

        R² and Spearman correlation are reported per layer.
        A positive R² means the hidden states carry frequency information
        beyond what a constant (mean) prediction would give.

        Ridge regression (L2 penalty) is chosen over OLS because
        hidden-state dimensionality (e.g. 2048) may exceed training
        samples, and L2 regularisation prevents degenerate coefficient
        inflation. Alpha is selected from {0.1, 1, 10, 100} by
        validation-set R².
        """
        logger.info(f"  [Analysis 7] Continuous frequency regression — {model_name}")

        is_word      = targets['is_word']
        log_freq     = targets['log_frequency']

        # Words with valid frequency only
        wm = (is_word == 1) & (~np.isnan(log_freq))

        if wm.sum() < 2 * self.config.MIN_SAMPLES_PER_GROUP:
            logger.warning("  Not enough words with frequency for regression")
            return {'regression_results': []}

        y_freq = log_freq[wm]

        regression_results = []

        for li in tqdm(range(len(all_hidden)),
                       desc="Continuous freq regression"):
            if li not in all_hidden:
                continue
            X = all_hidden[li][wm]

            try:
                Xtv, Xte, ytv, yte = train_test_split(
                    X, y_freq, test_size=self.config.TEST_SIZE,
                    random_state=SEED)
                vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
                Xtr, Xval, ytr, yval = train_test_split(
                    Xtv, ytv, test_size=vsz, random_state=SEED)

                sc = StandardScaler()
                Xtr_s  = sc.fit_transform(Xtr)
                Xval_s = sc.transform(Xval)
                Xte_s  = sc.transform(Xte)

                # Select alpha by validation R²
                best_alpha, best_val_r2 = 1.0, -np.inf
                for alpha in [0.1, 1.0, 10.0, 100.0]:
                    ridge = Ridge(alpha=alpha, random_state=SEED)
                    ridge.fit(Xtr_s, ytr)
                    val_r2 = ridge.score(Xval_s, yval)
                    if val_r2 > best_val_r2:
                        best_alpha = alpha
                        best_val_r2 = val_r2

                ridge = Ridge(alpha=best_alpha, random_state=SEED)
                ridge.fit(Xtr_s, ytr)
                y_pred = ridge.predict(Xte_s)

                r2   = r2_score(yte, y_pred)
                rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))
                spearman_r, spearman_p = stats.spearmanr(yte, y_pred)

                regression_results.append({
                    'layer': li,
                    'r2': float(r2),
                    'rmse': rmse,
                    'spearman_r': float(spearman_r),
                    'spearman_p': float(spearman_p),
                    'best_alpha': best_alpha,
                    'n_train': len(ytr),
                    'n_test': len(yte),
                })
            except Exception as e:
                logger.warning(f"  Regression layer {li} failed: {e}")
                regression_results.append({
                    'layer': li, 'r2': np.nan, 'rmse': np.nan,
                    'spearman_r': np.nan, 'spearman_p': np.nan,
                })

        return {'regression_results': regression_results}


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ════════════════════════════════════════════════════════════════════════════

class MultiModelExperiment:

    OVERALL_C = '#1A535C'
    HIGH_C    = '#2E86AB'
    LOW_C     = '#D62246'
    DIFF_SIG  = '#2DC653'
    DIFF_NS   = '#AAAAAA'

    def __init__(self, config: Config):
        self.config = config
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.3)
        plt.rcParams.update({'figure.dpi':150, 'savefig.dpi':300})

    # ── entry point ───────────────────────────────────────────────────────
    def run(self):
        t0 = datetime.now()
        print(f"\n{'='*64}\nMULTI-MODEL LDT v6 — WITH ALL EXTENDED ANALYSES\n"
              f"Started: {t0:%Y-%m-%d %H:%M:%S}\n{'='*64}\n")

        words_df    = pd.read_csv(self.config.WORDS_PATH)
        nonwords_df = pd.read_csv(self.config.NONWORDS_PATH)
        logger.info(f"Data: {len(words_df):,} words  {len(nonwords_df):,} nonwords")

        all_results = {}
        for mc in self.config.MODELS:
            print(f"\n{'='*64}\nMODEL: {mc.name}\n{'='*64}")
            try:
                res = self._process_model(mc, words_df, nonwords_df)
                all_results[mc.name] = res
                self._save_model_csv(mc.name, res['layer_results'])
                self._save_extended_csvs(mc.name, res)
                self._plot_model_all(mc.name, res['layer_results'])
                self._plot_extended(mc.name, res)
                if self.config.CLEAR_CACHE_BETWEEN_MODELS:
                    torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                logger.error(f"Failed {mc.name}: {e}")
                import traceback; traceback.print_exc()

        if len(all_results) > 1:
            self._plot_cross_model(all_results)
            self._save_cross_model_csv(all_results)

        t1 = datetime.now()
        print(f"\n{'='*64}\nDONE  ({t1-t0})  →  {self.config.OUTPUT_DIR}\n{'='*64}\n")
        return all_results

    # ── per-model pipeline ────────────────────────────────────────────────
    def _process_model(self, mc, words_df, nonwords_df) -> Dict:
        extractor = AllLayerExtractor(mc, self.config.DEVICE)
        dataset   = LDTDataset(words_df, nonwords_df, extractor.tokenizer, self.config)
        dataloader = DataLoader(dataset, batch_size=mc.batch_size,
                                shuffle=False, num_workers=0)

        # ════════════════════════════════════════════════════════════════
        # STEP 1: Extract hidden states with PRIMARY pooling (last_token)
        # ════════════════════════════════════════════════════════════════
        print(f"\n  [Step 1/4] Extracting hidden states (last_token) …")
        t_ext = datetime.now()
        all_hidden, targets = extractor.extract_all_layers(
            dataloader, pooling_strategy="last_token")
        print(f"  Extraction done in {datetime.now()-t_ext}\n")

        # ════════════════════════════════════════════════════════════════
        # Analysis 5: Pooling Ablation — extract with mean_pool too
        # ════════════════════════════════════════════════════════════════
        print(f"  [Step 1b/4] Extracting hidden states (mean_pool) …")
        t_ext2 = datetime.now()
        all_hidden_mean, _ = extractor.extract_all_layers(
            dataloader, pooling_strategy="mean_pool")
        print(f"  Mean-pool extraction done in {datetime.now()-t_ext2}\n")

        # ── free GPU memory before probe training ─────────────────────
        del extractor.model
        if self.config.DEVICE == 'cuda':
            torch.cuda.empty_cache(); torch.cuda.synchronize(); gc.collect()
            alloc = torch.cuda.memory_allocated(0)/1024**3
            logger.info(f"  GPU after model unload: alloc={alloc:.2f}GB")

        # ════════════════════════════════════════════════════════════════
        # STEP 2: Train one probe per layer (original LDT pipeline)
        # ════════════════════════════════════════════════════════════════
        num_layers = extractor.num_layers
        print(f"  [Step 2/4] Training LDT probe classifiers ({num_layers} layers) …")
        t_probe = datetime.now()
        analyzer      = FrequencyAnalyzer(self.config)
        layer_results = []

        for li in tqdm(range(num_layers), desc=f"LDT Probe | {mc.name}"):
            res = analyzer.analyze_layer(all_hidden[li], targets, li, mc.name)
            layer_results.append(res)

        print(f"  LDT Probe training done in {datetime.now()-t_probe}\n")
        self._apply_fdr(layer_results)

        # ════════════════════════════════════════════════════════════════
        # Analysis 5b: Mean-pool LDT probes
        # ════════════════════════════════════════════════════════════════
        print(f"  [Step 2b/4] Training mean-pool LDT probes …")
        mean_pool_results = []
        for li in tqdm(range(num_layers), desc=f"Mean-pool LDT | {mc.name}"):
            res = analyzer.analyze_layer(
                all_hidden_mean[li], targets, li, mc.name)
            mean_pool_results.append(res)
        self._apply_fdr(mean_pool_results)

        # Free mean-pool hidden states
        del all_hidden_mean
        gc.collect()

        # ════════════════════════════════════════════════════════════════
        # STEP 3: Extended Analyses (1–4, 6–7)
        # ════════════════════════════════════════════════════════════════
        print(f"\n  [Step 3/4] Running extended analyses …")
        ext = ExtendedAnalyses(self.config)

        # Analysis 1: Direct frequency probe
        direct_freq = ext.direct_frequency_probe(all_hidden, targets, mc.name)

        # Analysis 2: Tokenization-controlled
        token_ctrl = ext.tokenization_controlled_analysis(
            all_hidden, targets, mc.name)

        # Analysis 3: Confound-matched
        confound = ext.confound_matched_analysis(
            all_hidden, targets, mc.name)

        # Analysis 4: Multi-seed stability
        stability = ext.multi_seed_stability(
            all_hidden, targets, mc.name,
            n_seeds=self.config.N_SEEDS)

        # Analysis 6: Probe selectivity controls
        selectivity = ext.probe_selectivity_controls(
            all_hidden, targets, mc.name)

        # Analysis 7: Continuous frequency regression
        regression = ext.continuous_frequency_regression(
            all_hidden, targets, mc.name)

        # ── Free remaining hidden states ──────────────────────────────
        for li in list(all_hidden.keys()):
            del all_hidden[li]
        gc.collect()

        del extractor.tokenizer, extractor
        gc.collect()

        return {
            'model_config': mc,
            'layer_results': layer_results,
            'num_layers': num_layers,
            # Extended analyses
            'mean_pool_results': mean_pool_results,
            'direct_freq': direct_freq,
            'token_ctrl': token_ctrl,
            'confound_matched': confound,
            'stability': stability,
            'selectivity': selectivity,
            'regression': regression,
        }

    def _apply_fdr(self, lr):
        pv, ix = [], []
        for i, r in enumerate(lr):
            p = r['frequency_effect']['p_value']
            if not np.isnan(p): pv.append(p); ix.append(i)
        if not pv: return
        rej, pc, _, _ = multipletests(pv, alpha=self.config.ALPHA, method='fdr_bh')
        for i, p, s in zip(ix, pc, rej):
            lr[i]['frequency_effect']['p_value_corrected'] = p
            lr[i]['frequency_effect']['significant_fdr']   = bool(s)

    # ── metric extraction helper ──────────────────────────────────────────
    @staticmethod
    def _series(lr, group_key, metric):
        out = []
        for r in lr:
            g = r.get(group_key)
            out.append(g[metric] if g else np.nan)
        return np.array(out, dtype=float)

    @staticmethod
    def _L(lr): return [r['layer'] for r in lr]

    # ════════════════════════════════════════════════════════════════════════
    # PER-MODEL PLOTS (original)
    # ════════════════════════════════════════════════════════════════════════

    def _plot_model_all(self, model_name, lr):
        safe = model_name.replace(' ','_').replace('/','_')
        L    = self._L(lr)

        groups = {
            'overall':       'overall',
            'high_frequency':'high_frequency',
            'low_frequency': 'low_frequency',
        }
        metrics = ['accuracy','precision','recall','f1','auc']

        data = {g: {m: self._series(lr,g,m) for m in metrics} for g in groups}
        fe   = [r['frequency_effect']['accuracy_difference'] for r in lr]
        pc   = [r['frequency_effect'].get('p_value_corrected', np.nan) for r in lr]
        ch   = [r['frequency_effect'].get('cohens_h', np.nan)          for r in lr]

        ylabels = {'accuracy':'Accuracy','precision':'Precision',
                   'recall':'Recall','f1':'F1-Score','auc':'AUC-ROC'}
        for m in metrics:
            ov = data['overall'][m]
            hf = data['high_frequency'][m]
            lf = data['low_frequency'][m]
            self._three_group_plot(
                L, ov, hf, lf,
                ylabel=ylabels[m],
                title=f'{ylabels[m]} — All Groups  [{model_name}]',
                fname=os.path.join(self.config.FIGURES_DIR,
                                   f'{safe}_{m}_all_groups.png'))

        self._freq_effect_plot(L, fe, pc, ch, model_name, safe)
        self._all_metrics_grid(model_name, safe, L, data, fe, pc, ch)

    def _three_group_plot(self, L, ov, hf, lf, ylabel, title, fname):
        valid = [v for v in list(ov)+list(hf)+list(lf) if not np.isnan(v)]
        ymin  = max(0.0, min(valid, default=0.4) - 0.05) if valid else 0.4
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(L, ov, 'o-',  lw=2.5, ms=7, color=self.OVERALL_C, label='Overall Test', zorder=3)
        ax.plot(L, hf, 's--', lw=2.5, ms=7, color=self.HIGH_C,    label='High-Freq',    zorder=3)
        ax.plot(L, lf, '^-.', lw=2.5, ms=7, color=self.LOW_C,     label='Low-Freq',     zorder=3)
        ax.fill_between(L, hf, lf, alpha=0.12, color='#F18F01', label='High−Low gap')
        ax.axhline(0.5, color='gray', ls=':', lw=1.2, alpha=0.6)
        ax.set(xlabel='Layer Index', ylabel=ylabel, title=title, ylim=[ymin, 1.01])
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.close('all'); gc.collect()
        logger.info(f"✓ {os.path.basename(fname)}")

    def _freq_effect_plot(self, L, diff, p_corr, coh_h, model_name, safe):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        colors = [self.DIFF_SIG if (not np.isnan(p) and p<0.05) else self.DIFF_NS
                  for p in p_corr]
        ax1.bar(L, diff, color=colors, alpha=0.75, edgecolor='black', lw=0.6)
        ax1.axhline(0, color='black', lw=1.5)
        for li, d, p in zip(L, diff, p_corr):
            if np.isnan(p): continue
            s = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ''))
            if s:
                ax1.text(li, d, s, ha='center',
                         va='bottom' if d>0 else 'top', fontsize=10, fontweight='bold')
        ax1.set(xlabel='Layer', ylabel='Accuracy Diff (High − Low)',
                title=f'(A) Frequency Effect — {model_name}')
        ax1.grid(True, alpha=0.3, axis='y')
        from matplotlib.patches import Patch
        ax1.legend(handles=[Patch(fc=self.DIFF_SIG, label='p<0.05 FDR'),
                             Patch(fc=self.DIFF_NS,  label='n.s.')], fontsize=10)

        ax2.plot(L, coh_h, 'o-', lw=2.5, ms=7, color='#5C4B8A')
        for h, lbl, c in [(0.2,'Small','#e57373'),(0.5,'Medium','#ffa726'),(0.8,'Large','#66bb6a')]:
            ax2.axhline(h, color=c, ls='--', lw=1.4, alpha=0.7, label=f'{lbl} h={h}')
        ax2.axhline(0, color='black', lw=1.2)
        ax2.set(xlabel='Layer', ylabel="Cohen's h", ylim=[-0.05, None],
                title=f"(B) Effect Size — {model_name}")
        ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                 f'{safe}_frequency_effect.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all'); gc.collect()
        logger.info(f"✓ {safe}_frequency_effect.png")

    def _all_metrics_grid(self, model_name, safe, L, data, diff, p_corr, coh_h):
        fig = plt.figure(figsize=(28, 14))
        gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.42, wspace=0.35)

        metric_info = [
            (gs[0,0], 'accuracy',  'Accuracy'),
            (gs[0,1], 'precision', 'Precision'),
            (gs[0,2], 'recall',    'Recall'),
            (gs[0,3], 'f1',        'F1-Score'),
            (gs[0,4], 'auc',       'AUC-ROC'),
        ]
        for gspec, m, ylabel in metric_info:
            ax = fig.add_subplot(gspec)
            ax.plot(L, data['overall'][m],        'o-',  lw=2, ms=5, color=self.OVERALL_C, label='Overall')
            ax.plot(L, data['high_frequency'][m],  's--', lw=2, ms=5, color=self.HIGH_C,    label='High-Freq')
            ax.plot(L, data['low_frequency'][m],   '^-.', lw=2, ms=5, color=self.LOW_C,     label='Low-Freq')
            ax.fill_between(L, data['high_frequency'][m], data['low_frequency'][m],
                            alpha=0.10, color='#F18F01')
            ax.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.5)
            ax.set(xlabel='Layer', ylabel=ylabel, title=ylabel)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

        ax_b = fig.add_subplot(gs[1, 0:2])
        colors = [self.DIFF_SIG if (not np.isnan(p) and p<0.05) else self.DIFF_NS for p in p_corr]
        ax_b.bar(L, diff, color=colors, alpha=0.75, edgecolor='black', lw=0.5)
        ax_b.axhline(0, color='black', lw=1.3)
        for li, d, p in zip(L, diff, p_corr):
            if np.isnan(p): continue
            s = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ''))
            if s:
                ax_b.text(li, d, s, ha='center', va='bottom' if d>0 else 'top',
                          fontsize=9, fontweight='bold')
        ax_b.set(xlabel='Layer', ylabel='Acc Diff (High − Low)',
                 title='Frequency Effect (FDR corrected)')
        ax_b.grid(True, alpha=0.25, axis='y')
        from matplotlib.patches import Patch
        ax_b.legend(handles=[Patch(fc=self.DIFF_SIG, label='p<0.05'),
                              Patch(fc=self.DIFF_NS,  label='n.s.')], fontsize=9)

        ax_h = fig.add_subplot(gs[1, 2:4])
        ax_h.plot(L, coh_h, 'o-', lw=2, ms=5, color='#5C4B8A')
        for h, lbl, c in [(0.2,'Small','#e57373'),(0.5,'Medium','#ffa726'),(0.8,'Large','#66bb6a')]:
            ax_h.axhline(h, color=c, ls='--', lw=1.2, alpha=0.7, label=f'{lbl} h={h}')
        ax_h.axhline(0, color='black', lw=1.0)
        ax_h.set(xlabel='Layer', ylabel="Cohen's h", title="Effect Size (Cohen's h)")
        ax_h.legend(fontsize=8); ax_h.grid(True, alpha=0.25)

        ax_p = fig.add_subplot(gs[1, 4]); ax_p.axis('off')
        ax_p.text(0.5, 0.5, 'Probe: memory-only\n(no .pt written)',
                  ha='center', va='center', transform=ax_p.transAxes,
                  fontsize=11, color='gray', style='italic')

        fig.suptitle(f'Full Metric Suite — {model_name}',
                     fontsize=16, fontweight='bold', y=1.01)
        plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                 f'{safe}_all_metrics_grid.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all'); gc.collect()
        logger.info(f"✓ {safe}_all_metrics_grid.png")

    # ════════════════════════════════════════════════════════════════════════
    # EXTENDED ANALYSIS PLOTS
    # ════════════════════════════════════════════════════════════════════════

    def _plot_extended(self, model_name, res):
        safe = model_name.replace(' ', '_').replace('/', '_')

        # ── Analysis 1: Direct Frequency Probe ────────────────────────
        if res.get('direct_freq') and res['direct_freq']['binary']:
            df_bin = pd.DataFrame(res['direct_freq']['binary'])
            if not df_bin.empty and not df_bin['accuracy'].isna().all():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
                ax1.plot(df_bin['layer'], df_bin['accuracy'], 'o-', lw=2.5,
                         ms=7, color='#E63946', label='Binary (H vs L)')
                ax1.axhline(0.5, color='gray', ls='--', lw=1.2, label='Chance')
                ax1.set(xlabel='Layer', ylabel='Accuracy',
                        title=f'Direct Frequency Probe (Binary) — {model_name}')
                ax1.legend(); ax1.grid(True, alpha=0.3)

                if 'f1' in df_bin.columns:
                    ax2.plot(df_bin['layer'], df_bin['f1'], 's-', lw=2.5,
                             ms=7, color='#457B9D', label='F1')
                if 'auc' in df_bin.columns:
                    ax2.plot(df_bin['layer'], df_bin['auc'], '^-', lw=2.5,
                             ms=7, color='#2A9D8F', label='AUC')
                ax2.axhline(0.5, color='gray', ls='--', lw=1.2)
                ax2.set(xlabel='Layer', ylabel='Score',
                        title=f'Direct Freq Probe — F1 & AUC — {model_name}')
                ax2.legend(); ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                         f'{safe}_direct_freq_probe.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()

            # 3-way
            df_3w = pd.DataFrame(res['direct_freq'].get('three_way', []))
            if not df_3w.empty and not df_3w['accuracy'].isna().all():
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(df_3w['layer'], df_3w['accuracy'], 'D-', lw=2.5,
                        ms=7, color='#6A0572', label='3-way (H/M/L)')
                ax.axhline(1.0/3, color='gray', ls='--', lw=1.2, label='Chance (0.33)')
                ax.set(xlabel='Layer', ylabel='Accuracy',
                       title=f'3-Way Frequency Probe — {model_name}')
                ax.legend(); ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                         f'{safe}_3way_freq_probe.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()

        # ── Analysis 5: Pooling Ablation Comparison ───────────────────
        if res.get('mean_pool_results') and res.get('layer_results'):
            lr_lt = res['layer_results']
            lr_mp = res['mean_pool_results']
            L = self._L(lr_lt)

            acc_lt = self._series(lr_lt, 'overall', 'accuracy')
            acc_mp = self._series(lr_mp, 'overall', 'accuracy')
            diff_lt = [r['frequency_effect']['accuracy_difference'] for r in lr_lt]
            diff_mp = [r['frequency_effect']['accuracy_difference'] for r in lr_mp]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
            ax1.plot(L, acc_lt, 'o-', lw=2.5, color='#2E86AB', label='Last-Token')
            ax1.plot(L, acc_mp, 's--', lw=2.5, color='#D62246', label='Mean-Pool')
            ax1.axhline(0.5, color='gray', ls=':', lw=1.2)
            ax1.set(xlabel='Layer', ylabel='Accuracy',
                    title=f'Pooling Ablation: LDT Accuracy — {model_name}')
            ax1.legend(); ax1.grid(True, alpha=0.3)

            ax2.plot(L, diff_lt, 'o-', lw=2.5, color='#2E86AB', label='Last-Token')
            ax2.plot(L, diff_mp, 's--', lw=2.5, color='#D62246', label='Mean-Pool')
            ax2.axhline(0, color='black', lw=1.2)
            ax2.set(xlabel='Layer', ylabel='Acc Diff (High − Low)',
                    title=f'Pooling Ablation: Freq Effect — {model_name}')
            ax2.legend(); ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                     f'{safe}_pooling_ablation.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all'); gc.collect()

        # ── Analysis 4: Stability (error bars) ────────────────────────
        if res.get('stability') and res['stability']['layer_stats']:
            df_s = pd.DataFrame(res['stability']['layer_stats'])
            if not df_s.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

                ax1.errorbar(df_s['layer'], df_s['acc_mean'], yerr=df_s['acc_std'],
                            fmt='o-', lw=2, ms=6, capsize=4, color='#1A535C',
                            label=f'Mean ± SD ({res["stability"]["n_seeds"]} seeds)')
                ax1.axhline(0.5, color='gray', ls='--', lw=1.2)
                ax1.set(xlabel='Layer', ylabel='Accuracy',
                        title=f'Multi-Seed Stability: Accuracy — {model_name}')
                ax1.legend(); ax1.grid(True, alpha=0.3)

                ax2.errorbar(df_s['layer'], df_s['freq_diff_mean'],
                            yerr=df_s['freq_diff_std'],
                            fmt='s-', lw=2, ms=6, capsize=4, color='#D62246',
                            label=f'Mean ± SD ({res["stability"]["n_seeds"]} seeds)')
                ax2.axhline(0, color='black', lw=1.2)
                ax2.set(xlabel='Layer', ylabel='Freq Effect (Acc Diff)',
                        title=f'Multi-Seed Stability: Freq Effect — {model_name}')
                ax2.legend(); ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                         f'{safe}_stability.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()

        # ── Analysis 6: Selectivity ───────────────────────────────────
        if res.get('selectivity'):
            sel = res['selectivity']
            if sel.get('selectivity') and sel['selectivity']:
                df_sel = pd.DataFrame(sel['selectivity'])
                if not df_sel.empty:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

                    ax1.plot(df_sel['layer'], df_sel['acc_task'], 'o-',
                             lw=2.5, color='#2E86AB', label='Task (real labels)')
                    ax1.plot(df_sel['layer'], df_sel['acc_shuffled'], 's--',
                             lw=2.5, color='#AAAAAA', label='Shuffled labels')
                    ax1.axhline(0.5, color='gray', ls=':', lw=1.2)
                    ax1.set(xlabel='Layer', ylabel='Accuracy',
                            title=f'Probe Selectivity — {model_name}')
                    ax1.legend(); ax1.grid(True, alpha=0.3)

                    ax2.bar(df_sel['layer'], df_sel['selectivity'],
                            color='#2DC653', alpha=0.8, edgecolor='black', lw=0.5)
                    ax2.axhline(0, color='black', lw=1.2)
                    ax2.set(xlabel='Layer', ylabel='Selectivity (Task − Shuffled)',
                            title=f'Selectivity Score — {model_name}')
                    ax2.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                             f'{safe}_selectivity.png'),
                                bbox_inches='tight', dpi=300)
                    plt.close('all'); gc.collect()

            # Linear vs MLP comparison
            if sel.get('linear_probe') and sel['linear_probe']:
                df_lp = pd.DataFrame(sel['linear_probe'])
                lr_orig = res['layer_results']
                # Get MLP accuracy at same layers
                mlp_accs = {}
                for r in lr_orig:
                    if r.get('overall'):
                        mlp_accs[r['layer']] = r['overall']['accuracy']

                if not df_lp.empty:
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(df_lp['layer'], df_lp['accuracy'], 's--',
                            lw=2.5, color='#E63946', label='Linear Probe')
                    mlp_layers = sorted(mlp_accs.keys())
                    mlp_vals = [mlp_accs[l] for l in mlp_layers]
                    ax.plot(mlp_layers, mlp_vals, 'o-',
                            lw=2.5, color='#1A535C', label='MLP Probe')
                    ax.axhline(0.5, color='gray', ls=':', lw=1.2)
                    ax.set(xlabel='Layer', ylabel='Accuracy',
                           title=f'Linear vs MLP Probe — {model_name}')
                    ax.legend(); ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                             f'{safe}_linear_vs_mlp.png'),
                                bbox_inches='tight', dpi=300)
                    plt.close('all'); gc.collect()

        # ── Analysis 7: Continuous frequency regression ───────────────
        if res.get('regression') and res['regression'].get('regression_results'):
            df_reg = pd.DataFrame(res['regression']['regression_results'])
            if not df_reg.empty and not df_reg['r2'].isna().all():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

                ax1.plot(df_reg['layer'], df_reg['r2'], 'o-', lw=2.5,
                         ms=7, color='#6A0572', label='R²')
                ax1.axhline(0, color='gray', ls='--', lw=1.2, label='Baseline (mean)')
                ax1.set(xlabel='Layer', ylabel='R²',
                        title=f'Continuous Freq Regression: R² — {model_name}')
                ax1.legend(); ax1.grid(True, alpha=0.3)

                ax2.plot(df_reg['layer'], df_reg['spearman_r'], 's-', lw=2.5,
                         ms=7, color='#2A9D8F', label='Spearman ρ')
                ax2.axhline(0, color='gray', ls='--', lw=1.2)
                ax2.set(xlabel='Layer', ylabel='Spearman ρ',
                        title=f'Continuous Freq Regression: Correlation — {model_name}')
                ax2.legend(); ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                         f'{safe}_freq_regression.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()

        # ── Analysis 3: Confound-matched probe ────────────────────────
        if (res.get('confound_matched') and
                res['confound_matched'].get('matched_probe_results')):
            df_cm = pd.DataFrame(res['confound_matched']['matched_probe_results'])
            if not df_cm.empty and not df_cm['accuracy'].isna().all():
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(df_cm['layer'], df_cm['accuracy'], 'D-', lw=2.5,
                        ms=7, color='#E76F51',
                        label=f'Confound-Matched (n={res["confound_matched"]["n_matched_pairs"]} pairs)')
                # Overlay original direct freq probe for comparison
                if res.get('direct_freq') and res['direct_freq']['binary']:
                    df_orig = pd.DataFrame(res['direct_freq']['binary'])
                    if not df_orig.empty:
                        ax.plot(df_orig['layer'], df_orig['accuracy'], 'o--',
                                lw=2, ms=5, color='#264653', alpha=0.6,
                                label='Unmatched')
                ax.axhline(0.5, color='gray', ls=':', lw=1.2, label='Chance')
                ax.set(xlabel='Layer', ylabel='Accuracy',
                       title=f'Confound-Matched vs Unmatched Freq Probe — {model_name}')
                ax.legend(); ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                         f'{safe}_confound_matched.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()

    # ════════════════════════════════════════════════════════════════════════
    # EXTENDED ANALYSIS CSVs
    # ════════════════════════════════════════════════════════════════════════

    def _save_extended_csvs(self, model_name, res):
        safe = model_name.replace(' ', '_').replace('/', '_')
        ext_dir = os.path.join(self.config.RESULTS_DIR, f'{safe}_extended')
        os.makedirs(ext_dir, exist_ok=True)

        # Analysis 1: Direct frequency probe
        if res.get('direct_freq'):
            if res['direct_freq']['binary']:
                pd.DataFrame(res['direct_freq']['binary']).to_csv(
                    os.path.join(ext_dir, 'direct_freq_binary.csv'), index=False)
            if res['direct_freq'].get('three_way'):
                pd.DataFrame(res['direct_freq']['three_way']).to_csv(
                    os.path.join(ext_dir, 'direct_freq_3way.csv'), index=False)

        # Analysis 2: Tokenization control
        if res.get('token_ctrl'):
            tc = res['token_ctrl']
            with open(os.path.join(ext_dir, 'token_stats.json'), 'w') as f:
                json.dump(tc['token_stats'], f, indent=2)
            if tc.get('single_token_results'):
                rows = []
                for r in tc['single_token_results']:
                    fe = r['frequency_effect']
                    ov = r.get('overall')
                    rows.append({
                        'layer': r['layer'],
                        'accuracy': ov['accuracy'] if ov else np.nan,
                        'f1': ov['f1'] if ov else np.nan,
                        'freq_diff': fe['accuracy_difference'],
                    })
                pd.DataFrame(rows).to_csv(
                    os.path.join(ext_dir, 'single_token_ldt.csv'), index=False)
            if tc.get('token_matched_results'):
                rows = []
                for r in tc['token_matched_results']:
                    fe = r['frequency_effect']
                    ov = r.get('overall')
                    rows.append({
                        'layer': r['layer'],
                        'accuracy': ov['accuracy'] if ov else np.nan,
                        'f1': ov['f1'] if ov else np.nan,
                        'freq_diff': fe['accuracy_difference'],
                    })
                pd.DataFrame(rows).to_csv(
                    os.path.join(ext_dir, 'token_matched_ldt.csv'), index=False)

        # Analysis 3: Confound matching
        if res.get('confound_matched'):
            cm = res['confound_matched']
            balance = cm['balance_before'] + cm['balance_after']
            pd.DataFrame(balance).to_csv(
                os.path.join(ext_dir, 'balance_table.csv'), index=False)
            if cm.get('matched_probe_results'):
                pd.DataFrame(cm['matched_probe_results']).to_csv(
                    os.path.join(ext_dir, 'confound_matched_probe.csv'), index=False)

        # Analysis 4: Stability
        if res.get('stability') and res['stability']['layer_stats']:
            pd.DataFrame(res['stability']['layer_stats']).to_csv(
                os.path.join(ext_dir, 'stability_stats.csv'), index=False)

        # Analysis 5: Pooling ablation
        if res.get('mean_pool_results'):
            rows = []
            for r in res['mean_pool_results']:
                ov = r.get('overall')
                fe = r['frequency_effect']
                rows.append({
                    'layer': r['layer'], 'pooling': 'mean_pool',
                    'accuracy': ov['accuracy'] if ov else np.nan,
                    'f1': ov['f1'] if ov else np.nan,
                    'freq_diff': fe['accuracy_difference'],
                    'cohens_h': fe.get('cohens_h', np.nan),
                })
            # Also add last_token for comparison
            for r in res['layer_results']:
                ov = r.get('overall')
                fe = r['frequency_effect']
                rows.append({
                    'layer': r['layer'], 'pooling': 'last_token',
                    'accuracy': ov['accuracy'] if ov else np.nan,
                    'f1': ov['f1'] if ov else np.nan,
                    'freq_diff': fe['accuracy_difference'],
                    'cohens_h': fe.get('cohens_h', np.nan),
                })
            pd.DataFrame(rows).to_csv(
                os.path.join(ext_dir, 'pooling_ablation.csv'), index=False)

        # Analysis 6: Selectivity
        if res.get('selectivity'):
            for key in ['linear_probe', 'shuffled_label', 'selectivity']:
                data = res['selectivity'].get(key, [])
                if data:
                    pd.DataFrame(data).to_csv(
                        os.path.join(ext_dir, f'selectivity_{key}.csv'), index=False)

        # Analysis 7: Regression
        if res.get('regression') and res['regression'].get('regression_results'):
            pd.DataFrame(res['regression']['regression_results']).to_csv(
                os.path.join(ext_dir, 'freq_regression.csv'), index=False)

        logger.info(f"✓ Extended CSVs saved to {ext_dir}")

    # ════════════════════════════════════════════════════════════════════════
    # PER-MODEL CSV (original)
    # ════════════════════════════════════════════════════════════════════════

    def _save_model_csv(self, model_name, lr):
        safe = model_name.replace(' ','_').replace('/','_')
        rows = []
        for r in lr:
            fe = r['frequency_effect']
            ov = r.get('overall'); hf = r.get('high_frequency'); lf = r.get('low_frequency')
            def mg(g,m): return g[m] if g else np.nan
            rows.append({
                'model': model_name, 'layer': r['layer'],
                'test_accuracy': mg(ov,'accuracy'), 'test_precision': mg(ov,'precision'),
                'test_recall':   mg(ov,'recall'),   'test_f1':        mg(ov,'f1'),
                'test_auc':      mg(ov,'auc'),       'test_n':         mg(ov,'n_samples'),
                'hf_accuracy': mg(hf,'accuracy'), 'hf_precision': mg(hf,'precision'),
                'hf_recall':   mg(hf,'recall'),   'hf_f1':        mg(hf,'f1'),
                'hf_auc':      mg(hf,'auc'),       'hf_n':         mg(hf,'n_samples'),
                'lf_accuracy': mg(lf,'accuracy'), 'lf_precision': mg(lf,'precision'),
                'lf_recall':   mg(lf,'recall'),   'lf_f1':        mg(lf,'f1'),
                'lf_auc':      mg(lf,'auc'),       'lf_n':         mg(lf,'n_samples'),
                'accuracy_difference': fe['accuracy_difference'],
                'cohens_h':            fe.get('cohens_h', np.nan),
                'z_statistic':         fe['z_statistic'],
                'p_value':             fe['p_value'],
                'p_value_corrected':   fe.get('p_value_corrected', np.nan),
                'significant_fdr':     fe.get('significant_fdr', False),
                'n_high': fe['n_high'], 'n_low': fe['n_low'],
            })
        out = os.path.join(self.config.RESULTS_DIR, f'{safe}_full_metrics.csv')
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info(f"✓ CSV: {os.path.basename(out)}")

    # ════════════════════════════════════════════════════════════════════════
    # CROSS-MODEL PLOTS
    # ════════════════════════════════════════════════════════════════════════

    def _plot_cross_model(self, all_results):
        rows = []
        for mn, md in all_results.items():
            for r in md['layer_results']:
                fe = r['frequency_effect']
                ov = r.get('overall'); hf = r.get('high_frequency'); lf = r.get('low_frequency')
                def mg(g,m): return g[m] if g else np.nan
                n_layers = md['num_layers']
                rows.append({
                    'model': mn, 'layer': r['layer'],
                    'layer_norm': r['layer'] / max(n_layers - 1, 1),
                    'test_acc':  mg(ov,'accuracy'), 'test_pr': mg(ov,'precision'),
                    'test_rc':   mg(ov,'recall'),   'test_f1': mg(ov,'f1'),
                    'test_auc':  mg(ov,'auc'),
                    'hf_acc':    mg(hf,'accuracy'), 'hf_f1':   mg(hf,'f1'), 'hf_auc': mg(hf,'auc'),
                    'lf_acc':    mg(lf,'accuracy'), 'lf_f1':   mg(lf,'f1'), 'lf_auc': mg(lf,'auc'),
                    'acc_diff':  fe['accuracy_difference'],
                    'cohens_h':  fe.get('cohens_h', np.nan),
                    'p_corr':    fe.get('p_value_corrected', np.nan),
                })
        df      = pd.DataFrame(rows)
        models  = df['model'].unique()
        palette = sns.color_palette("husl", len(models))

        def _line(col, ylabel, title, fname, hline=None, refs=None, use_norm=False):
            fig, ax = plt.subplots(figsize=(16, 7))
            x_col = 'layer_norm' if use_norm else 'layer'
            x_label = 'Normalised Depth (0=first, 1=last)' if use_norm else 'Layer'
            for m, c in zip(models, palette):
                sub = df[df['model']==m]
                ax.plot(sub[x_col], sub[col], 'o-', label=m, lw=2.8, ms=8, color=c)
            if hline is not None:
                ax.axhline(hline, color='red', ls='--', lw=1.8, label='Chance/Zero')
            if refs:
                for h, lbl, c in refs:
                    ax.axhline(h, color=c, ls='--', lw=1.2, alpha=0.6, label=lbl)
            ax.set(xlabel=x_label, ylabel=ylabel, title=title)
            ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.COMPARISON_DIR, fname),
                        bbox_inches='tight', dpi=300)
            plt.close('all'); gc.collect()
            logger.info(f"✓ {fname}")

        # Raw layer plots
        _line('test_acc',  'Accuracy',           'Cross-Model: Overall Test Accuracy',
              'cross_model_test_accuracy.png',   hline=0.5)
        _line('hf_acc',    'Accuracy',           'Cross-Model: High-Freq Accuracy',
              'cross_model_high_freq_accuracy.png', hline=0.5)
        _line('lf_acc',    'Accuracy',           'Cross-Model: Low-Freq Accuracy',
              'cross_model_low_freq_accuracy.png',  hline=0.5)
        _line('test_f1',   'F1-Score',           'Cross-Model: Overall Test F1',
              'cross_model_test_f1.png')
        _line('test_auc',  'AUC-ROC',            'Cross-Model: Overall Test AUC',
              'cross_model_test_auc.png',        hline=0.5)
        _line('acc_diff',  'Acc Diff (High−Low)','Cross-Model: Frequency Effect',
              'cross_model_frequency_effect.png', hline=0.0)
        _line('cohens_h',  "Cohen's h",          "Cross-Model: Effect Size",
              'cross_model_effect_size.png',
              refs=[(0.2,'Small','#e57373'),(0.5,'Medium','#ffa726'),(0.8,'Large','#66bb6a')])

        # Normalised depth plots
        _line('test_acc',  'Accuracy',
              'Cross-Model: Test Accuracy (Normalised Depth)',
              'cross_model_test_accuracy_normdepth.png', hline=0.5, use_norm=True)
        _line('acc_diff',  'Acc Diff (High−Low)',
              'Cross-Model: Frequency Effect (Normalised Depth)',
              'cross_model_freq_effect_normdepth.png', hline=0.0, use_norm=True)

        # heatmaps
        for col, label, cmap, center, fname in [
            ('test_acc', 'Overall Test Accuracy',   'YlOrRd', None,  'cross_model_heatmap_test_acc.png'),
            ('acc_diff', 'Frequency Effect',        'RdBu_r', 0.0,   'cross_model_heatmap_freq_effect.png'),
            ('test_f1',  'Test F1-Score',           'YlOrRd', None,  'cross_model_heatmap_f1.png'),
            ('hf_acc',   'High-Freq Accuracy',      'YlOrRd', None,  'cross_model_heatmap_hf_acc.png'),
            ('lf_acc',   'Low-Freq Accuracy',       'YlOrRd', None,  'cross_model_heatmap_lf_acc.png'),
        ]:
            try:
                pivot = df.pivot(index='layer', columns='model', values=col)
                fig, ax = plt.subplots(figsize=(max(10, len(models)*4), 10))
                sns.heatmap(pivot.T, annot=True, fmt='.3f', cmap=cmap, center=center,
                            linewidths=0.4, ax=ax, cbar_kws={'label': label})
                ax.set(xlabel='Layer', ylabel='Model', title=f'Heatmap: {label}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.COMPARISON_DIR, fname),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()
                logger.info(f"✓ {fname}")
            except Exception as e:
                logger.warning(f"Heatmap {fname} skipped: {e}")

        # 3×3 all-metrics grid
        panels = [
            ('test_acc', 'Overall Accuracy',   0.5),
            ('hf_acc',   'High-Freq Accuracy', 0.5),
            ('lf_acc',   'Low-Freq Accuracy',  0.5),
            ('test_pr',  'Overall Precision',  None),
            ('test_rc',  'Overall Recall',     None),
            ('test_f1',  'Overall F1-Score',   None),
            ('test_auc', 'Overall AUC-ROC',    0.5),
            ('acc_diff', 'Freq Effect',         0.0),
            ('cohens_h', "Cohen's h",           0.0),
        ]
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        for ax, (col, ylabel, hl) in zip(axes.flat, panels):
            for m, c in zip(models, palette):
                sub = df[df['model']==m]
                ax.plot(sub['layer'], sub[col], 'o-', label=m, lw=2, ms=5, color=c)
            if hl is not None:
                ax.axhline(hl, color='gray', ls='--', lw=1.2)
            ax.set(xlabel='Layer', ylabel=ylabel, title=ylabel)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        fig.suptitle('Cross-Model Comparison — All Metrics', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.COMPARISON_DIR,
                                 'cross_model_all_metrics_grid.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all'); gc.collect()
        logger.info("✓ cross_model_all_metrics_grid.png")

        # ── Cross-model extended comparison plots ─────────────────────
        self._plot_cross_model_extended(all_results, models, palette)

    def _plot_cross_model_extended(self, all_results, models, palette):
        """Cross-model comparison plots for extended analyses."""

        # ── Cross-model: Direct Frequency Probe ──────────────────────
        has_direct = any(
            r.get('direct_freq', {}).get('binary')
            for r in all_results.values()
        )
        if has_direct:
            fig, ax = plt.subplots(figsize=(16, 7))
            for m, c in zip(models, palette):
                if m not in all_results:
                    continue
                df_bin = all_results[m].get('direct_freq', {}).get('binary', [])
                if df_bin:
                    df_b = pd.DataFrame(df_bin)
                    nl = all_results[m]['num_layers']
                    df_b['layer_norm'] = df_b['layer'] / max(nl - 1, 1)
                    ax.plot(df_b['layer_norm'], df_b['accuracy'], 'o-',
                            label=m, lw=2.5, ms=7, color=c)
            ax.axhline(0.5, color='red', ls='--', lw=1.5, label='Chance')
            ax.set(xlabel='Normalised Depth', ylabel='Accuracy',
                   title='Cross-Model: Direct Frequency Probe (Binary)')
            ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.COMPARISON_DIR,
                                     'cross_model_direct_freq_probe.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all'); gc.collect()

        # ── Cross-model: Regression R² ────────────────────────────────
        has_reg = any(
            r.get('regression', {}).get('regression_results')
            for r in all_results.values()
        )
        if has_reg:
            fig, ax = plt.subplots(figsize=(16, 7))
            for m, c in zip(models, palette):
                if m not in all_results:
                    continue
                reg = all_results[m].get('regression', {}).get('regression_results', [])
                if reg:
                    df_r = pd.DataFrame(reg)
                    nl = all_results[m]['num_layers']
                    df_r['layer_norm'] = df_r['layer'] / max(nl - 1, 1)
                    ax.plot(df_r['layer_norm'], df_r['r2'], 'o-',
                            label=m, lw=2.5, ms=7, color=c)
            ax.axhline(0, color='gray', ls='--', lw=1.2)
            ax.set(xlabel='Normalised Depth', ylabel='R²',
                   title='Cross-Model: Continuous Frequency Regression R²')
            ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.COMPARISON_DIR,
                                     'cross_model_freq_regression_r2.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all'); gc.collect()

        # ── Cross-model: Pooling ablation comparison ──────────────────
        has_pool = any(
            r.get('mean_pool_results') for r in all_results.values()
        )
        if has_pool:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
            for m, c in zip(models, palette):
                if m not in all_results:
                    continue
                md = all_results[m]
                nl = md['num_layers']
                if md.get('layer_results') and md.get('mean_pool_results'):
                    lt_acc = [r.get('overall', {}).get('accuracy', np.nan)
                              if r.get('overall') else np.nan
                              for r in md['layer_results']]
                    mp_acc = [r.get('overall', {}).get('accuracy', np.nan)
                              if r.get('overall') else np.nan
                              for r in md['mean_pool_results']]
                    layers_norm = [r['layer'] / max(nl - 1, 1)
                                   for r in md['layer_results']]

                    # Accuracy difference: mean_pool - last_token
                    diff = [mp - lt for mp, lt in zip(mp_acc, lt_acc)]
                    ax1.plot(layers_norm, diff, 'o-', label=m, lw=2, ms=5, color=c)

                    # Freq effect difference
                    lt_fe = [r['frequency_effect']['accuracy_difference']
                             for r in md['layer_results']]
                    mp_fe = [r['frequency_effect']['accuracy_difference']
                             for r in md['mean_pool_results']]
                    fe_diff = [mp - lt for mp, lt in zip(mp_fe, lt_fe)]
                    ax2.plot(layers_norm, fe_diff, 'o-', label=m, lw=2, ms=5, color=c)

            ax1.axhline(0, color='black', lw=1.2)
            ax1.set(xlabel='Normalised Depth',
                    ylabel='Accuracy Diff (Mean−Pool − Last−Token)',
                    title='Cross-Model: Pooling Strategy Impact on Accuracy')
            ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

            ax2.axhline(0, color='black', lw=1.2)
            ax2.set(xlabel='Normalised Depth',
                    ylabel='Freq Effect Diff (Mean−Pool − Last−Token)',
                    title='Cross-Model: Pooling Strategy Impact on Freq Effect')
            ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.COMPARISON_DIR,
                                     'cross_model_pooling_ablation.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all'); gc.collect()

    # ════════════════════════════════════════════════════════════════════════
    # CROSS-MODEL CSV + METADATA
    # ════════════════════════════════════════════════════════════════════════

    def _save_cross_model_csv(self, all_results):
        detailed, summary = [], []
        for mn, md in all_results.items():
            lr = md['layer_results']
            for r in lr:
                fe = r['frequency_effect']
                ov = r.get('overall'); hf = r.get('high_frequency'); lf = r.get('low_frequency')
                def mg(g,m): return g[m] if g else np.nan
                detailed.append({
                    'model': mn, 'architecture': md['model_config'].architecture_type.value,
                    'input_mode': md['model_config'].input_mode.value, 'layer': r['layer'],
                    'test_accuracy': mg(ov,'accuracy'), 'test_precision': mg(ov,'precision'),
                    'test_recall':   mg(ov,'recall'),   'test_f1':        mg(ov,'f1'),
                    'test_auc':      mg(ov,'auc'),
                    'hf_accuracy': mg(hf,'accuracy'), 'hf_f1': mg(hf,'f1'), 'hf_auc': mg(hf,'auc'),
                    'lf_accuracy': mg(lf,'accuracy'), 'lf_f1': mg(lf,'f1'), 'lf_auc': mg(lf,'auc'),
                    'accuracy_difference': fe['accuracy_difference'],
                    'cohens_h':            fe.get('cohens_h', np.nan),
                    'z_statistic':         fe['z_statistic'],
                    'p_value':             fe['p_value'],
                    'p_value_corrected':   fe.get('p_value_corrected', np.nan),
                    'significant_fdr':     fe.get('significant_fdr', False),
                })

            def _nm(lst): return float(np.nanmean(lst)) if lst else np.nan
            def _nx(lst): return float(np.nanmax(lst))  if lst else np.nan
            accs  = [r['overall']['accuracy'] for r in lr if r.get('overall')]
            f1s   = [r['overall']['f1']       for r in lr if r.get('overall')]
            aucs  = [r['overall']['auc']       for r in lr if r.get('overall')]
            diffs = [r['frequency_effect']['accuracy_difference'] for r in lr
                     if not np.isnan(r['frequency_effect']['accuracy_difference'])]
            hs    = [r['frequency_effect']['cohens_h'] for r in lr
                     if not np.isnan(r['frequency_effect'].get('cohens_h', np.nan))]
            pcs   = [r['frequency_effect'].get('p_value_corrected', np.nan) for r in lr]
            summary.append({
                'model':               mn,
                'architecture':        md['model_config'].architecture_type.value,
                'input_mode':          md['model_config'].input_mode.value,
                'num_layers':          md['num_layers'],
                'mean_test_accuracy':  _nm(accs),  'max_test_accuracy': _nx(accs),
                'mean_test_f1':        _nm(f1s),   'mean_test_auc':     _nm(aucs),
                'mean_freq_effect':    _nm(diffs),  'max_freq_effect':   _nx(diffs),
                'mean_cohens_h':       _nm(hs),     'max_cohens_h':      _nx(hs),
                'n_significant_layers':sum(1 for p in pcs
                                           if not np.isnan(p) and p < 0.05),
            })

        pd.DataFrame(detailed).to_csv(
            os.path.join(self.config.COMPARISON_DIR, 'cross_model_detailed_results.csv'),
            index=False)
        pd.DataFrame(summary).to_csv(
            os.path.join(self.config.COMPARISON_DIR, 'cross_model_statistics.csv'),
            index=False)

        meta = {
            'version': 'Extended analyses for defensible frequency claims',
            'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': [{'name': m.name, 'model_id': m.model_id,
                        'architecture': m.architecture_type.value,
                        'input_mode': m.input_mode.value,
                        'padding_side': m.padding_side.value}
                       for m in self.config.MODELS],
            'extended_analyses': [
                '1. Direct frequency probe (binary H-vs-L + 3-way H/M/L)',
                '2. Tokenization-controlled rerun (single-token + token-matched)',
                '3. Lexical-confound-matched rerun (length + Ortho_N + token-count matching)',
                '4. Multi-seed stability (N seeds with mean ± std)',
                '5. Pooling ablation (last-token vs mean-pool)',
                '6. Probe selectivity controls (linear baseline + shuffled labels)',
                '7. Continuous frequency regression (Ridge on log_HAL)',
            ],
            'references': [
                'He et al. (2015): Kaiming init for ReLU',
                'He et al. (2016): Pre-activation residual blocks',
                'Lin et al. (2017): Focal Loss',
                'Ioffe & Szegedy (2015): Batch Normalisation',
                'Agresti (2002): Two-proportion z-test',
                'Muennighoff et al. (2022): SGPT — last-token pooling',
                'Hewitt & Liang (2019): Designing and interpreting probes',
                'Yang et al. (2024): Qwen2 Technical Report',
                'Falcon-LLM Team (2024): Falcon 3 Family of Open Models',
                'Meta AI (2024): Llama 3.2 — Llama 3.2-3B base model',
            ],
        }
        with open(os.path.join(self.config.COMPARISON_DIR, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info("✓ Saved all cross-model CSVs and metadata.json")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    config = Config()
    print(f"\n{'='*64}\nVERIFYING DATA FILES\n{'='*64}")
    for path, name in [(config.WORDS_PATH,'Words'),(config.NONWORDS_PATH,'NonWords')]:
        if os.path.exists(path):
            print(f"  ✓  {name}  →  {path}")
        else:
            print(f"  ✗  MISSING: {name}  →  {path}"); return None
    print()
    return MultiModelExperiment(config).run()


if __name__ == "__main__":
    main()
