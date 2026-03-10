# Generalized Functional Approximation (GFA) for Transformer Attention

A PyTorch-based research codebase implementing **Generalized Functional Approximation (GFA)** variants for transformer attention mechanisms, with applications to language modeling (LM) and vision tasks (CIFAR-10, MNIST).

## Overview

This project explores efficient attention mechanisms through functional approximation techniques. The core contribution involves implementing several GFA variants that can serve as drop-in replacements for standard multi-head attention in transformer architectures, with both original and FFT-based implementations.

### Key Features

- **Multiple GFA Variants**: Original, FFT-based, K-normalized, and combined implementations
- **Flexible Architecture**: Supports language modeling and image classification tasks
- **Advanced Components**: RoPE positional encoding, RMSNorm, SwiGLU activation, and dynamic causal masking
- **Comprehensive Experiments**: Pre-configured notebooks for CIFAR-10, MNIST, and WikiText-2
- **Production-Ready Training Utilities**: Integrated ModelTrainer with cross-validation support

## Architecture

### Core Attention Mechanisms (`models.py`)

#### Functional Projections
- **`gen_func_proj_ori()`**: Original functional attention implementation using tensor operations
- **`gen_func_proj_fft()`**: FFT-based fast attention variant
- **`gen_func_proj_ori_k()`**: K-normalized original variant
- **`gen_func_proj_fft_k()`**: K-normalized FFT variant

Each function computes a generalized functional approximation of the attention mechanism by:
1. Computing outer products between queries and keys
2. Aggregating results into frequency bins
3. Applying learnable scaling factors

#### Attention Layer Classes
All attention layers inherit from `GenFuncProA` base class:
- **`GFAPO`**: Uses original projection
- **`GFAPF`**: Uses FFT projection
- **`GFAPOK`**: Uses K-normalized original
- **`GFAPFK`**: Uses K-normalized FFT

Standard attention processing workflow:
```
Input → Positional Encoding → Projection → Causal Mask → Output
```

### Transformer Blocks (`ori_model.py`, `lm_models.py`)

#### Base Components
- **`BaseAttentionBlock`**: Standard multi-head attention with optional RoPE
- **`BaseTransformerBlock`**: Residual connections + RMSNorm + attention + SwiGLU FFN
  - Architecture: `Attn → Residual → Norm → FFN → Residual → Norm`

#### Language Models
- **`BaseMiniLM`**: Baseline LM with conventional multi-head attention
- **`GFAPMiniLM`**: Base class for all GFA variants
  - Subclasses: `GFAPMiniLM_PO`, `GFAPMiniLM_PF`, `GFAPMiniLM_POK`, `GFAPMiniLM_PFK`
  - Uses factory method `build_block()` to swap attention implementations

### Utility Components (`uni_func.py`)

- **`RMSNorm`**: Root Mean Square normalization layer
  - Stable alternative to LayerNorm without learnable bias/shift parameters
- **`RotaryEmbedding (RoPE)`**: Rotary position embeddings
  - Cached sinusoidal position encodings with configurable maximum sequence length
- **`SwiGLU`**: Gated linear unit activation
  - FFN implementation: `Dense → SwiGLU → Projection`
- **`jaxtyping` Annotations**: Shape documentation for all tensor operations

## Project Structure

```
gfa1-2/
├── README.md                           # This file
├── models.py                           # GFA attention mechanisms (460 lines)
├── ori_model.py                        # Baseline transformer blocks
├── lm_models.py                        # Language model implementations
├── uni_func.py                         # Utility functions and components
├── ult.py                              # Training utilities and ModelTrainer class
├── bpe_3000.json                       # Pre-trained BPE tokenizer (vocab_size=3000)
│
├── exp1.ipynb                          # Classification experiments (CIFAR-10, MNIST)
├── exp_lm.ipynb                        # Language modeling experiments (WikiText-2)
├── res.ipynb                           # Results analysis and visualization
│
├── data/
│   ├── cifar-10-batches-py/           # CIFAR-10 dataset (5 train + 1 test batch)
│   │   ├── batches.meta
│   │   ├── data_batch_1-5
│   │   └── test_batch
│   └── MNIST/raw/                     # MNIST dataset (raw IDX format)
│       ├── train-images-idx3-ubyte
│       ├── train-labels-idx1-ubyte
│       ├── t10k-images-idx3-ubyte
│       └── t10k-labels-idx1-ubyte
│
└── exp_res/                            # Experiment results (CSV logs)
    ├── experiment_logs_so_ko_*.csv    # Results across all variants
    ├── experiment_logs_so_ko_lm_*.csv # LM-specific results
    └── experiment_logs_so_ko_cifr10_*.csv # CIFAR-10 results
```

## Conventions & Patterns

### Tensor Shape Documentation
All major functions use `jaxtyping` for shape annotations:
```python
def forward(self, x: Float[Array, "batch seq d_model"]) -> Float[Array, "batch seq d_model"]:
    ...
```

**Common Dimension Names**:
- `B` or `batch`: Batch size
- `n`: Sequence length or number of heads
- `m`: Dimension size or kernel size
- `d_k`, `d_v`: Head dimensions
- `d_model`: Model embedding dimension

### Tensor Operations
- **einops**: Used for reshaping via `rearrange()` and batch operations via `einsum()`
- **einx**: Dynamic batch dimension handling for masks
- **torch.fft**: FFT-based fast attention computations

### Device & Type Handling
- Models support `device` and `dtype` kwargs
- Always ensure masks, embeddings, and tensors are on the same device
- Use `.to(device)` explicitly when loading data

### Causal Masking Convention
- Causal masks are boolean tensors: `True` = attend, `False` = mask out
- Dynamically built to match batch dimensions using `einx.rearrange()`

## Datasets & Tokenization

### Supported Datasets
- **CIFAR-10**: 10-class image classification (32×32 RGB images)
  - Location: `data/cifar-10-batches-py/`
  - Training: 50,000 samples | Test: 10,000 samples
  
- **MNIST**: 10-class digit classification (28×28 grayscale images)
  - Location: `data/MNIST/raw/`
  - Raw IDX format (requires unpacking)
  
- **WikiText-2**: Language modeling dataset
  - Automatically downloaded via Hugging Face `datasets` library
  - Contains Wikipedia text for pretraining

### Tokenization
- **BPE Tokenizer**: Pre-trained with 3,000 vocabulary size
- File: `bpe_3000.json`
- Training procedure available in `exp_lm.ipynb` (cell 2)

## Experiment Tracking

Results are automatically logged to CSV files in `exp_res/`:

### Naming Convention
`experiment_logs_<variants>_<config>.csv`

- `so`: Standard Original attention
- `ko`: K-normalized Original
- Variants: baseline, gelu, nosqrt, rope, fft, etc.

### Logged Metrics
- `loss`: Model loss (cross-entropy for classification, language modeling loss for LM)
- `accuracy`: Task-specific accuracy
- `epoch`: Training epoch number
- `train/val`: Train vs validation splits

### Available Results
- Classification on CIFAR-10/MNIST with various configurations
- Language modeling on WikiText-2
- Ablations: activation functions (GELU), scaling factors (nosqrt), positional encoding (RoPE)

## Usage Guide

### Installation

Install required dependencies:
```bash
pip install torch einops jaxtyping einx tokenizers datasets scikit-learn pandas
```

### Running Classification Experiments

Open `exp1.ipynb`:

1. **Cell 1**: Import all models and utilities
2. **Cell 2-3**: Load and preprocess CIFAR-10/MNIST data
3. **Cell 4+**: Instantiate models and train

Example:
```python
from lm_models import GFAPMiniLM_PF
from ult import ModelTrainer

# Initialize model
model = GFAPMiniLM_PF(
    vocab_size=10,  # CIFAR-10 classes
    context_length=32,
    d_model=256,
    num_layers=4,
    num_heads=8,
    rope_theta=10000.0
)

# Train with cross-validation
trainer = ModelTrainer(model, device='cpu', dtype=torch.float32)
trainer.train_epoch(train_loader, optimizer)
metrics = trainer.eval_epoch(val_loader)
```

### Running Language Modeling Experiments

Open `exp_lm.ipynb`:

1. **Cell 1**: Import models and utilities
2. **Cell 2**: Initialize or load BPE tokenizer
3. **Cell 3**: Download and preprocess WikiText-2
4. **Cell 4+**: Configure and train LM models

Example:
```python
from lm_models import BaseMiniLM

# Create language model
lm = BaseMiniLM(
    vocab_size=3000,
    context_length=512,
    d_model=512,
    num_layers=6,
    num_heads=8,
    rope_theta=10000.0
)

# Training loop
for epoch in range(num_epochs):
    loss = trainer.train_lm_epoch(lm, train_loader, optimizer)
    val_loss = trainer.eval_lm_epoch(lm, val_loader)
```

### Adding New GFA Variants

1. **Define projection function** in `models.py`:
   ```python
   def gen_func_proj_new(q, k, v, mask):
       # Implement your custom functional approximation
       ...
   ```

2. **Create attention layer** in `models.py`:
   ```python
   class GFAPNEW(GenFuncProA):
       def forward(self, x, pe=None):
           # Apply positional encoding and call projection
           ...
   ```

3. **Create transformer block** in `models.py`:
   ```python
   GFAPNEWBLOCK = functools.partial(GenFuncTransformerBlock, attention_layer=GFAPNEW)
   ```

4. **Add LM model** in `lm_models.py`:
   ```python
   class GFAPMiniLM_NEW(GFAPMiniLM):
       def build_block(self):
           return GFAPNEWBLOCK(...)
   ```

5. **Test in notebook** with `ModelTrainer`

## Training Utilities (`ult.py`)

### ModelTrainer Class

The `ModelTrainer` class encapsulates training logic for both classification and LM tasks:

```python
class ModelTrainer:
    def __init__(self, model, device='cpu', dtype=torch.float32):
        ...
    
    def train_epoch(self, dataloader, optimizer):
        """Train classification model for one epoch"""
        
    def eval_epoch(self, dataloader):
        """Evaluate classification model"""
        
    def train_lm_epoch(self, model, dataloader, optimizer):
        """Train language model for one epoch"""
        
    def eval_lm_epoch(self, model, dataloader):
        """Evaluate language model loss"""
```

### Helper Functions

- **`count_parameters(model)`**: Returns total, trainable, and non-trainable parameters
- **`print_model_mib(model, name, include_buffers)`**: Prints model memory footprint in MiB

### Cross-Validation

Built-in KFold cross-validation support for quick benchmarking across dataset splits.

## Troubleshooting

### Shape Mismatch Errors
- Check `jaxtyping` annotations in function signatures
- Verify tensor dimensions match expected shapes
- Use `einops` to debug reshape operations

### Device Mismatch Errors
- Ensure all tensors, masks, and embeddings are on the same device
- Set device explicitly: `model.to(device)`
- Verify data loading moves tensors to correct device

### NaN Losses During Training
- Check `log_softmax` dimensions (should be `dim=-1`)
- Verify causal mask is applied correctly
- Reduce learning rate or add gradient clipping
- Check FFT implementation for numerical instability

### Out of Memory Errors
- Reduce `d_model`, `num_layers`, or batch size
- Switch to FFT variant (`GFAPF`, `GFAPFK`) for faster memory usage
- Use `torch.cuda.empty_cache()` between training runs

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Core neural network framework |
| `einops` | Efficient tensor reshaping and operations |
| `jaxtyping` | Type annotations for tensor shapes |
| `einx` | Advanced indexing for dynamic batch dimensions |
| `tokenizers` | Fast BPE tokenization |
| `datasets` | Hugging Face dataset loading |
| `scikit-learn` | KFold cross-validation utilities |
| `pandas` | Data logging and analysis |

## References

- **Core Attention**: `models.py` (lines 15-110 for projections, 240-300 for layer classes)
- **Baseline Components**: `ori_model.py` 
- **Language Models**: `lm_models.py` (lines 60-90 for GFAPMiniLM subclasses)
- **Training Loop**: `ult.py` (lines 56-120 for ModelTrainer)
- **Utilities**: `uni_func.py` (RMSNorm, RoPE, SwiGLU)



## License

This project is provided as-is for research purposes.


