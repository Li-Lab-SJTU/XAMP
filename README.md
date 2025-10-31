# XAMP: Dual-Engine Antimicrobial Peptide Prediction

A dual-engine deep learning framework for accurate and efficient antimicrobial peptide (AMP) prediction.

![Figure1](./Figure1_model_architecture.png)
**Figure 1. Integrated framework for AMP prediction and analysis.**
- **(A)** Data composition and motif characterization showing distribution in the 'Mix' dataset
- **(B)** Dual-engine architecture integrating XAMP-T (Transformer-based) and XAMP-E (ESM-2 based)
- **(C)** Data partitioning strategy with 10-fold cross-validation


## 📊 Model Performance

| Model | Precision | AUC | AUPR | Speed | Use Case |
|-------|-----------|-----|------|-------|----------|
| XAMP-T | 88.1% | 0.9668 | 0.8472 | Fast | Large-scale screening |
| XAMP-E | 91.8% |0.9706 | 0.8779 | Accurate | High-confidence prediction |
| **XAMP (Integrated)** | **96.1%** | **0.9715** | **0.8801** | Balanced | **Recommended** |

![Figure2](./Figure2_benchmark_results.png)

**Figure 2. Benchmarking results of XAMP.**
- **(A-C)** Superior predictive performance on testsets
- **(D)** Efficient model parameter counts
- **(E)** Fast inference speed across different dataset sizes

## 📦 Installation

```bash
# Using conda environment
conda env create -f XAMP.yaml
conda activate XAMP

# Install package in development mode to enable CLI
pip install -e .
```

## 🔧 Usage

### 1. Python API

```python
from XAMP import XAMP

# Initialize predictor
predictor = XAMP(
    model_type='integrated',  # 'integrated', 'xamp_t', or 'xamp_e'
    device='auto'  # 'auto', 'cpu', or 'cuda'
)

# Single sequence prediction
result = predictor.predict("LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES")

# Batch prediction
sequences = [
    "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "ACDEFGHIKLMNPQRSTVWY",
    "MKGLGLGLGLGLGLGLGLGL"
]
results = predictor.predict(
    sequences, 
    batch_size=64, 
    remove_n_met=True,  # Remove N-terminal methionine
    threshold=0.5       # Classification threshold
)

# Quick prediction function
from XAMP import predict_amp
result = predict_amp("LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES")
```

### 2. Command Line
```bash
usage: xamp-predict [-h] [--input INPUT] [--output OUTPUT] 
                    [--model_type {integrated,xamp_t,xamp_e}]
                    [--device {auto,cpu,cuda}] [--batch_size BATCH_SIZE]
                    [--threshold THRESHOLD] [--remove_n_met]

XAMP - Antimicrobial Peptide Prediction Tool

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Input file path (FASTA or text format).
  --output OUTPUT, -o OUTPUT
                        Output file path. If not provided, results will be printed to console.
  --model_type {integrated,xamp_t,xamp_e}
                        Model type to use for prediction (default: integrated)
  --device {auto,cpu,cuda}
                        Device to run inference on (default: auto)
  --batch_size BATCH_SIZE
                        Batch size for prediction (default: 64)
  --threshold THRESHOLD
                        Probability threshold for classification (default: 0.5)
  --remove_n_met        Remove N-terminal methionine from sequences

Examples:
  xamp-predict --input test.fasta --output test.xamp_pred.tsv --remove_n_met
  xamp-predict --input sequences.txt --device cuda --batch_size 128
```

## 📋 Output Format

Returns DataFrame with:
- `seq`: Input sequence
- `seq_processed`: Input sequence (After N-terminal trimming, if enabled)
- `prob_xamp_t`: XAMP-T probability
- `prob_xamp_e`: XAMP-E probability  
- `proba_final`: Final probability
- `prediction`: Binary classification (1=AMP, 0=non-AMP)


## 🧩 Features

- **Dual-engine architecture** (ESM-2 + Transformer)
- **Debiased training** addressing dataset biases
- **Batch processing** for large datasets
- **Flexible model selection** based on needs

## 📚 Citation

```bibtex
@article{chen2025xamp,
  title={A Global Discovery of Antimicrobial Peptides in Deep-Sea Microbiomes Driven by an ESM-2 and Transformer-based Dual-Engine Framework},
  author={Chen, Bairun and Mou, Xinyi and Song, Zhuoxuan and Lin, Huaying and Zhang, Yu and Li, Jing},
  journal={BioRxiv},
  year={2025}
}
```

**Issues**: https://github.com/Li-Lab-SJTU/XAMP/issues  
**Contact**: jing.li@sjtu.edu.cn
