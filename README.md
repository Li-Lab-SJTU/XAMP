<<<<<<< HEAD
# XAMP: Dual-Engine Antimicrobial Peptide Prediction

A dual-engine deep learning framework for accurate and efficient antimicrobial peptide (AMP) prediction.

## 📊 Model Performance

| Model | AUC | Speed | Use Case |
|-------|-----|-------|----------|
| XAMP-T | 0.9668 | Fast | Large-scale screening |
| XAMP-E | 0.9706 | Accurate | High-confidence prediction |
| **Integrated** | **0.9715** | Balanced | **Recommended** |

## 📦 Installation

```bash
# Using conda environment
conda env create -f XAMP.yaml
conda activate XAMP

# Install package in development mode to enable CLI
pip install -e .

# Unzip pre-trained models
gunzip models/xamp_*.state_dict.pth.gz
```

## 🔧 Usage

```python
from xamp import XAMP

predictor = XAMP(
    model_type='integrated',  # 'integrated', 'xamp_t', or 'xamp_e'
    device='cuda'  # Use GPU if available
)

# Batch prediction
results = predictor.predict(sequences, batch_size=64, remove_n_met=True)
```

## 📋 Output Format

Returns DataFrame with:
- `sequence`: Input sequence
- `probability_xamp_t`: XAMP-T probability
- `probability_xamp_e`: XAMP-E probability  
- `probability_final`: Final probability
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
=======
# XAMP: Dual-Engine Antimicrobial Peptide Prediction

A dual-engine deep learning framework for accurate and efficient antimicrobial peptide (AMP) prediction.

## 📊 Model Performance

| Model | AUC | Speed | Use Case |
|-------|-----|-------|----------|
| XAMP-T | 0.9668 | Fast | Large-scale screening |
| XAMP-E | 0.9706 | Accurate | High-confidence prediction |
| **Integrated** | **0.9715** | Balanced | **Recommended** |

## 📦 Installation

```bash
# Using conda environment
conda env create -f XAMP.yaml
conda activate XAMP

# Install package in development mode to enable CLI
pip install -e .
```

## 🔧 Usage

```python
from xamp import XAMP

predictor = XAMP(
    model_type='integrated',  # 'integrated', 'xamp_t', or 'xamp_e'
    device='cuda'  # Use GPU if available
)

# Batch prediction
results = predictor.predict(sequences, batch_size=64, remove_n_met=True)
```

## 📋 Output Format

Returns DataFrame with:
- `sequence`: Input sequence
- `probability_xamp_t`: XAMP-T probability
- `probability_xamp_e`: XAMP-E probability  
- `probability_final`: Final probability
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
>>>>>>> c23de09314e7aae4c5d2ecfd8180921a998209f5
