"""
XAMP - Antimicrobial Peptide Prediction Tool
A dual-engine framework combining ESM-2 and Transformer for AMP prediction
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional
import esm

import warnings
warnings.filterwarnings('ignore')

# Amino acid encoding dictionary
aa_encoding_dict = {'X':0, 'Y':1, 'S':2, 'M':3, 'R':4, 'E':5, 
                    'I':6, 'N':7, 'V':8, 'G':9, 'L':10,'D':11,
                    'T':12, 'W':13, 'H':14, 'K':15, 'A':16,
                    'F':17, 'Q':18, 'C':19, 'P':20,'*':21}



class model_XAMP(nn.Module):
    """
    XAMP Transformer-based model for antimicrobial peptide prediction.
    This model uses a custom Transformer architecture with positional encoding
    and global average pooling for sequence classification.
    """
    def __init__(self, dropout=0.2, num_heads=8, vocab_size=22, num_encoder_layers=1, d_embedding=128, max_len=300):
        """
        Initialize the XAMP Transformer model.
        
        Args:
            dropout: Dropout rate for regularization
            num_heads: Number of attention heads in Transformer
            vocab_size: Size of amino acid vocabulary (20 standard + 2 special tokens)
            num_encoder_layers: Number of Transformer encoder layers
            d_embedding: Dimension of embedding vectors
            max_len: Maximum sequence length
        """
        super(model_XAMP, self).__init__()

        # Token embedding layer
        self.embeddingLayer = nn.Embedding(vocab_size, d_embedding)
        
        # Sinusoidal positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embedding, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_embedding))
        pe = torch.zeros(max_len, d_embedding)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.positionalEncodings = nn.Parameter(pe, requires_grad=False)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_embedding, 
            num_heads, 
            dim_feedforward=1024, 
            dropout=dropout
        )
        encoder_norm = nn.LayerNorm(d_embedding)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)
        
        # Classification head
        self.fc1 = nn.Linear(d_embedding, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.outputlayer = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode_sequences(self, sequences):
        sequences_encoded = []
        for seq in sequences:
            seq_encoded = [aa_encoding_dict[AA.upper()] for AA in seq.ljust(300, "*")]
            sequences_encoded.append(seq_encoded)
        return torch.LongTensor(sequences_encoded)
    
    def make_len_mask(self, inp):
        mask = (inp == 21)  # 21 is the padding token index
        return mask
    
    def forward(self, sequences):
        # Encode sequences to numerical tensors
        sequences_encoded = self.encode_sequences(sequences).to(next(self.parameters()).device)
        
        # Create padding mask
        pad_mask = self.make_len_mask(sequences_encoded).to(next(self.parameters()).device)
        
        # Embed sequences and add positional encodings
        embeddings = self.embeddingLayer(sequences_encoded)
        embeddings = embeddings + self.positionalEncodings
        
        # Pass through Transformer encoder
        embeddings = embeddings.permute(1, 0, 2)  # (seq_len, batch_size, d_embedding)
        features = self.transformer_encoder(embeddings, src_key_padding_mask=pad_mask)
        features = features.permute(1, 0, 2)  # (batch_size, seq_len, d_embedding)
        
        # Global average pooling with padding mask
        mask_weights = 1 - pad_mask.float()
        representation = torch.sum(
            mask_weights.unsqueeze(2) * features, dim=1
        ) / torch.sum(mask_weights, dim=1).unsqueeze(1)

        # Classification head
        x = self.fc1(representation)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        output = self.outputlayer(x)
        output = self.sigmoid(output)
        
        return output


class model_ESM2(nn.Module):
    """
    ESM-2 based model for antimicrobial peptide prediction.
    This model uses pre-trained ESM-2 protein language model representations
    with a custom MLP classifier head.
    """
    def __init__(self, esm_model_name="esm2_t12_35M_UR50D", hidden_dim=256, output_dim=1, dropout=0.2):
        """
        Initialize the ESM-2 based model.
        
        Args:
            esm_model_name: Name of pre-trained ESM-2 model to load
            hidden_dim: Hidden dimension of MLP layers
            output_dim: Output dimension (1 for binary classification)
            dropout: Dropout rate for regularization
        """
        super(model_ESM2, self).__init__()
        
        # Load pre-trained ESM-2 model
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Freeze ESM-2 parameters to prevent fine-tuning
        for param in self.esm_model.parameters():
            param.requires_grad = False
            
        # Get model properties
        self.num_layers = self.esm_model.num_layers
        esm_output_dim = self.esm_model.embed_tokens.embedding_dim
        
        # MLP classification head
        self.mlp = nn.Sequential(
            nn.Linear(esm_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(self, sequences):
        # Convert sequences to ESM-2 input format
        data = [(str(i), seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(next(self.parameters()).device)

        # Extract ESM-2 representations
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[self.num_layers])
            esm_representations = results["representations"][self.num_layers]
        
        # Create padding mask
        mask = batch_tokens != self.alphabet.padding_idx
        
        # Global average pooling with padding mask
        valid_lengths = mask.sum(dim=1).float()
        masked_representations = esm_representations * mask.unsqueeze(-1)
        sequence_representations = masked_representations.sum(dim=1) / valid_lengths.unsqueeze(-1)

        # Pass through MLP classifier
        output = self.mlp(sequence_representations)
        return output


def check_has_sigmoid_layer(model):
    """
    检查模型最后一层为sigmoid
    return: bool
    """
    last_layer = list(model.children())[-1]
    
    # 处理Sequential情况
    if isinstance(last_layer, nn.Sequential):
        return isinstance(last_layer[-1], nn.Sigmoid)
    return isinstance(last_layer, nn.Sigmoid)


class XAMP:
    """
    XAMP Antimicrobial Peptide Prediction Tool
    
    A dual-engine framework that combines:
    - XAMP-T: Transformer-based model for fast screening
    - XAMP-E: ESM-2-based model for high accuracy prediction
    """
    
    def __init__(self, 
                 model_type: str = 'integrated',
                 device: str = 'auto',
                 model_paths: dict = None):
        """
        Initialize XAMP predictor
        
        Args:
            model_type (str): Type of model to use. Options:
                'integrated' - Use both models and take higher probability (default)
                'xamp_t' - Use only XAMP-T (fast)
                'xamp_e' - Use only XAMP-E (accurate)
            device (str): Device to run inference on. 'auto', 'cuda', or 'cpu'
            model_paths (dict): Custom paths to model files
        """
        self.model_type = model_type
        self.device = self._setup_device(device)
        
        # Default model paths (update these with your actual model paths)
        default_paths = {
            'xamp_t': 'models/xamp_t.state_dict.pth',
            'xamp_e': 'models/xamp_e.state_dict.pth'
        }

        self.model_paths = model_paths or default_paths
        self.models = {}
        
        self._load_models()
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_models(self):
        """Load the selected models using state_dict (recommended approach)"""
        if self.model_type in ['integrated', 'xamp_t']:
            try:
                # 创建模型实例
                model = model_XAMP().to(self.device)
                # 加载state_dict
                model.load_state_dict(torch.load(
                    self.model_paths['xamp_t'], 
                    map_location=self.device
                ))
                model.eval()
                self.models['xamp_t'] = model
                print("✓ XAMP-T model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load XAMP-T model: {e}")
        
        if self.model_type in ['integrated', 'xamp_e']:
            try:
                # 创建模型实例
                model = model_ESM2().to(self.device)
                # 加载state_dict
                model.load_state_dict(torch.load(
                    self.model_paths['xamp_e'], 
                    map_location=self.device
                ))
                model.eval()
                self.models['xamp_e'] = model
                print("✓ XAMP-E model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load XAMP-E model: {e}")
    
    def preprocess_sequences(self, 
                             sequences: List[str], 
                             remove_n_met: bool = True) -> List[str]:
        """
        Preprocess peptide sequences
        
        Args:
            sequences: List of peptide sequences
            remove_n_met: Whether to remove N-terminal methionine
            
        Returns:
            Preprocessed sequences
        """
        processed = []
        for seq in sequences:
            # Convert to uppercase and remove whitespace
            seq = seq.upper().strip()
            
            # Remove N-terminal methionine if specified
            if remove_n_met and seq.startswith('M'):
                seq = seq[1:]
            
            # Validate amino acids
            if not all(aa in aa_encoding_dict for aa in seq):
                invalid_aas = set(seq) - set(aa_encoding_dict.keys())
                warnings.warn(f"Sequence contains invalid amino acids: {invalid_aas}")
            
            processed.append(seq)
        
        return processed
    
    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """
        Encode sequences to tensor format
        
        Args:
            sequences: List of peptide sequences
            
        Returns:
            Encoded sequences tensor
        """
        max_len = 300
        sequences_encoded = []
        
        for seq in sequences:
            # Pad or truncate to max_len
            padded_seq = seq.ljust(max_len, "*")[:max_len]
            seq_encoded = [aa_encoding_dict[aa] for aa in padded_seq]
            sequences_encoded.append(seq_encoded)
        
        return torch.LongTensor(sequences_encoded).to(self.device)
    
    def predict(self, 
                sequences: Union[str, List[str]], 
                threshold: float = 0.5,
                batch_size: int = 32,
                remove_n_met: bool = True) -> pd.DataFrame:
        """
        Predict antimicrobial activity for peptide sequences
        
        Args:
            sequences: Single sequence or list of sequences
            threshold: Probability threshold for classification
            batch_size: Batch size for prediction
            remove_n_met: Remove N-terminal methionine during preprocessing
            
        Returns:
            DataFrame with predictions including:
            - sequence: Original sequence
            - prob_xamp_t: XAMP-T prediction probability
            - prob_xamp_e: XAMP-E prediction probability  
            - prob_final: Final integrated probability
            - prediction: Binary prediction (1=AMP, 0=non-AMP)
        """
        # Handle single sequence input
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # Preprocess sequences
        processed_seqs = self.preprocess_sequences(sequences, remove_n_met)
        
        results = []
        
        # Process in batches
        for i in range(0, len(processed_seqs), batch_size):
            batch_seqs = processed_seqs[i:i + batch_size]
            batch_results = self._predict_batch(batch_seqs, threshold)
            results.extend(batch_results)
        
        # Create results DataFrame
        df_results = pd.DataFrame(results)
        df_results.insert(0, 'seq', sequences[:len(df_results)])
        
        return df_results
    
    def _predict_batch(self, sequences: List[str], threshold: float) -> List[dict]:
        """Predict a single batch of sequences"""
        with torch.no_grad():
            encoded_seqs = self.encode_sequences(sequences)
            
            results = []
            
            for i, seq in enumerate(sequences):
                result = {'seq_processed': seq}
                
                # XAMP-T prediction
                if 'xamp_t' in self.models:
                    try:
                        prob_t = self.models['xamp_t']([seq]).cpu().numpy()[0][0]
                        if not check_has_sigmoid_layer(self.models['xamp_t']):  # 如果最后一层不是sigmoid，使用Focal loss
                            prob_t = torch.sigmoid(torch.tensor(prob_t)).numpy()  # 将logits转换为概率
                        result['prob_xamp_t'] = float(prob_t)

                    except Exception as e:
                        result['prob_xamp_t'] = np.nan
                        warnings.warn(f"XAMP-T prediction failed for sequence {i}: {e}")
                
                # XAMP-E prediction  
                if 'xamp_e' in self.models:
                    try:
                        prob_e = self.models['xamp_e']([seq]).cpu().numpy()[0][0]
                        if not check_has_sigmoid_layer(self.models['xamp_e']):  # 如果最后一层不是sigmoid，使用Focal loss
                            prob_e = torch.sigmoid(torch.tensor(prob_e)).numpy()  # 将logits转换为概率
                        result['prob_xamp_e'] = float(prob_e)
                    except Exception as e:
                        result['prob_xamp_e'] = np.nan
                        warnings.warn(f"XAMP-E prediction failed for sequence {i}: {e}")
                
                # Calculate final probability
                probs = []
                if 'prob_xamp_t' in result and not np.isnan(result['prob_xamp_t']):
                    probs.append(result['prob_xamp_t'])
                if 'prob_xamp_e' in result and not np.isnan(result['prob_xamp_e']):
                    probs.append(result['prob_xamp_e'])
                
                if probs:
                    result['prob_final'] = min(probs)  # Take lower probability
                    result['prediction'] = 1 if result['prob_final'] >= threshold else 0
                else:
                    result['prob_final'] = np.nan
                    result['prediction'] = np.nan  
                
                results.append(result)
            
            return results
    
    def predict_proba(self, 
                     sequences: Union[str, List[str]], 
                     **kwargs) -> np.ndarray:
        """
        Predict probabilities only (no classification)
        
        Args:
            sequences: Single sequence or list of sequences
            **kwargs: Additional arguments passed to predict()
            
        Returns:
            Array of probabilities
        """
        df = self.predict(sequences, **kwargs)
        return df['prob_final'].values
    
    def predict_binary(self, 
                      sequences: Union[str, List[str]], 
                      **kwargs) -> np.ndarray:
        """
        Predict binary classifications only
        
        Args:
            sequences: Single sequence or list of sequences
            **kwargs: Additional arguments passed to predict()
            
        Returns:
            Array of binary predictions (1=AMP, 0=non-AMP)
        """
        df = self.predict(sequences, **kwargs)
        return df['prediction'].values
    
    def benchmark_models(self, 
                       sequences: List[str],
                       remove_n_met: bool = True) -> pd.DataFrame:
        """
        Benchmark both models and compare predictions
        
        Args:
            sequences: List of sequences to benchmark
            remove_n_met: Remove N-terminal methionine
            
        Returns:
            DataFrame with detailed benchmarking results
        """
        results = self.predict(sequences, remove_n_met=remove_n_met)
        
        # Add agreement analysis
        if 'prob_xamp_t' in results.columns and 'prob_xamp_e' in results.columns:
            results['agreement'] = results['prob_xamp_t'] * results['prob_xamp_e']
            results['disagreement'] = abs(results['prob_xamp_t'] - results['prob_xamp_e'])
        
        return results
    
    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        info = {
            'model_type': self.model_type,
            'device': str(self.device),
            'loaded_models': list(self.models.keys()),
            'xamp_t_available': 'xamp_t' in self.models,
            'xamp_e_available': 'xamp_e' in self.models
        }
        
        # Add model specifics if available
        for name, model in self.models.items():
            info[f'{name}_parameters'] = sum(p.numel() for p in model.parameters())
        
        return info

# Example usage and helper functions
def predict_amp(sequences: Union[str, List[str]], 
                model_type: str = 'integrated',
                threshold: float = 0.5,
                **kwargs) -> pd.DataFrame:
    """
    Quick prediction function for convenience
    
    Args:
        sequences: Sequence or list of sequences to predict
        model_type: 'integrated', 'xamp_t', or 'xamp_e'
        threshold: Classification threshold
        **kwargs: Additional arguments for XAMP
        
    Returns:
        Prediction results
    """
    predictor = XAMP(model_type=model_type, **kwargs)
    return predictor.predict(sequences, threshold=threshold)

def load_example_sequences() -> List[str]:
    """Load example sequences for testing"""
    return [
        "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "MKGLGLGLGLGLGLGLGLGL",
        "ACDEFGHIKLMNPQRSTVWY",
        "RRRRRRRRRR",
        "LLLLLLLLLL"
    ]

if __name__ == "__main__":
    # Example usage
    print("XAMP - Antimicrobial Peptide Predictor")
    print("=" * 50)
    
    # Initialize predictor
    predictor = XAMP(model_type='integrated')
    
    # Get model info
    info = predictor.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example sequences
    examples = load_example_sequences()
    print(f"\nTesting with {len(examples)} example sequences...")
    
    # Make predictions
    results = predictor.predict(examples)
    print("\nPrediction Results:")
    print(results.to_string(index=False))
    
    # Quick prediction example
    print("\nQuick prediction example:")
    quick_result = predict_amp("LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES")
    print(quick_result.to_string(index=False))