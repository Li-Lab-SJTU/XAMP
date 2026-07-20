#!/usr/bin/env python3
"""
Command line interface for XAMP - Antimicrobial Peptide Prediction Tool
"""

import argparse
import sys
import pandas as pd
from typing import List
from XAMP import XAMP, load_example_sequences

def read_sequences_from_file(filepath: str) -> List[str]:
    """
    Read sequences from input file.
    Supports FASTA format and plain text (one sequence per line).
    
    Args:
        filepath: Path to input file
        
    Returns:
        List of sequences
    """
    sequences = []
    
    try:
        # Try to detect file format
        with open(filepath, 'r') as f:
            first_line = f.readline()
            f.seek(0)  # Reset file pointer
            
            if first_line.startswith('>'):
                # FASTA format
                seq = ""
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if seq:
                            sequences.append(seq)
                            seq = ""
                    else:
                        seq += line
                if seq:
                    sequences.append(seq)
            else:
                # Plain text format (one sequence per line)
                for line in f:
                    line = line.strip()
                    if line:
                        sequences.append(line)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        sys.exit(1)
        
    return sequences

def write_results_to_file(results: pd.DataFrame, filepath: str):
    """
    Write prediction results to output file.
    
    Args:
        results: Prediction results DataFrame
        filepath: Output file path
    """
    try:
        results.to_csv(filepath, sep='\t', index=False)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error writing to file {filepath}: {e}")
        sys.exit(1)

def main():
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        sys.argv.append('--help')
    
    parser = argparse.ArgumentParser(
        description="XAMP - Antimicrobial Peptide Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  xamp-predict --input test.fasta --output test.xamp_pred.tsv --remove_n_met
  xamp-predict --input sequences.txt --device cuda --batch_size 128
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input file path (FASTA or text format).'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path. If not provided, results will be printed to console.'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        default='integrated',
        choices=['integrated', 'xamp_t', 'xamp_e'],
        help='Model type to use for prediction (default: integrated)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='mix',
        choices=['mix', 'unknown'],
        help='Training dataset variant: "mix" for general prediction (default), "unknown" for Benchmark 2 reproduction'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run inference on (default: auto)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for prediction (default: 64)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold for classification (default: 0.5)'
    )
    
    parser.add_argument(
        '--remove_n_met',
        action='store_true',
        help='Remove N-terminal methionine from sequences'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize predictor
    print("Initializing XAMP predictor...")
    predictor = XAMP(
        model_type=args.model_type,
        device=args.device,
        dataset=args.dataset
    )
    
    # Get sequences
    if args.input:
        print(f"Reading sequences from {args.input}...")
        sequences = read_sequences_from_file(args.input)
    else:
        print("Using example sequences...")
        sequences = load_example_sequences()
    
    print(f"Processing {len(sequences)} sequences...")
    
    # Make predictions
    results = predictor.predict(
        sequences,
        threshold=args.threshold,
        batch_size=args.batch_size,
        remove_n_met=args.remove_n_met
    )
    
    # Output results
    if args.output:
        write_results_to_file(results, args.output)
    else:
        print("\nPrediction Results:")
        print(results.to_string(index=False))

if __name__ == "__main__":
    main()