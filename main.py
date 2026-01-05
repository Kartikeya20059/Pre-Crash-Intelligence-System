"""
Pre-Crash Intelligence System for Two-Wheelers
Main Pipeline Entry Point

This script runs the complete pipeline:
1. Load sensor data from all datasets
2. Perform EDA and generate visualizations
3. Extract features and prepare training data
4. Train and evaluate ML models
5. Generate performance report
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import MotorcycleDataLoader
from feature_extractor import FeatureExtractor, FeatureConfig, prepare_training_data
from models import train_and_evaluate_models, RandomForestCrashPredictor
from eda_analysis import run_eda


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def main(args):
    """Main pipeline execution."""
    base_path = Path(args.base_path)
    
    # Dataset paths
    dataset_paths = [
        str(base_path / "dataset" / "Corrected_DataSet_DataInBrief"),
        str(base_path / "dataset2" / "Falls scenarios"),
        str(base_path / "dataset3" / "Extreme manoeuvres"),
    ]
    
    # Output directories
    output_dir = base_path / "output"
    viz_dir = output_dir / "visualizations"
    models_dir = base_path / "models"
    
    for d in [output_dir, viz_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print_header("STEP 1: LOADING SENSOR DATA")
    
    loader = MotorcycleDataLoader(dataset_paths)
    
    # Get scenario summary
    print("\nDataset Summary:")
    summary = loader.get_scenario_summary()
    print(summary.to_string(index=False))
    
    # Load all data
    data = loader.load_all_data()
    
    print(f"\n✓ Loaded {len(data):,} sensor samples")
    print(f"  - Normal scenarios: {len(data[data['classification'] == 'normal']):,} samples")
    print(f"  - High-risk scenarios: {len(data[data['classification'] == 'high_risk']):,} samples")
    print(f"  - Crash scenarios: {len(data[data['classification'] == 'crash']):,} samples")
    
    # =========================================================================
    # STEP 2: Exploratory Data Analysis
    # =========================================================================
    if not args.skip_eda:
        print_header("STEP 2: EXPLORATORY DATA ANALYSIS")
        
        viz_files = run_eda(data, str(viz_dir))
        print(f"\n✓ Generated {len(viz_files)} visualization files")
    else:
        print_header("STEP 2: SKIPPING EDA (--skip-eda flag set)")
    
    # =========================================================================
    # STEP 3: Feature Extraction
    # =========================================================================
    print_header("STEP 3: FEATURE EXTRACTION")
    
    config = FeatureConfig(
        window_size_sec=args.window_size,
        step_size_sec=args.step_size,
        sampling_rate=100
    )
    
    print(f"Window configuration:")
    print(f"  - Window size: {config.window_size_sec}s ({config.window_samples} samples)")
    print(f"  - Step size: {config.step_size_sec}s ({config.step_samples} samples)")
    
    X, y, feature_names = prepare_training_data(data, config)
    
    print(f"\n✓ Feature extraction complete")
    print(f"  - Feature matrix shape: {X.shape}")
    print(f"  - Number of features: {len(feature_names)}")
    print(f"  - Class distribution: Safe={sum(y==0)}, Dangerous={sum(y==1)}")
    
    # Save feature names
    feature_info = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_samples': X.shape[0],
        'window_size_sec': config.window_size_sec,
        'step_size_sec': config.step_size_sec,
    }
    with open(output_dir / "feature_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # =========================================================================
    # STEP 4: Model Training and Evaluation
    # =========================================================================
    print_header("STEP 4: MODEL TRAINING AND EVALUATION")
    
    results, best_model, importance_df = train_and_evaluate_models(
        X, y, feature_names, test_size=args.test_size
    )
    
    # Save feature importance
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    
    # =========================================================================
    # STEP 5: Generate Performance Report
    # =========================================================================
    print_header("STEP 5: PERFORMANCE REPORT")
    
    report = {
        'dataset_summary': {
            'total_samples': len(data),
            'total_windows': X.shape[0],
            'n_features': len(feature_names),
            'scenarios': summary.to_dict('records'),
        },
        'model_results': {},
        'best_model': None,
        'early_warning_capability': {
            'target_seconds': '1-3',
            'window_size_sec': config.window_size_sec,
            'achievable': True,
        }
    }
    
    best_recall = 0
    for name, metrics in results.items():
        report['model_results'][name] = metrics.to_dict()
        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics.accuracy:.4f}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall:    {metrics.recall:.4f} (Crash Detection Rate)")
        print(f"  F1 Score:  {metrics.f1:.4f}")
        print(f"  AUC:       {metrics.auc:.4f}")
        
        if metrics.recall > best_recall:
            best_recall = metrics.recall
            report['best_model'] = name
    
    print(f"\n✓ Best Model for Crash Detection: {report['best_model']}")
    print(f"  Crash Detection Rate (Recall): {best_recall:.2%}")
    
    # Save report
    with open(output_dir / "performance_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print_header("PIPELINE COMPLETE")
    print(f"\nOutput files saved to: {output_dir}")
    print(f"  - feature_info.json")
    print(f"  - feature_importance.csv")
    print(f"  - performance_report.json")
    print(f"\nModels saved to: {models_dir}")
    print(f"  - random_forest_model.pkl")
    print(f"  - gradient_boosting_model.pkl")
    print(f"\nVisualizations saved to: {viz_dir}")
    
    return report


def demo_mode(args):
    """Run demo with trained model."""
    print_header("RUNNING DEMO MODE")
    
    from predictor import RealTimePredictor, demo_predictor
    
    base_path = Path(args.base_path)
    model_path = base_path / "models" / "random_forest_model.pkl"
    
    if not model_path.exists():
        print("No trained model found. Please run training first:")
        print("  python main.py --mode train")
        return
    
    # Load model
    model = RandomForestCrashPredictor()
    model.load(str(model_path))
    
    # Load sample data
    dataset_paths = [
        str(base_path / "dataset" / "Corrected_DataSet_DataInBrief"),
        str(base_path / "dataset2" / "Falls scenarios"),
        str(base_path / "dataset3" / "Extreme manoeuvres"),
    ]
    
    loader = MotorcycleDataLoader(dataset_paths)
    data = loader.load_all_data()
    
    # Run demo on crash scenarios
    crash_data = data[data['classification'] == 'crash'].head(1000)
    
    if len(crash_data) == 0:
        print("No crash scenarios found in data")
        return
    
    demo_predictor(model, crash_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-Crash Intelligence System for Two-Wheelers"
    )
    
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "demo", "test"],
        help="Execution mode: train, demo, or test"
    )
    
    parser.add_argument(
        "--base-path", type=str,
        default=r"d:\codes\python\kartikeya ev",
        help="Base path to project directory"
    )
    
    parser.add_argument(
        "--window-size", type=float, default=1.0,
        help="Window size in seconds for feature extraction"
    )
    
    parser.add_argument(
        "--step-size", type=float, default=0.5,
        help="Step size in seconds for sliding window"
    )
    
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Proportion of data for testing"
    )
    
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip EDA visualization generation"
    )
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demo_mode(args)
    else:
        main(args)
