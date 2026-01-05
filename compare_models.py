"""
Model Comparison Script
Trains and compares multiple ML models for crash prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

# Models to compare
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import MotorcycleDataLoader
from feature_extractor import prepare_training_data, FeatureConfig


class ModelComparator:
    """Compare multiple models on the same dataset."""
    
    def __init__(self, X_train, X_test, y_train, y_test, feature_names):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.results = {}
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE
        class_counts = np.bincount(y_train.astype(int))
        if min(class_counts) >= 6:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(class_counts)-1))
            self.X_train_scaled, self.y_train = smote.fit_resample(self.X_train_scaled, y_train)
    
    def evaluate_model(self, name, model):
        """Train and evaluate a single model."""
        print(f"\nTraining {name}...")
        try:
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                y_proba = model.decision_function(self.X_test_scaled) if hasattr(model, 'decision_function') else y_pred
            
            # Compute metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred, zero_division=0),
                'f1': f1_score(self.y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(self.y_test, y_proba) if len(np.unique(self.y_test)) > 1 else 0.5,
            }
            
            self.results[name] = metrics
            print(f"  ✓ {name}: Acc={metrics['accuracy']:.4f}, Rec={metrics['recall']:.4f}, AUC={metrics['auc']:.4f}")
            return metrics
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            return None
    
    def compare_all(self):
        """Train and compare all models."""
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, early_stopping=True, random_state=42),
        }
        
        print("="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        for name, model in models.items():
            self.evaluate_model(name, model)
        
        return self.results
    
    def generate_report(self, output_dir):
        """Generate comparison report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        results_df = results_df.sort_values('recall', ascending=False)  # Sort by recall (crash detection)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON RESULTS")
        print("="*70)
        print(results_df.to_string())
        
        # Save CSV
        csv_path = output_path / "model_comparison.csv"
        results_df.to_csv(csv_path)
        print(f"\n✓ Saved to {csv_path}")
        
        # Save JSON
        json_path = output_path / "model_comparison.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved to {json_path}")
        
        # Generate visualization
        self.plot_comparison(results_df, output_path)
        
        # Print top model
        best_recall = results_df['recall'].idxmax()
        best_balanced = (results_df['precision'] * results_df['recall']).idxmax()
        print(f"\n🏆 Best for Crash Detection (Recall): {best_recall}")
        print(f"🏆 Best Balanced (Precision × Recall): {best_balanced}")
    
    def plot_comparison(self, results_df, output_path):
        """Create comparison plots."""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            data = results_df[metric].sort_values(ascending=False)
            colors = ['green' if m == data.idxmax() else 'steelblue' for m in data.index]
            ax.barh(data.index, data.values, color=colors)
            ax.set_xlabel(metric.capitalize())
            ax.set_xlim(0, 1)
            for i, v in enumerate(data.values):
                ax.text(v + 0.02, i, f'{v:.4f}', va='center')
        
        # Hide the 6th subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plot_path = output_path / "model_comparison_chart.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {plot_path}")
        plt.close()


def main():
    """Main comparison pipeline."""
    base_path = Path("/Users/kartikeyamishra/Downloads/kartikeya ev")
    
    # Load data
    print("Loading data...")
    dataset_paths = [
        str(base_path / "dataset" / "Corrected_DataSet_DataInBrief"),
        str(base_path / "dataset2" / "Falls scenarios"),
        str(base_path / "dataset3" / "Extreme manoeuvres"),
    ]
    
    loader = MotorcycleDataLoader(dataset_paths)
    data = loader.load_all_data()
    
    # Extract features
    print("Extracting features...")
    config = FeatureConfig(window_size_sec=1.0, step_size_sec=0.5, sampling_rate=100)
    X, y, feature_names = prepare_training_data(data, config)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Compare models
    comparator = ModelComparator(X_train, X_test, y_train, y_test, feature_names)
    comparator.compare_all()
    
    # Generate report
    output_dir = base_path / "output"
    comparator.generate_report(output_dir)


if __name__ == "__main__":
    main()
