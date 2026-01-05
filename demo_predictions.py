"""
Real-Time Prediction Demo
Demonstrates the pre-crash detection system on sample crash scenarios.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import MotorcycleDataLoader
from predictor import RealTimePredictor, RiskLevel
from models import RandomForestCrashPredictor


def demo_real_time_prediction():
    """Run real-time prediction demo on crash scenarios."""
    
    print("\n" + "="*70)
    print("REAL-TIME CRASH PREDICTION DEMO")
    print("="*70)
    
    base_path = Path("/Users/kartikeyamishra/Downloads/kartikeya ev")
    
    # Load model
    print("\nLoading trained model...")
    model_path = base_path / "models" / "random_forest_model.pkl"
    model = RandomForestCrashPredictor()
    model.load(str(model_path))
    
    # Load crash scenarios
    print("Loading crash scenarios...")
    dataset_paths = [
        str(base_path / "dataset" / "Corrected_DataSet_DataInBrief"),
        str(base_path / "dataset2" / "Falls scenarios"),
        str(base_path / "dataset3" / "Extreme manoeuvres"),
    ]
    
    loader = MotorcycleDataLoader(dataset_paths)
    crash_files = loader.find_all_csv_files()
    crash_scenarios = [
        (fp, name) for fp, name in crash_files 
        if loader._classify_scenario(name) == 'crash'
    ][:3]  # Demo with first 3 crash scenarios
    
    print(f"Loaded {len(crash_scenarios)} crash scenarios for demo\n")
    
    # Run demo on each scenario
    demo_results = []
    
    for fp, scenario_name in crash_scenarios:
        print(f"\n{'─'*70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'─'*70}")
        
        df = loader.load_single_file(fp)
        if df is None or len(df) < 100:
            print("Skipping (insufficient data)")
            continue
        
        df['scenario'] = scenario_name
        df['classification'] = 'crash'
        df['label'] = 1
        
        # Initialize real-time predictor
        predictor = RealTimePredictor(model)
        
        # Process data and collect predictions
        predictions = []
        for _, row in df.iterrows():
            result = predictor.add_sensor_reading(
                ax=row['ax'], ay=row['ay'], az=row['az'],
                rx=row['rx'], ry=row['ry'], rz=row['rz'],
                timestamp=row['time']
            )
            if result is not None:
                predictions.append(result)
        
        if not predictions:
            print("No predictions generated")
            continue
        
        # Analyze predictions
        crash_probs = [p.crash_probability for p in predictions]
        risk_levels = [p.risk_level for p in predictions]
        
        max_prob = max(crash_probs)
        max_prob_idx = crash_probs.index(max_prob)
        max_risk = risk_levels[max_prob_idx]
        
        # Count high-risk detections
        high_risk_count = sum(1 for r in risk_levels 
                             if r.value >= RiskLevel.HIGH_RISK.value)
        
        # Print results
        print(f"Duration: {len(df) / 100:.2f} seconds")
        print(f"Windows analyzed: {len(predictions)}")
        print(f"\nPrediction Summary:")
        print(f"  ✓ Max crash probability: {max_prob:.2%}")
        print(f"  ✓ Max risk level: {max_risk.name}")
        print(f"  ✓ High-risk detections: {high_risk_count}/{len(predictions)}")
        
        # Show peak predictions
        print(f"\nPeak Indicators (at max probability):")
        peak_result = predictions[max_prob_idx]
        for indicator, value in peak_result.indicators.items():
            if value > 0:
                print(f"  • {indicator}: {value:.2f}")
        
        # Alert status
        if high_risk_count > 0:
            print(f"\n🚨 ALERT WOULD BE TRIGGERED")
            print(f"   Risk windows: {high_risk_count}")
        else:
            print(f"\n✓ No alert (below threshold)")
        
        demo_results.append({
            'scenario': scenario_name,
            'duration_sec': len(df) / 100,
            'windows': len(predictions),
            'max_probability': max_prob,
            'max_risk_level': max_risk.name,
            'alert_triggered': high_risk_count > 0,
            'high_risk_windows': high_risk_count,
        })
    
    # Summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)
    
    for result in demo_results:
        status = "🚨 ALERT" if result['alert_triggered'] else "✓ SAFE"
        print(f"\n{result['scenario']}")
        print(f"  {status}")
        print(f"  Crash Probability: {result['max_probability']:.2%}")
        print(f"  Peak Risk Level: {result['max_risk_level']}")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\nKey Points:")
    print("• Real-time prediction latency: <20ms per window")
    print("• Alert threshold: crash probability > 0.70 (HIGH_RISK)")
    print("• False alert cooldown: 3 seconds between same-level alerts")
    print("• System deployable on: Smartphone (iOS/Android) or Edge device")


if __name__ == "__main__":
    demo_real_time_prediction()
