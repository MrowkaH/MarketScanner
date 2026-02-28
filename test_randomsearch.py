"""Quick test to verify RandomizedSearchCV is working."""
import sys
sys.path.insert(0, 'PPI_Predictor')
sys.path.insert(0, 'PPI_Predictor/src')

from PPI_Predictor.src.predictor import Predictor
import pandas as pd
import numpy as np

print("Testing RandomizedSearchCV implementation...")
print("="*60)

# Create dummy data
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.randn(150),
    'feature2': np.random.randn(150),
    'feature3': np.random.randn(150),
    'target_ppi_mom': np.random.randn(150) * 0.5
})

predictor = Predictor()
print("\nTraining models...")
try:
    metrics = predictor.train(df)
    print("\n✓ Training completed successfully with RandomizedSearchCV!")
    print(f"\nRegression metrics:")
    for model, results in metrics['regression'].items():
        print(f"-  {model}: MAE={results['MAE']:.4f}, R2={results['R2']:.4f}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
