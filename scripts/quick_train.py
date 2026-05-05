"""Quick smoke-test trainer: trains on a small subset to validate pipeline."""
from pathlib import Path
import joblib
import pandas as pd

import sys
from pathlib import Path
# ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.airquality_pipeline import prepare_for_model
from src.models.pollution import PollutionModel


def main():
    try:
        features_path = Path("src/data/processed/airquality_features_v11.csv")
        print("Looking for features file at:", features_path)
        if not features_path.exists():
            print("features file not found:", features_path)
            return

        print("Reading features CSV...")
        df = pd.read_csv(features_path)
        print("Features shape:", df.shape)
        # use a small subset
        df = df.head(500)
        print("Subset shape:", df.shape)
        X, y, scaler = prepare_for_model(df, target="C6H6(GT)")
        print("Prepared X,y shapes:", getattr(X, 'shape', None), getattr(y, 'shape', None))
        model = PollutionModel()
        # make model lightweight for quick run
        try:
            model.model.set_params(n_estimators=10, max_depth=8)
        except Exception:
            pass
        print("Training model (small)...")
        model.train(X, y)
        out = Path("src/artifacts/test_model_quick.joblib")
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(out)
        print("Quick train completed, model saved to", out)
    except Exception as e:
        import traceback

        print("Error during quick_train:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
