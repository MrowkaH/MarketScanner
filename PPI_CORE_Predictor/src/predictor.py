"""
PPI Core Predictor - ML Prediction Engine

Dual approach:
  1. REGRESSION  - Gradient Boosting, Random Forest, Ridge for point estimates
  2. CLASSIFICATION - Gradient Boosting + Random Forest classifier to produce
     probability distribution over discrete PPI Core MoM bins (e.g. -0.5%, -0.4%, ..., +0.5%)

Uses TimeSeriesSplit for proper temporal cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config.settings import (
    MODELS,
    PRIMARY_MODEL,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    CONFIDENCE_LEVEL,
    DISPLAY_DECIMALS,
    BIN_STEP,
    BIN_RANGE,
)


class Predictor:
    """ML engine for PPI Core prediction - regression + classification."""

    def __init__(self):
        # Regression
        self.reg_models: dict = {}
        self.reg_scaler = StandardScaler()
        self.reg_imputer = SimpleImputer(strategy="median")

        # Classification
        self.clf_models: dict = {}
        self.clf_scaler = StandardScaler()
        self.clf_imputer = SimpleImputer(strategy="median")
        self.bin_edges: np.ndarray | None = None
        self.bin_labels: list[str] = []

        self.feature_names: list[str] = []
        self.is_trained = False

        # Results cache
        self.cv_scores: dict = {}
        self.test_metrics: dict = {}
        self.clf_metrics: dict = {}

    # ══════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train regression + classification models.

        Returns dict with evaluation metrics for both approaches.
        """
        X, y = self._prepare_xy(df)

        # Temporal train/test split
        split_idx = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # ── REGRESSION ────────────────────────────────
        X_train_reg = self.reg_imputer.fit_transform(X_train)
        X_test_reg = self.reg_imputer.transform(X_test)
        X_train_reg = self.reg_scaler.fit_transform(X_train_reg)
        X_test_reg = self.reg_scaler.transform(X_test_reg)

        print(f"\n{'='*60}")
        print(f" REGRESSION MODEL TRAINING")
        print(f"  Train: {X_train_reg.shape[0]}  |  Test: {X_test_reg.shape[0]}  |  Features: {X_train_reg.shape[1]}")
        print(f"{'='*60}")

        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

        for name, params in MODELS.items():
            base_model = self._build_reg_model(name, {})
            param_grid = self._get_param_grid(name)
            n_iter = 15 if name == "gradient_boosting" else 12 if name == "random_forest" else 3
            
            print(f"\n  {name} - RandomizedSearchCV in progress...")
            
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,  # Matches parameter space size
                cv=tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                verbose=0,
                random_state=RANDOM_STATE,
            )
            random_search.fit(X_train_reg, y_train)
            
            best_model = random_search.best_estimator_
            self.reg_models[name] = best_model
            
            # Use RandomizedSearchCV best score (already cross-validated)
            self.cv_scores[name] = {
                "mean_mae": -random_search.best_score_,
                "std_mae": random_search.cv_results_['std_test_score'][random_search.best_index_]
            }

            y_pred = best_model.predict(X_test_reg)
            self.test_metrics[name] = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R2": r2_score(y_test, y_pred),
            }
            print(f"     Best params: {random_search.best_params_}")
            print(f"     CV MAE  : {self.cv_scores[name]['mean_mae']:.4f} +/- {self.cv_scores[name]['std_mae']:.4f}")
            print(f"     Test MAE: {self.test_metrics[name]['MAE']:.4f}")
            print(f"     Test R2 : {self.test_metrics[name]['R2']:.4f}")

        # ── CLASSIFICATION ────────────────────────────
        y_binned_train, y_binned_test = self._bin_target(y_train, y_test)

        X_train_clf = self.clf_imputer.fit_transform(X_train)
        X_test_clf = self.clf_imputer.transform(X_test)
        X_train_clf = self.clf_scaler.fit_transform(X_train_clf)
        X_test_clf = self.clf_scaler.transform(X_test_clf)

        print(f"\n{'='*60}")
        print(f" CLASSIFICATION MODEL TRAINING  (bins: {len(self.bin_labels)})")
        print(f"{'='*60}")

        clf_configs = {
            "gb_classifier": {
                "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 4],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0],
                }
            },
            "rf_classifier": {
                "model": RandomForestClassifier(random_state=RANDOM_STATE),
                "params": {
                    "n_estimators": [200, 300],
                    "max_depth": [8, 10, None],
                    "min_samples_split": [5, 10],
                }
            },
        }

        for name, config in clf_configs.items():
            print(f"\n  {name} - RandomizedSearchCV in progress...")
            
            random_search = RandomizedSearchCV(
                estimator=config["model"],
                param_distributions=config["params"],
                n_iter=10,  # Test 10 random combinations
                cv=2,  # Use 2-fold (some classes have few samples)
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
                random_state=RANDOM_STATE,
            )
            random_search.fit(X_train_clf, y_binned_train)
            
            best_clf = random_search.best_estimator_
            self.clf_models[name] = best_clf

            y_pred_cls = best_clf.predict(X_test_clf)
            y_proba = best_clf.predict_proba(X_test_clf)

            acc = accuracy_score(y_binned_test, y_pred_cls)
            try:
                ll = log_loss(y_binned_test, y_proba, labels=best_clf.classes_)
            except Exception:
                ll = float("nan")

            self.clf_metrics[name] = {"accuracy": acc, "log_loss": ll}
            print(f"     Best params: {random_search.best_params_}")
            print(f"     Accuracy: {acc:.4f}")
            print(f"     Log Loss: {ll:.4f}")

        self.is_trained = True
        print(f"\n{'='*60}\n")
        return {"regression": self.test_metrics, "classification": self.clf_metrics}

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Generate predictions from both regression and classification models.

        Returns dict with:
          - regression: point estimates per model + ensemble
          - probability_distribution: list of (bin_label, probability%)
        """
        if not self.is_trained:
            raise RuntimeError("Models not trained yet. Call train() first.")

        X, _ = self._prepare_xy(df)
        X_latest = X[-1:].copy()

        # ── Regression predictions ────────────────────
        X_reg = self.reg_scaler.transform(self.reg_imputer.transform(X_latest))
        reg_predictions = {}
        for name, model in self.reg_models.items():
            reg_predictions[name] = round(model.predict(X_reg)[0], DISPLAY_DECIMALS)

        # Ensemble (inverse-MAE weighted)
        weights = {n: 1.0 / max(self.test_metrics[n]["MAE"], 1e-6) for n in self.reg_models}
        total_w = sum(weights.values())
        ensemble = sum(reg_predictions[n] * weights[n] for n in reg_predictions) / total_w
        reg_predictions["ensemble"] = round(ensemble, DISPLAY_DECIMALS)

        # Confidence interval
        preds = [v for k, v in reg_predictions.items() if k != "ensemble"]
        std_dev = np.std(preds)
        z = 1.645 if CONFIDENCE_LEVEL == 0.90 else 1.96
        reg_predictions["confidence_interval"] = {
            "low": round(ensemble - z * std_dev, DISPLAY_DECIMALS),
            "high": round(ensemble + z * std_dev, DISPLAY_DECIMALS),
            "level": CONFIDENCE_LEVEL,
        }

        # ── Classification probabilities ──────────────
        X_clf = self.clf_scaler.transform(self.clf_imputer.transform(X_latest))
        all_probas: dict[str, np.ndarray] = {}
        for name, clf in self.clf_models.items():
            all_probas[name] = clf.predict_proba(X_clf)[0]

        # Average probabilities across classifiers
        clf_classes = list(self.clf_models.values())[0].classes_
        avg_proba = np.zeros(len(clf_classes))
        for proba in all_probas.values():
            aligned = np.zeros(len(clf_classes))
            for i, cls in enumerate(clf_classes):
                aligned[i] = proba[i] if i < len(proba) else 0.0
            avg_proba += aligned
        avg_proba /= len(all_probas)

        # Build probability distribution aligned to bin_labels
        prob_distribution = []
        for label in self.bin_labels:
            idx = np.where(clf_classes == label)[0]
            if len(idx) > 0:
                prob_distribution.append((label, round(avg_proba[idx[0]] * 100, 2)))
            else:
                prob_distribution.append((label, 0.0))

        return {
            "regression": reg_predictions,
            "probability_distribution": prob_distribution,
            "classifier_probas": {n: p.tolist() for n, p in all_probas.items()},
            "bin_labels": self.bin_labels,
            "clf_classes": clf_classes.tolist(),
        }

    def feature_importance(self, top_n: int = 20) -> pd.Series:
        """Feature importance from the primary regression model."""
        model = self.reg_models.get(PRIMARY_MODEL)
        if model is None or not hasattr(model, "feature_importances_"):
            print("[Predictor] Primary model has no feature_importances_.")
            return pd.Series(dtype=float)

        imp = pd.Series(
            model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False).head(top_n)

        print(f"\n{'─'*55}")
        print(f" Top {top_n} Feature Importances ({PRIMARY_MODEL})")
        print(f"{'─'*55}")
        for feat, val in imp.items():
            bar = "#" * int(val * 200)
            print(f"  {val:.4f}  {bar}  {feat}")
        print(f"{'─'*55}\n")
        return imp

    # ══════════════════════════════════════════════════
    #  PRIVATE
    # ══════════════════════════════════════════════════

    def _prepare_xy(self, df: pd.DataFrame):
        exclude = {"target_ppi_core_mom"}
        self.feature_names = [c for c in df.columns if c not in exclude]
        X = df[self.feature_names].values
        y = df["target_ppi_core_mom"].values
        return X, y

    def _bin_target(self, y_train: np.ndarray, y_test: np.ndarray):
        """
        Discretize continuous MoM% into bins of BIN_STEP width.
        E.g. with BIN_STEP=0.1 and BIN_RANGE=(-1.0, 1.0):
          bins centered on: -1.0, -0.9, -0.8, ..., 0.0, ..., +0.8, +0.9, +1.0
          plus overflow bins: <-1.0% and >+1.0%
        """
        low, high = BIN_RANGE
        step = BIN_STEP

        # Centers of each bin
        centers = np.round(np.arange(low, high + step / 2, step), 4)

        # Edges between bins: halfway between centers, plus -inf/+inf
        edges = []
        for i in range(len(centers) - 1):
            edges.append((centers[i] + centers[i + 1]) / 2)
        edges = np.array(edges)

        # Labels: overflow-low, center bins, overflow-high
        labels = [f"<{low:+.1f}%"]
        for c in centers:
            labels.append(f"{c:+.1f}%")
        labels.append(f">{high:+.1f}%")

        self.bin_edges = edges
        self.bin_labels = labels

        def digitize(y):
            result = []
            for val in y:
                if val < edges[0]:
                    if val < low - step / 2:
                        result.append(labels[0])   # <-1.0%
                    else:
                        result.append(labels[1])   # first center
                elif val >= edges[-1]:
                    if val >= high + step / 2:
                        result.append(labels[-1])   # >+1.0%
                    else:
                        result.append(labels[-2])   # last center
                else:
                    idx = np.searchsorted(edges, val, side="right")
                    result.append(labels[idx + 1])  # +1 for overflow-low offset
            return np.array(result)

        return digitize(y_train), digitize(y_test)

    @staticmethod
    def _build_reg_model(name: str, params: dict):
        """Factory for regression models."""
        if name == "gradient_boosting":
            return GradientBoostingRegressor(random_state=42, **params)
        elif name == "random_forest":
            return RandomForestRegressor(random_state=42, **params)
        elif name == "ridge":
            return Ridge(**params)
        else:
            raise ValueError(f"Unknown model: {name}")
    
    @staticmethod
    def _get_param_grid(name: str) -> dict:
        """Return hyperparameter grid for GridSearchCV."""
        if name == "gradient_boosting":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "min_samples_split": [5, 10],
            }
        elif name == "random_forest":
            return {
                "n_estimators": [200, 300, 400],
                "max_depth": [8, 10, 15, None],
                "min_samples_split": [5, 10, 15],
                "min_samples_leaf": [2, 5],
            }
        elif name == "ridge":
            return {
                "alpha": [0.1, 0.5, 1.0, 5.0, 10.0],
            }
        else:
            raise ValueError(f"Unknown model: {name}")
