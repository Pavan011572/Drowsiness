import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class DrowsinessEnsembleTrainer:
    """
    Trains a Stacking Ensemble Classifier combining XGBoost, Random Forest, 
    and ExtraTrees/MLP base learners with a Logistic Regression meta-learner.
    """

    def __init__(self, data_csv_path, model_save_dir):
        self.data_csv_path = data_csv_path
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)

        self.scaler = StandardScaler()
        self.stacking_model = None

    def load_and_preprocess_data(self):
        print(f"Loading feature dataset from {self.data_csv_path}...")
        df = pd.read_csv(self.data_csv_path, on_bad_lines='skip')

        # Drop non-feature columns
        feature_cols = [col for col in df.columns if col not in ['target', 'filename']]
        
        # Coerce feature columns to numeric
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['target'] = pd.to_numeric(df['target'], errors='coerce')

        df = df.dropna(subset=['target'])
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

        X = df[feature_cols].values.astype(np.float32)
        y = df['target'].values.astype(int)

        print(f"Dataset shape: {X.shape}, Class distribution: Non-Drowsy (0)={np.sum(y==0)}, Drowsy (1)={np.sum(y==1)}")
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)




    def get_base_models(self):
        """Define Base Learners"""
        xgb_base = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

        rf_base = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )

        et_base = ExtraTreesClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        return {
            'XGBoost Classifier': xgb_base,
            'Random Forest Classifier': rf_base,
            'Extra Trees Classifier': et_base
        }

    def build_stacking_ensemble(self):
        """Define Stacking Ensemble with Base Learners and Meta Classifier"""
        base_dict = self.get_base_models()
        estimators = [
            ('xgboost', base_dict['XGBoost Classifier']),
            ('random_forest', base_dict['Random Forest Classifier']),
            ('extra_trees', base_dict['Extra Trees Classifier'])
        ]

        # Meta Learner
        meta_classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )

        # Stacking Ensemble
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_classifier,
            cv=5,
            n_jobs=-1,
            passthrough=False
        )

        return stacking_clf

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()

        # Fit Scaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # ----------------------------------------------------
        # 1. EVALUATE INDIVIDUAL BASE MODELS
        # ----------------------------------------------------
        print("\n" + "="*70)
        print("STEP 1: EVALUATING INDIVIDUAL BASE MODELS")
        print("="*70)

        base_models = self.get_base_models()
        individual_metrics = {}

        for name, model in base_models.items():
            print(f"\nTraining base model: {name}...")
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

            individual_metrics[name] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "roc_auc": float(auc)
            }

            print(f"--- {name} Metrics ---")
            print(f"  Accuracy:  {acc * 100:.2f}%")
            print(f"  Precision: {prec * 100:.2f}%")
            print(f"  Recall:    {rec * 100:.2f}%")
            print(f"  F1 Score:  {f1 * 100:.2f}%")
            print(f"  ROC-AUC:   {auc:.4f}")

        # ----------------------------------------------------
        # 2. EVALUATE COMBINED STACKING ENSEMBLE
        # ----------------------------------------------------
        print("\n" + "="*70)
        print("STEP 2: TRAINING COMBINED STACKING ENSEMBLE ARCHITECTURE")
        print("="*70)

        self.stacking_model = self.build_stacking_ensemble()
        self.stacking_model.fit(X_train_scaled, y_train)

        # Evaluate Stacking Ensemble on Test set
        y_pred = self.stacking_model.predict(X_test_scaled)
        y_proba = self.stacking_model.predict_proba(X_test_scaled)[:, 1]

        acc_ens = accuracy_score(y_test, y_pred)
        prec_ens = precision_score(y_test, y_pred)
        rec_ens = recall_score(y_test, y_pred)
        f1_ens = f1_score(y_test, y_pred)
        auc_ens = roc_auc_score(y_test, y_proba)
        cm_ens = confusion_matrix(y_test, y_pred)

        print("\n" + "="*70)
        print("COMBINED STACKING ENSEMBLE MODEL EVALUATION METRICS")
        print("="*70)
        print(f"  Accuracy:  {acc_ens * 100:.2f}%")
        print(f"  Precision: {prec_ens * 100:.2f}%")
        print(f"  Recall:    {rec_ens * 100:.2f}%")
        print(f"  F1 Score:  {f1_ens * 100:.2f}%")
        print(f"  ROC-AUC:   {auc_ens:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Non-Drowsy', 'Drowsy']))

        # ----------------------------------------------------
        # 3. PRINT COMPARATIVE SUMMARY TABLE
        # ----------------------------------------------------
        print("\n" + "="*85)
        print("MODEL PERFORMANCE COMPARISON SUMMARY (INDIVIDUAL BASE MODELS VS COMBINED ENSEMBLE)")
        print("="*85)
        header = f"{'Model Name':<30} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'ROC-AUC':<10}"
        print(header)
        print("-" * len(header))

        for name, m in individual_metrics.items():
            print(f"{name:<30} | {m['accuracy']*100:>8.2f}% | {m['precision']*100:>8.2f}% | {m['recall']*100:>8.2f}% | {m['f1_score']*100:>8.2f}% | {m['roc_auc']:>9.4f}")

        print("-" * len(header))
        print(f"{'COMBINED STACKING ENSEMBLE':<30} | {acc_ens*100:>8.2f}% | {prec_ens*100:>8.2f}% | {rec_ens*100:>8.2f}% | {f1_ens*100:>8.2f}% | {auc_ens:>9.4f}")
        print("="*85)

        combined_metrics = {
            "accuracy": float(acc_ens),
            "precision": float(prec_ens),
            "recall": float(rec_ens),
            "f1_score": float(f1_ens),
            "roc_auc": float(auc_ens),
            "confusion_matrix": cm_ens.tolist()
        }

        all_metrics_payload = {
            "individual_models": individual_metrics,
            "combined_ensemble": combined_metrics,
            # Top-level backward compatibility for existing API endpoints
            "accuracy": float(acc_ens),
            "precision": float(prec_ens),
            "recall": float(rec_ens),
            "f1_score": float(f1_ens),
            "roc_auc": float(auc_ens),
            "confusion_matrix": cm_ens.tolist()
        }

        # Save artifacts
        model_path = os.path.join(self.model_save_dir, "stacking_ensemble.pkl")
        scaler_path = os.path.join(self.model_save_dir, "scaler.pkl")
        metrics_path = os.path.join(self.model_save_dir, "model_metrics.json")

        joblib.dump(self.stacking_model, model_path)
        joblib.dump(self.scaler, scaler_path)

        with open(metrics_path, 'w') as f:
            json.dump(all_metrics_payload, f, indent=4)

        print(f"\nTrained ensemble model saved to: {model_path}")
        print(f"StandardScaler saved to: {scaler_path}")
        print(f"Full metrics report saved to: {metrics_path}")

        return all_metrics_payload

if __name__ == "__main__":
    csv_file = r"c:\Users\surya\Downloads\drow\data\extracted_features.csv"
    save_dir = r"c:\Users\surya\Downloads\drow\models"

    if not os.path.exists(csv_file):
        print(f"Error: Feature CSV file not found at {csv_file}. Please run feature_extraction.py first.")
    else:
        trainer = DrowsinessEnsembleTrainer(csv_file, save_dir)
        trainer.train_and_evaluate()
