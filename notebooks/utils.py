import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from pyexpat import model
import mlflow



def calculate_woe(df, feature, target, smoothing=0.5):
    """
    Calculate Weight of Evidence for a categorical feature.
    
    Parameters:
        df: pandas DataFrame
        feature: name of categorical column
        target: name of binary target (0 = Good / Non-default, 1 = Bad / Default)
        smoothing: small value to avoid division by zero (default 0.5)
    """
    
    # Group by feature
    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bads']
    grouped['goods'] = grouped['total'] - grouped['bads']
    
    # Total goods and bads
    total_goods = grouped['goods'].sum()
    total_bads = grouped['bads'].sum()
    
    # Apply smoothing
    grouped['goods'] = grouped['goods'] + smoothing
    grouped['bads'] = grouped['bads'] + smoothing
    total_goods += smoothing * len(grouped)
    total_bads += smoothing * len(grouped)
    
    # Calculate distributions
    grouped['dist_goods'] = grouped['goods'] / total_goods
    grouped['dist_bads'] = grouped['bads'] / total_bads
    
    # Calculate WOE
    grouped['woe'] = np.log(grouped['dist_goods'] / grouped['dist_bads'])
    
    # Calculate Information Value (IV)
    grouped['iv'] = (grouped['dist_goods'] - grouped['dist_bads']) * grouped['woe']
    
    # Final summary
    woe_dict = grouped['woe'].to_dict()
    iv_total = grouped['iv'].sum()
    
    print(f"Total Information Value (IV) for '{feature}': {iv_total:.4f}")
    
    return grouped[['total', 'goods', 'bads', 'dist_goods', 'dist_bads', 'woe', 'iv']], woe_dict


def woe_multiple(df, features, target, smoothing=0.5):
    results = {}
    iv_values = {}
    
    for feat in features:
        table, mapping = calculate_woe(df, feat, target, smoothing)
        results[feat] = table
        iv_values[feat] = table['iv'].sum()
    
    # Sort by predictive power
    iv_df = pd.DataFrame.from_dict(iv_values, orient='index', columns=['IV']).sort_values('IV', ascending=False)
    print("Information Value Ranking:")
    print(iv_df)
    
    return results, iv_df


def category_to_woe(df, category_columns, woe_dict, delete_original=False):
    df = df.copy()
    for category_column in category_columns:
        df[f'{category_column}_woe'] = df[category_column].map(woe_dict[category_column]['woe'].to_dict())
    if delete_original:
        df = df.drop(columns=category_columns)
    return df


class ExperimentLogger:
    def __init__(self, log_file='credit_risk_experiments.csv'):
        self.log_file = log_file
        self.columns = [
            'timestamp', 'experiment_name', 'model_name', 'description',
            'auc', 'gini', 'ks', 'brier_score',
            'train_auc', 'val_auc', 'test_auc',
            'n_features', 'n_train_samples', 'default_rate',
            'hyperparameters', 'features_used', 'notes'
        ]
        
        if not pd.io.common.file_exists(log_file):
            pd.DataFrame(columns=self.columns).to_csv(log_file, index=False)
    
    def log_experiment(self, **kwargs):
        new_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs
        }
        pd.DataFrame([new_entry]).to_csv(self.log_file, mode='a', header=False, index=False)
        print(f"✅ Experiment logged: {kwargs.get('experiment_name')} | AUC: {kwargs.get('auc'):.4f}")


def evaluate_credit_model(model, 
                         X, 
                         y, 
                         dataset_name="Validation",
                         logger=None,
                         experiment_name=None,
                         model_name=None,
                         description="",
                         hyperparameters=None,
                         features_used=None,
                         notes=""):
    """
    Comprehensive evaluation with optional experiment logging
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    # Basic Metrics
    auc_score = roc_auc_score(y, y_pred_proba)
    gini = 2 * auc_score - 1
    brier = brier_score_loss(y, y_pred_proba)
    
    print(f"=== {dataset_name} Set Evaluation ===")
    print(f"AUC (ROC)       : {auc_score:.4f}")
    print(f"Gini Coefficient: {gini:.4f}")
    print(f"Brier Score     : {brier:.4f}")
    print("-" * 50)
    
    # KS Statistic (Very Important in Credit Risk)
    def ks_statistic(y_true, y_pred_proba):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        ks = max(tpr - fpr)
        return ks
    
    ks = ks_statistic(y, y_pred_proba)
    print(f"KS Statistic    : {ks:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend()
    plt.show()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name}')
    plt.show()
    
    # Log experiment if logger is provided
    if logger is not None and experiment_name is not None:
        log_data = {
            'experiment_name': experiment_name,
            'model_name': model_name or model.__class__.__name__,
            'description': description,
            'auc': round(auc_score, 4),
            'gini': round(gini, 4),
            'ks': round(ks, 4),
            'brier_score': round(brier, 4),
            'n_features': X.shape[1],
            'n_train_samples': X.shape[0],   # Note: this is actually test size here
            'default_rate': round(y.mean(), 4),
            'hyperparameters': json.dumps(hyperparameters) if hyperparameters else None,
            'features_used': json.dumps(features_used) if features_used else None,
            'notes': notes
        }
        logger.log_experiment(**log_data)
    
    # Optional: Show classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Return results dictionary
    return {
        'auc': auc_score,
        'gini': gini,
        'ks': ks,
        'brier': brier,
        'y_proba': y_pred_proba,
        'y_pred': y_pred
    }


def split_credit_data(df, target='Default', stratify=None, date_col=None, train_ratio=0.70, val_ratio=0.15):
    
    if date_col and date_col in df.columns:
        print("→ Using Time-based split")
        df = df.sort_values(date_col).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
    else:
        print("→ Using Stratified Random split")
        X = df.drop(target, axis=1)
        y = df[target]
        if stratify is None:
            stratify = y
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=1-train_ratio, random_state=42, stratify=stratify
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # Separate features and target
    X_train = train.drop(target, axis=1)
    y_train = train[target]
    X_val = val.drop(target, axis=1)
    y_val = val[target]
    X_test = test.drop(target, axis=1)
    y_test = test[target]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

class MLflowLogger:
    """
    MLflow Logger for Credit Risk Models with extended metrics
    """
    
    def __init__(self, experiment_name="Credit_Risk_PD_Modeling"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def fit(self, 
            model, 
            X_train, y_train, 
            X_val, y_val,
            X_test=None, y_test=None,
            run_name=None,
            model_name=None,
            description="",
            **fit_params):
        
        if run_name is None:
            model_type = model_name or self._get_model_name(model)
            run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"

        with mlflow.start_run(run_name=run_name):
            
            mlflow.log_param("model_name", model_name or self._get_model_name(model))
            mlflow.set_tag("mlflow.note.content", description)   # ← This is the key line
            mlflow.log_param("description", description)

            if isinstance(model, Pipeline):
                self._log_pipeline_info(model)

            print(f"Training {model_name or self._get_model_name(model)}...")
            
            if isinstance(model, Pipeline):
                step_name = model.steps[-1][0]   # Get the name of the last step automatically
                final_fit_params = {f'{step_name}__{k}': v for k, v in fit_params.items()}

                model.fit(X_train, y_train, **final_fit_params)
            else:
                model.fit(X_train, y_train, **fit_params)

            # Log all sets
            self._log_set(model, X_train, y_train, "train")
            self._log_set(model, X_val, y_val, "val")
            
            if X_test is not None and y_test is not None:
                self._log_set(model, X_test, y_test, "test")

            self.log_feature_importance(model, X_train, prefix="train")

            self._log_model(model, model_name)

            print(f"✅ Run completed → {run_name}\n")
            return model

    # ====================== METRICS LOGGING ======================
    
    def _log_set(self, model, X, y, prefix="val"):
        """Log comprehensive metrics for any dataset"""
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Calculate all metrics
        auc = roc_auc_score(y, y_proba)
        gini = 2 * auc - 1
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='binary')
        precision = precision_score(y, y_pred, average='binary', zero_division=0)
        recall = recall_score(y, y_pred, average='binary', zero_division=0)

        # Log all metrics with prefix
        mlflow.log_metrics({
            f"{prefix}_auc": round(auc, 4),
            f"{prefix}_gini": round(gini, 4),
            f"{prefix}_accuracy": round(acc, 4),
            f"{prefix}_f1": round(f1, 4),
            f"{prefix}_precision": round(precision, 4),
            f"{prefix}_recall": round(recall, 4),
            f"{prefix}_default_rate": round(y.mean(), 4),
        })

        # Log plots
        self._log_roc_curve(y, y_proba, prefix)
        self._log_confusion_matrix(y, y_pred, prefix)

        print(f"   {prefix.capitalize()}: AUC={auc:.4f} | Acc={acc:.4f} | "
              f"F1={f1:.4f} | Prec={precision:.4f} | Rec={recall:.4f}")

    # ====================== PLOTS ======================
    
    def _log_roc_curve(self, y_true, y_proba, prefix="val"):
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title(f'ROC Curve - {prefix.capitalize()}')
        plt.legend()
        mlflow.log_figure(plt.gcf(), f"plots/{prefix}_roc_curve.png")
        plt.close()

    def _log_confusion_matrix(self, y_true, y_pred, prefix="val"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {prefix.capitalize()}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        mlflow.log_figure(plt.gcf(), f"plots/{prefix}_confusion_matrix.png")
        plt.close()

    # ====================== FEATURE IMPORTANCE ======================
    
    def log_feature_importance(self, model, X_train, prefix="train"):
        estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model

        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
        elif hasattr(estimator, 'coef_'):
            coef = estimator.coef_[0]
            df = pd.DataFrame({
                'feature': X_train.columns, 
                'importance': np.abs(coef),
                'coefficient': coef
            })
        else:
            return None

        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        mlflow.log_table(df, f"features/{prefix}_feature_importance.json")
        
        df.to_csv(f"{prefix}_feature_importance.csv", index=False)
        mlflow.log_artifact(f"{prefix}_feature_importance.csv", artifact_path="features")

        print(f"✅ Feature importance logged for {prefix} set")
        return df

    # ====================== HELPERS ======================
    
    def _get_model_name(self, model):
        if isinstance(model, Pipeline):
            return model.steps[-1][1].__class__.__name__
        return model.__class__.__name__

    def _log_pipeline_info(self, pipeline):
        for name, step in pipeline.steps:
            mlflow.log_param(f"pipeline_step_{name}", step.__class__.__name__)

    def _log_model(self, model, model_name):
        try:
            if "lightgbm" in str(type(model)).lower():
                mlflow.lightgbm.log_model(model, name=model_name)
            else:
                mlflow.sklearn.log_model(model, name=model_name)
        except:
            print("Warning: Could not log model.")


def plot_calibration(y_true, y_proba, n_bins=20):
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Default Rate')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.show()