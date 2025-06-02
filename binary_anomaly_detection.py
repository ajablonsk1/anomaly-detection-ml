import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report, roc_curve, auc, make_scorer)
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from process_data import (X_dtc_dos_hulk_train_scaled, X_dos_hulk_test_shared_scaled,
                          X_dtc_ftp_patator_train_scaled, X_ftp_patator_test_shared_scaled)


# ==========================================
# Experiment Manager Class
# ==========================================

class ExperimentManager:
    def __init__(self, attack_type):
        self.attack_type = attack_type
        self.base_path = f"data/classifier/{attack_type}"
        self.history_file = f"{self.base_path}/experiments/history.json"
        self.results_dir = f"{self.base_path}/results"
        self.plots_dir = f"{self.base_path}/plots"
        self.models_dir = f"{self.base_path}/models"

        # Create directories
        os.makedirs(f"{self.base_path}/experiments", exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # Load or create history
        self.history = self.load_history()

    def load_history(self):
        """Load experiment history from JSON file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {"experiments": [], "best_results": {}}

    def save_history(self):
        """Save experiment history to JSON file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_experiment_id(self):
        """Generate unique experiment ID"""
        return f"exp_{len(self.history['experiments']) + 1:03d}"

    def add_experiment(self, experiment_data):
        """Add new experiment to history"""
        experiment_data['experiment_id'] = self.get_experiment_id()
        experiment_data['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # Calculate improvements from previous experiments
        if self.history['experiments']:
            previous_best = self.get_best_experiment()
            experiment_data['improvements'] = self.calculate_improvements(
                experiment_data, previous_best
            )
        else:
            experiment_data['improvements'] = "First experiment - baseline"

        self.history['experiments'].append(experiment_data)

        # Update best results if necessary
        self.update_best_results(experiment_data)

        self.save_history()
        return experiment_data['experiment_id']

    def calculate_improvements(self, current, previous):
        """Calculate improvements compared to previous best"""
        improvements = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']

        for metric in metrics:
            if metric in current['metrics'] and metric in previous['metrics']:
                current_val = current['metrics'][metric]
                previous_val = previous['metrics'][metric]
                delta = current_val - previous_val
                improvements[metric] = {
                    'delta': round(delta, 4),
                    'percent_change': round((delta / previous_val) * 100, 2) if previous_val > 0 else 0,
                    'previous': previous_val,
                    'current': current_val
                }

        return improvements

    def get_best_experiment(self):
        """Get experiment with best F1 score"""
        if not self.history['experiments']:
            return None

        best_exp = max(self.history['experiments'],
                       key=lambda x: x['metrics'].get('f1_score', 0))
        return best_exp

    def update_best_results(self, experiment):
        """Update best results tracking"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']

        for metric in metrics:
            if metric in experiment['metrics']:
                current_value = experiment['metrics'][metric]
                if (metric not in self.history['best_results'] or
                        current_value > self.history['best_results'][metric]['value']):
                    self.history['best_results'][metric] = {
                        'value': current_value,
                        'experiment_id': experiment['experiment_id'],
                        'algorithm': experiment['algorithm'],
                        'parameters': experiment['parameters']
                    }

    def print_experiment_summary(self, experiment_id):
        """Print detailed summary of experiment"""
        exp = next((e for e in self.history['experiments']
                    if e['experiment_id'] == experiment_id), None)

        if not exp:
            print(f"Experiment {experiment_id} not found!")
            return

        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT SUMMARY: {experiment_id}")
        print(f"{'=' * 60}")
        print(f"Timestamp: {exp['timestamp']}")
        print(f"Algorithm: {exp['algorithm']}")
        print(f"Attack Type: {self.attack_type}")

        print(f"\nParameters:")
        for param, value in exp['parameters'].items():
            print(f"  - {param}: {value}")

        print(f"\nMetrics:")
        for metric, value in exp['metrics'].items():
            print(f"  - {metric}: {value:.4f}")

        if 'improvements' in exp and isinstance(exp['improvements'], dict):
            print(f"\nImprovements vs Previous Best:")
            for metric, improvement in exp['improvements'].items():
                delta = improvement['delta']
                percent = improvement['percent_change']
                symbol = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
                print(f"  {symbol} {metric}: {delta:+.4f} ({percent:+.2f}%)")

        print(f"\nConfusion Matrix:")
        cm = np.array(exp['confusion_matrix'])
        print(f"  TN: {cm[0, 0]:<6} FP: {cm[0, 1]:<6}")
        print(f"  FN: {cm[1, 0]:<6} TP: {cm[1, 1]:<6}")


# ==========================================
# Binary Classifier Training Functions
# ==========================================

def get_algorithm_configs():
    """Get predefined algorithm configurations for grid search"""
    return {
        'random_forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'param_grid': {
                'n_estimators': [100, 200],  # 4→2 values
                'max_depth': [10, 20],  # 5→2 values
                'min_samples_split': [2, 5],  # 3→2 values
                'min_samples_leaf': [1, 2]  # 3→2 values
            }
        },
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=2000),
            'param_grid': {
                'C': [0.1, 1, 10],  # 6→3 values
                'penalty': ['l2'],  # Only l2 (fastest)
                'solver': ['lbfgs']  # Fastest solver
            }
        }
    }


def train_classifier_with_gridsearch(X_train, y_train, X_test, y_test,
                                     algorithm='random_forest',
                                     custom_params=None,
                                     cv_folds=3):  # 5→3 folds
    """Train classifier with grid search optimization"""

    configs = get_algorithm_configs()

    if algorithm not in configs:
        raise ValueError(f"Algorithm {algorithm} not supported. Choose from: {list(configs.keys())}")

    config = configs[algorithm]
    model = config['model']
    param_grid = custom_params if custom_params else config['param_grid']

    print(f"🔍 Starting Grid Search for {algorithm}")
    print(f"   Parameter combinations to test: {np.prod([len(v) for v in param_grid.values()])}")

    # Handle elasticnet special case for logistic regression
    if algorithm == 'logistic_regression':
        param_grid = handle_elasticnet_params(param_grid)

    # Handle string labels for binary classification
    if len(np.unique(y_train)) == 2:
        # For binary classification with string labels, use the attack class as positive
        attack_labels = [label for label in np.unique(y_train) if label != 'BENIGN']
        if attack_labels:
            pos_label = attack_labels[0]
            f1_scorer = make_scorer(f1_score, pos_label=pos_label)
        else:
            f1_scorer = 'f1_weighted'
    else:
        f1_scorer = 'f1_weighted'

    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid,
        cv=cv_folds,
        scoring=f1_scorer,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Get best model and make predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'cv_score_mean': grid_search.best_score_,
        'cv_score_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    }

    # Additional metrics for binary classification
    if len(np.unique(y_test)) == 2:
        metrics.update({
            'precision_binary': precision_score(y_test, y_pred, pos_label=y_test.unique()[1]),
            'recall_binary': recall_score(y_test, y_pred, pos_label=y_test.unique()[1]),
            'f1_binary': f1_score(y_test, y_pred, pos_label=y_test.unique()[1])
        })

    results = {
        'model': best_model,
        'metrics': metrics,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'best_parameters': grid_search.best_params_,
        'grid_search_results': grid_search.cv_results_,
        'predictions': y_pred,
        'prediction_probabilities': y_pred_proba,
        'feature_importance': get_feature_importance(best_model, X_train.columns) if hasattr(X_train,
                                                                                             'columns') else None
    }

    return results


def handle_elasticnet_params(param_grid):
    """Handle elasticnet parameter constraints for logistic regression"""
    # ElasticNet only works with 'saga' solver
    if 'elasticnet' in param_grid.get('penalty', []):
        # Create separate parameter grids
        elasticnet_grid = []
        other_grid = []

        for penalty in param_grid['penalty']:
            if penalty == 'elasticnet':
                elasticnet_grid.append({
                    'penalty': ['elasticnet'],
                    'solver': ['saga'],
                    'C': param_grid['C'],
                    'l1_ratio': param_grid['l1_ratio']
                })
            else:
                for solver in param_grid['solver']:
                    if solver != 'saga' or penalty != 'elasticnet':
                        other_grid.append({
                            'penalty': [penalty],
                            'solver': [solver],
                            'C': param_grid['C']
                        })

        return elasticnet_grid + other_grid

    return param_grid


def get_feature_importance(model, feature_names):
    """Extract feature importance from trained model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importance)}
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        return {name: float(imp) for name, imp in zip(feature_names, importance)}
    return None


# ==========================================
# Visualization Functions
# ==========================================

def plot_confusion_matrix(cm, algorithm, attack_type, experiment_id, save_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['BENIGN', attack_type.upper()],
                yticklabels=['BENIGN', attack_type.upper()])
    plt.title(f'Confusion Matrix - {algorithm} ({experiment_id})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_{experiment_id}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(experiment_manager, experiment_id, save_dir):
    """Plot metrics comparison with previous experiments"""
    history = experiment_manager.history['experiments']

    if len(history) < 2:
        return  # Need at least 2 experiments to compare

    # Extract data for plotting
    exp_ids = [exp['experiment_id'] for exp in history]
    algorithms = [exp['algorithm'] for exp in history]
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        values = [exp['metrics'].get(metric, 0) for exp in history]
        colors = ['red' if exp_id == experiment_id else 'skyblue' for exp_id in exp_ids]

        bars = axes[idx].bar(range(len(exp_ids)), values, color=colors)
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[idx].set_xlabel('Experiment')
        axes[idx].set_ylabel(metric.replace("_", " ").title())
        axes[idx].set_xticks(range(len(exp_ids)))
        axes[idx].set_xticklabels([f'{exp_id}\n({alg})' for exp_id, alg in zip(exp_ids, algorithms)],
                                  rotation=45, ha='right')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Metrics Evolution - Current: {experiment_id}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_comparison_{experiment_id}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(feature_importance, algorithm, experiment_id, save_dir, top_n=15):
    """Plot feature importance"""
    if not feature_importance:
        return

    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]

    features, importance = zip(*top_features)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(features)), importance)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance - {algorithm} ({experiment_id})')
    plt.gca().invert_yaxis()

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, importance)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{imp:.3f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance_{experiment_id}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_timeline(experiment_manager, save_dir):
    """Plot performance timeline across all experiments"""
    history = experiment_manager.history['experiments']

    if len(history) < 2:
        return

    # Extract data
    timestamps = [datetime.strptime(exp['timestamp'], "%Y-%m-%d_%H:%M:%S") for exp in history]
    exp_ids = [exp['experiment_id'] for exp in history]
    f1_scores = [exp['metrics'].get('f1_score', 0) for exp in history]
    algorithms = [exp['algorithm'] for exp in history]

    plt.figure(figsize=(14, 8))

    # Color by algorithm
    unique_algorithms = list(set(algorithms))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_algorithms)))
    color_map = dict(zip(unique_algorithms, colors))

    for i, (timestamp, f1, algorithm, exp_id) in enumerate(zip(timestamps, f1_scores, algorithms, exp_ids)):
        plt.scatter(timestamp, f1, c=[color_map[algorithm]], s=100, alpha=0.7)
        plt.annotate(exp_id, (timestamp, f1), xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.plot(timestamps, f1_scores, 'k--', alpha=0.3, linewidth=1)

    # Create legend
    for algorithm, color in color_map.items():
        plt.scatter([], [], c=[color], label=algorithm, s=100)

    plt.xlabel('Experiment Time')
    plt.ylabel('F1 Score')
    plt.title(f'Performance Timeline - {experiment_manager.attack_type}')
    plt.legend(title='Algorithm')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# Main Training Functions for Each Attack Type
# ==========================================

def train_dos_hulk_classifier(algorithm='random_forest', custom_params=None, cv_folds=5):
    """Train classifier for DoS Hulk attack detection"""

    print(f"\n{'=' * 60}")
    print(f"DoS HULK ATTACK CLASSIFICATION - {algorithm.upper()}")
    print(f"{'=' * 60}")

    # Initialize experiment manager
    exp_manager = ExperimentManager('dos_hulk')

    # Prepare data
    X_train = X_dtc_dos_hulk_train_scaled.drop(columns=['Label'])
    y_train = X_dtc_dos_hulk_train_scaled['Label']

    X_test = X_dos_hulk_test_shared_scaled.drop(columns=['Label'])
    y_test = X_dos_hulk_test_shared_scaled['Label']

    print(f"📊 Dataset Info:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(X_train.columns)}")
    print(f"   Class distribution (train): {y_train.value_counts().to_dict()}")
    print(f"   Class distribution (test): {y_test.value_counts().to_dict()}")

    # Train classifier
    results = train_classifier_with_gridsearch(
        X_train, y_train, X_test, y_test,
        algorithm=algorithm,
        custom_params=custom_params,
        cv_folds=cv_folds
    )

    # Prepare experiment data
    experiment_data = {
        'algorithm': algorithm,
        'attack_type': 'dos_hulk',
        'parameters': results['best_parameters'],
        'metrics': results['metrics'],
        'confusion_matrix': results['confusion_matrix'],
        'dataset_info': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(X_train.columns),
            'train_distribution': y_train.value_counts().to_dict(),
            'test_distribution': y_test.value_counts().to_dict()
        }
    }

    # Add to experiment history
    experiment_id = exp_manager.add_experiment(experiment_data)

    # Save model
    model_path = f"{exp_manager.models_dir}/{experiment_id}_{algorithm}.joblib"
    joblib.dump(results['model'], model_path)

    # Generate visualizations
    cm = np.array(results['confusion_matrix'])
    plot_confusion_matrix(cm, algorithm, 'dos_hulk', experiment_id, exp_manager.plots_dir)
    plot_metrics_comparison(exp_manager, experiment_id, exp_manager.plots_dir)
    plot_feature_importance(results['feature_importance'], algorithm, experiment_id, exp_manager.plots_dir)
    plot_performance_timeline(exp_manager, exp_manager.plots_dir)

    # Save detailed results
    results_file = f"{exp_manager.results_dir}/detailed_results_{experiment_id}.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'experiment_id': experiment_id,
            'algorithm': algorithm,
            'parameters': results['best_parameters'],
            'metrics': results['metrics'],
            'confusion_matrix': results['confusion_matrix'],
            'feature_importance': results['feature_importance']
        }
        json.dump(json_results, f, indent=2)

    # Print experiment summary
    exp_manager.print_experiment_summary(experiment_id)

    print(f"\n💾 Files saved:")
    print(f"   Model: {model_path}")
    print(f"   Results: {results_file}")
    print(f"   Plots: {exp_manager.plots_dir}/")
    print(f"   History: {exp_manager.history_file}")

    return results, experiment_id, exp_manager


def train_ftp_patator_classifier(algorithm='random_forest', custom_params=None, cv_folds=5):
    """Train classifier for FTP Patator attack detection"""

    print(f"\n{'=' * 60}")
    print(f"FTP PATATOR ATTACK CLASSIFICATION - {algorithm.upper()}")
    print(f"{'=' * 60}")

    # Initialize experiment manager
    exp_manager = ExperimentManager('ftp_patator')

    # Prepare data
    X_train = X_dtc_ftp_patator_train_scaled.drop(columns=['Label'])
    y_train = X_dtc_ftp_patator_train_scaled['Label']

    X_test = X_ftp_patator_test_shared_scaled.drop(columns=['Label'])
    y_test = X_ftp_patator_test_shared_scaled['Label']

    print(f"📊 Dataset Info:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(X_train.columns)}")
    print(f"   Class distribution (train): {y_train.value_counts().to_dict()}")
    print(f"   Class distribution (test): {y_test.value_counts().to_dict()}")

    # Train classifier
    results = train_classifier_with_gridsearch(
        X_train, y_train, X_test, y_test,
        algorithm=algorithm,
        custom_params=custom_params,
        cv_folds=cv_folds
    )

    # Prepare experiment data
    experiment_data = {
        'algorithm': algorithm,
        'attack_type': 'ftp_patator',
        'parameters': results['best_parameters'],
        'metrics': results['metrics'],
        'confusion_matrix': results['confusion_matrix'],
        'dataset_info': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(X_train.columns),
            'train_distribution': y_train.value_counts().to_dict(),
            'test_distribution': y_test.value_counts().to_dict()
        }
    }

    # Add to experiment history
    experiment_id = exp_manager.add_experiment(experiment_data)

    # Save model
    model_path = f"{exp_manager.models_dir}/{experiment_id}_{algorithm}.joblib"
    joblib.dump(results['model'], model_path)

    # Generate visualizations
    cm = np.array(results['confusion_matrix'])
    plot_confusion_matrix(cm, algorithm, 'ftp_patator', experiment_id, exp_manager.plots_dir)
    plot_metrics_comparison(exp_manager, experiment_id, exp_manager.plots_dir)
    plot_feature_importance(results['feature_importance'], algorithm, experiment_id, exp_manager.plots_dir)
    plot_performance_timeline(exp_manager, exp_manager.plots_dir)

    # Save detailed results
    results_file = f"{exp_manager.results_dir}/detailed_results_{experiment_id}.json"
    with open(results_file, 'w') as f:
        json_results = {
            'experiment_id': experiment_id,
            'algorithm': algorithm,
            'parameters': results['best_parameters'],
            'metrics': results['metrics'],
            'confusion_matrix': results['confusion_matrix'],
            'feature_importance': results['feature_importance']
        }
        json.dump(json_results, f, indent=2)

    # Print experiment summary
    exp_manager.print_experiment_summary(experiment_id)

    print(f"\n💾 Files saved:")
    print(f"   Model: {model_path}")
    print(f"   Results: {results_file}")
    print(f"   Plots: {exp_manager.plots_dir}/")
    print(f"   History: {exp_manager.history_file}")

    return results, experiment_id, exp_manager


# ==========================================
# Utility Functions for Experiment Analysis
# ==========================================

def compare_algorithms(attack_type, algorithms=['random_forest', 'logistic_regression']):
    """Compare multiple algorithms for given attack type"""

    print(f"\n{'=' * 70}")
    print(f"ALGORITHM COMPARISON FOR {attack_type.upper()}")
    print(f"{'=' * 70}")

    results_summary = []

    for algorithm in algorithms:
        print(f"\n🚀 Training {algorithm}...")

        if attack_type == 'dos_hulk':
            results, exp_id, exp_manager = train_dos_hulk_classifier(algorithm=algorithm)
        elif attack_type == 'ftp_patator':
            results, exp_id, exp_manager = train_ftp_patator_classifier(algorithm=algorithm)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        results_summary.append({
            'algorithm': algorithm,
            'experiment_id': exp_id,
            'metrics': results['metrics'],
            'best_params': results['best_parameters']
        })

    # Print comparison summary
    print(f"\n{'=' * 70}")
    print(f"COMPARISON SUMMARY - {attack_type.upper()}")
    print(f"{'=' * 70}")

    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']

    for metric in metrics_to_compare:
        print(f"\n📊 {metric.upper()}:")
        for result in results_summary:
            value = result['metrics'].get(metric, 0)
            print(f"   {result['algorithm']:<20}: {value:.4f}")

    # Determine best algorithm
    best_algorithm = max(results_summary, key=lambda x: x['metrics'].get('f1_score', 0))
    print(f"\n🏆 BEST ALGORITHM: {best_algorithm['algorithm']} (F1: {best_algorithm['metrics']['f1_score']:.4f})")

    return results_summary


def load_experiment_history(attack_type):
    """Load and display experiment history for given attack type"""
    exp_manager = ExperimentManager(attack_type)

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT HISTORY - {attack_type.upper()}")
    print(f"{'=' * 60}")

    if not exp_manager.history['experiments']:
        print("No experiments found!")
        return

    print(f"Total experiments: {len(exp_manager.history['experiments'])}")

    # Show summary table
    print(f"\n{'ID':<8} {'Algorithm':<20} {'F1 Score':<10} {'Accuracy':<10} {'Timestamp':<20}")
    print("-" * 80)

    for exp in exp_manager.history['experiments']:
        exp_id = exp['experiment_id']
        algorithm = exp['algorithm']
        f1 = exp['metrics'].get('f1_score', 0)
        accuracy = exp['metrics'].get('accuracy', 0)
        timestamp = exp['timestamp']

        print(f"{exp_id:<8} {algorithm:<20} {f1:<10.4f} {accuracy:<10.4f} {timestamp:<20}")

    # Show best results
    if exp_manager.history['best_results']:
        print(f"\n🏆 BEST RESULTS:")
        for metric, best in exp_manager.history['best_results'].items():
            print(f"   {metric}: {best['value']:.4f} ({best['experiment_id']} - {best['algorithm']})")

    return exp_manager


# ==========================================
# Main Function for Running Experiments
# ==========================================

def main():
    """Main function to run classification experiments"""

    print("\n" + "=" * 80)
    print("NETWORK INTRUSION DETECTION - BINARY CLASSIFIER")
    print("Advanced Experiment Tracking & Parameter Optimization")
    print("=" * 80)

    # Menu for user interaction
    while True:
        print(f"\n🚀 EXPERIMENT MENU:")
        print(f"1. Train DoS Hulk Classifier (Random Forest)")
        print(f"2. Train DoS Hulk Classifier (Logistic Regression)")
        print(f"3. Train FTP Patator Classifier (Random Forest)")
        print(f"4. Train FTP Patator Classifier (Logistic Regression)")
        print(f"5. Compare All Algorithms - DoS Hulk")
        print(f"6. Compare All Algorithms - FTP Patator")
        print(f"7. Custom Parameter Experiment")
        print(f"8. View Experiment History (DoS Hulk)")
        print(f"9. View Experiment History (FTP Patator)")
        print(f"10. Advanced Analysis & Recommendations")
        print(f"0. Exit")

        choice = input(f"\nSelect option (0-10): ").strip()

        try:
            if choice == '1':
                train_dos_hulk_classifier(algorithm='random_forest')

            elif choice == '2':
                train_dos_hulk_classifier(algorithm='logistic_regression')

            elif choice == '3':
                train_ftp_patator_classifier(algorithm='random_forest')

            elif choice == '4':
                train_ftp_patator_classifier(algorithm='logistic_regression')

            elif choice == '5':
                compare_algorithms('dos_hulk')

            elif choice == '6':
                compare_algorithms('ftp_patator')

            elif choice == '7':
                run_custom_experiment()

            elif choice == '8':
                load_experiment_history('dos_hulk')

            elif choice == '9':
                load_experiment_history('ftp_patator')

            elif choice == '10':
                run_advanced_analysis()

            elif choice == '0':
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice! Please select 0-10.")

        except Exception as e:
            print(f"❌ Error occurred: {str(e)}")
            print("Please try again.")


def run_custom_experiment():
    """Run experiment with custom parameters"""

    print(f"\n{'=' * 60}")
    print(f"CUSTOM PARAMETER EXPERIMENT")
    print(f"{'=' * 60}")

    # Select attack type
    print(f"\nSelect attack type:")
    print(f"1. DoS Hulk")
    print(f"2. FTP Patator")

    attack_choice = input("Choice (1-2): ").strip()
    if attack_choice == '1':
        attack_type = 'dos_hulk'
    elif attack_choice == '2':
        attack_type = 'ftp_patator'
    else:
        print("❌ Invalid choice!")
        return

    # Select algorithm
    print(f"\nSelect algorithm:")
    print(f"1. Random Forest")
    print(f"2. Logistic Regression")

    algo_choice = input("Choice (1-2): ").strip()
    if algo_choice == '1':
        algorithm = 'random_forest'
        print(f"\n📝 Custom Random Forest Parameters:")
        print(f"Enter parameters (press Enter for default):")

        n_estimators = input("n_estimators [100, 200, 300]: ") or "100,200,300"
        max_depth = input("max_depth [10, 15, 20]: ") or "10,15,20"
        min_samples_split = input("min_samples_split [2, 5, 10]: ") or "2,5,10"
        min_samples_leaf = input("min_samples_leaf [1, 2, 4]: ") or "1,2,4"

        custom_params = {
            'n_estimators': [int(x.strip()) for x in n_estimators.split(',')],
            'max_depth': [int(x.strip()) if x.strip().lower() != 'none' else None for x in max_depth.split(',')],
            'min_samples_split': [int(x.strip()) for x in min_samples_split.split(',')],
            'min_samples_leaf': [int(x.strip()) for x in min_samples_leaf.split(',')]
        }

    elif algo_choice == '2':
        algorithm = 'logistic_regression'
        print(f"\n📝 Custom Logistic Regression Parameters:")
        print(f"Enter parameters (press Enter for default):")

        C_values = input("C [0.1, 1, 10]: ") or "0.1,1,10"
        penalty = input("penalty [l1, l2]: ") or "l1,l2"
        solver = input("solver [liblinear, saga]: ") or "liblinear,saga"

        custom_params = {
            'C': [float(x.strip()) for x in C_values.split(',')],
            'penalty': [x.strip() for x in penalty.split(',')],
            'solver': [x.strip() for x in solver.split(',')]
        }

        if 'elasticnet' in custom_params['penalty']:
            l1_ratio = input("l1_ratio [0.1, 0.5, 0.9]: ") or "0.1,0.5,0.9"
            custom_params['l1_ratio'] = [float(x.strip()) for x in l1_ratio.split(',')]
    else:
        print("❌ Invalid choice!")
        return

    print(f"\n🔧 Custom parameters: {custom_params}")

    # Run experiment
    if attack_type == 'dos_hulk':
        train_dos_hulk_classifier(algorithm=algorithm, custom_params=custom_params)
    else:
        train_ftp_patator_classifier(algorithm=algorithm, custom_params=custom_params)


def run_advanced_analysis():
    """Run advanced analysis and provide recommendations"""

    print(f"\n{'=' * 70}")
    print(f"ADVANCED ANALYSIS & RECOMMENDATIONS")
    print(f"{'=' * 70}")

    # Analyze both attack types
    for attack_type in ['dos_hulk', 'ftp_patator']:
        print(f"\n📊 ANALYZING {attack_type.upper()}:")
        exp_manager = ExperimentManager(attack_type)

        if not exp_manager.history['experiments']:
            print(f"   No experiments found for {attack_type}")
            continue

        # Performance trends
        experiments = exp_manager.history['experiments']
        f1_scores = [exp['metrics'].get('f1_score', 0) for exp in experiments]
        algorithms = [exp['algorithm'] for exp in experiments]

        # Algorithm performance analysis
        algo_performance = {}
        for exp in experiments:
            algo = exp['algorithm']
            f1 = exp['metrics'].get('f1_score', 0)
            if algo not in algo_performance:
                algo_performance[algo] = []
            algo_performance[algo].append(f1)

        print(f"\n   Algorithm Performance Summary:")
        for algo, scores in algo_performance.items():
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            print(f"   {algo:20}: Avg F1: {avg_score:.4f}, Best F1: {max_score:.4f} ({len(scores)} experiments)")

        # Best parameters analysis
        best_exp = exp_manager.get_best_experiment()
        if best_exp:
            print(f"\n   🏆 Best Configuration:")
            print(f"   Algorithm: {best_exp['algorithm']}")
            print(f"   F1 Score: {best_exp['metrics']['f1_score']:.4f}")
            print(f"   Parameters: {best_exp['parameters']}")

        # Recommendations
        print(f"\n   💡 Recommendations:")

        if len(experiments) < 5:
            print(f"   - Run more experiments to get better insights")

        # Parameter recommendations based on best results
        if best_exp['algorithm'] == 'random_forest':
            best_params = best_exp['parameters']
            if best_params.get('max_depth') == 20:
                print(f"   - Try higher max_depth values (25, 30)")
            if best_params.get('n_estimators') == 300:
                print(f"   - Try more estimators (400, 500)")

        if len(algo_performance) == 1:
            print(f"   - Try different algorithms for comparison")

        # Trend analysis
        if len(f1_scores) >= 3:
            recent_trend = np.mean(f1_scores[-3:]) - np.mean(f1_scores[:-3])
            if recent_trend > 0.01:
                print(f"   - Performance is improving! Continue current approach")
            elif recent_trend < -0.01:
                print(f"   - Performance declining. Try different parameter ranges")


def generate_experiment_report(attack_type):
    """Generate comprehensive experiment report"""

    exp_manager = ExperimentManager(attack_type)

    if not exp_manager.history['experiments']:
        print(f"No experiments found for {attack_type}")
        return

    report_file = f"{exp_manager.base_path}/experiment_report.txt"

    with open(report_file, 'w') as f:
        f.write(f"EXPERIMENT REPORT - {attack_type.upper()}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary statistics
        experiments = exp_manager.history['experiments']
        f.write(f"SUMMARY:\n")
        f.write(f"Total Experiments: {len(experiments)}\n")
        f.write(f"Algorithms Tested: {len(set(exp['algorithm'] for exp in experiments))}\n")

        # Best results
        f.write(f"\nBEST RESULTS:\n")
        for metric, best in exp_manager.history['best_results'].items():
            f.write(f"{metric}: {best['value']:.4f} ({best['experiment_id']} - {best['algorithm']})\n")

        # Detailed experiment log
        f.write(f"\nDETAILED EXPERIMENT LOG:\n")
        f.write("-" * 50 + "\n")

        for exp in experiments:
            f.write(f"\nExperiment: {exp['experiment_id']}\n")
            f.write(f"Algorithm: {exp['algorithm']}\n")
            f.write(f"Timestamp: {exp['timestamp']}\n")
            f.write(f"Parameters: {exp['parameters']}\n")
            f.write(f"Metrics:\n")
            for metric, value in exp['metrics'].items():
                f.write(f"  {metric}: {value:.4f}\n")

            if 'improvements' in exp and isinstance(exp['improvements'], dict):
                f.write(f"Improvements:\n")
                for metric, improvement in exp['improvements'].items():
                    delta = improvement['delta']
                    percent = improvement['percent_change']
                    f.write(f"  {metric}: {delta:+.4f} ({percent:+.2f}%)\n")
            f.write("-" * 30 + "\n")

    print(f"📄 Report saved: {report_file}")


# ==========================================
# Quick Start Functions
# ==========================================

def quick_start_dos_hulk():
    """Quick start function for DoS Hulk experiments"""
    print("🚀 Quick Start: DoS Hulk Classification")

    # Run basic experiments with both algorithms
    print("\n1️⃣ Testing Random Forest...")
    train_dos_hulk_classifier(algorithm='random_forest')

    print("\n2️⃣ Testing Logistic Regression...")
    train_dos_hulk_classifier(algorithm='logistic_regression')

    # Generate report
    generate_experiment_report('dos_hulk')


def quick_start_ftp_patator():
    """Quick start function for FTP Patator experiments"""
    print("🚀 Quick Start: FTP Patator Classification")

    # Run basic experiments with both algorithms
    print("\n1️⃣ Testing Random Forest...")
    train_ftp_patator_classifier(algorithm='random_forest')

    print("\n2️⃣ Testing Logistic Regression...")
    train_ftp_patator_classifier(algorithm='logistic_regression')

    # Generate report
    generate_experiment_report('ftp_patator')


def run_full_experiment_suite():
    """Run complete experiment suite for both attack types"""
    print("🎯 Running Full Experiment Suite")
    print("This will run comprehensive experiments for both attack types...")

    # DoS Hulk experiments
    print(f"\n{'=' * 60}")
    print("PHASE 1: DoS HULK EXPERIMENTS")
    print(f"{'=' * 60}")
    quick_start_dos_hulk()

    # FTP Patator experiments
    print(f"\n{'=' * 60}")
    print("PHASE 2: FTP PATATOR EXPERIMENTS")
    print(f"{'=' * 60}")
    quick_start_ftp_patator()

    # Final analysis
    print(f"\n{'=' * 60}")
    print("PHASE 3: FINAL ANALYSIS")
    print(f"{'=' * 60}")
    run_advanced_analysis()


if __name__ == "__main__":
    # You can run different modes:

    # Interactive mode
    # main()

    # quick_start_dos_hulk()
    # Co robi: Uczy się około 15 minut
    # Random Forest dla DoS Hulk
    # Logistic Regression dla DoS Hulk
    # Generuje raport dla DoS Hulk

    # quick_start_ftp_patator()
    # Co robi: Random Forest dla FTP Patator Uczy się około 15 minut
    # Logistic Regression dla FTP Patator
    # Generuje raport dla FTP Patator

     run_full_experiment_suite()
    # Co robi: PHASE 1: DoS Hulk (oba algorytmy) - 30 minut
    # PHASE 2: FTP Patator (oba algorytmy
    # PHASE 3: Advanced Analysis & porównania
