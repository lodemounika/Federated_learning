# ============================================================
# ADVANCED PRIVACY-PRESERVING FEDERATED IDS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import logging
import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# IMPORT YOUR EXISTING PREPROCESSING MODULE
# ============================================================

from preprocessing.preprocess import (
    load_data,
    preprocess,
    split_data,
    partition_for_nodes
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLIENTS = 3
FEDERATED_ROUNDS = 5

# Differential Privacy Noise
NOISE_SCALE = 0.05

# Attack Simulation
ATTACK_ENABLED = True
ATTACK_CLIENT = 0

# ============================================================
# DATA LOADING + PREPROCESSING
# ============================================================

def load_and_preprocess_data():

    print("\nLoading and preprocessing data...")

    # Uses your existing preprocessing pipeline
    train_df, test_df = load_data()

    X, y, feature_names, label_encoder, scaler = preprocess(train_df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    return X_train, X_test, y_train, y_test


# ============================================================
# CENTRALIZED IDS
# ============================================================

def centralized_training(X_train, X_test, y_train, y_test):

    print("\nTraining Centralized IDS Model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = evaluate_model(
        y_test,
        predictions,
        "Centralized IDS"
    )

    return model, metrics


# ============================================================
# DIFFERENTIAL PRIVACY
# ============================================================

def add_differential_privacy(X, noise_scale=0.01):

    """
    Add Gaussian noise to training features.

    This simulates Differential Privacy by
    perturbing client-side training data.
    """

    noise = np.random.normal(
        loc=0,
        scale=noise_scale,
        size=X.shape
    )

    X_noisy = X + noise

    return X_noisy



# ============================================================
# MODEL POISONING ATTACK
# ============================================================

def perform_model_poisoning(model):

    """
    Simulate malicious client poisoning attack.

    Instead of modifying protected tree internals,
    we randomly shuffle estimators to simulate
    corrupted model updates.
    """

    poisoned_model = copy.deepcopy(model)

    np.random.shuffle(poisoned_model.estimators_)

    return poisoned_model



# ============================================================
# SECURE AGGREGATION
# ============================================================

def secure_aggregate(client_models):

    """
    Simulate secure aggregation.
    """

    print("\nPerforming Secure Aggregation...")

    aggregated_model = copy.deepcopy(client_models[0])

    all_estimators = []

    for model in client_models:
        all_estimators.extend(model.estimators_)

    aggregated_model.estimators_ = all_estimators

    return aggregated_model


# ============================================================
# CLIENT TRAINING
# ============================================================

def train_client(client_id, X, y):

    """
    Train local federated client model
    with Differential Privacy protection.
    """

    logging.info(f"Training Client {client_id + 1}")

    # --------------------------------------------------------
    # Apply Differential Privacy
    # --------------------------------------------------------

    X_private = add_differential_privacy(
        X,
        noise_scale=NOISE_SCALE
    )

    # --------------------------------------------------------
    # Train Local Model
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=20,
        random_state=42
    )

    model.fit(X_private, y)

    # --------------------------------------------------------
    # Simulate Model Poisoning Attack
    # --------------------------------------------------------

    if ATTACK_ENABLED and client_id == ATTACK_CLIENT:

        print(f"\nClient {client_id + 1} launching poisoning attack!")

        model = perform_model_poisoning(model)

    return model
# ============================================================
# FEDERATED LEARNING
# ============================================================

def federated_training(client_data, X_test, y_test):

    federated_accuracies = []

    global_model = None

    for round_num in range(FEDERATED_ROUNDS):

        print("\n" + "=" * 50)
        print(f"Federated Round {round_num + 1}")
        print("=" * 50)

        client_models = []

        # Local client training
        for client in client_data:

            client_model = train_client(
                client['node_id'] - 1,
                client['X'],
                client['y']
            )

            client_models.append(client_model)

        # Secure Aggregation
        global_model = secure_aggregate(client_models)

        predictions = global_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        federated_accuracies.append(accuracy)

        print(f"Round Accuracy: {accuracy:.4f}")

    final_metrics = evaluate_model(
        y_test,
        predictions,
        "Federated IDS"
    )

    return global_model, final_metrics, federated_accuracies


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(y_true, y_pred, title):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average='weighted'
    )

    recall = recall_score(
        y_true,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average='weighted'
    )

    print(f"\n{title}")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(
    centralized_metrics,
    federated_metrics,
    federated_accuracies
):

    metrics = ['accuracy', 'precision', 'recall', 'f1']

    centralized_values = [
        centralized_metrics[m]
        for m in metrics
    ]

    federated_values = [
        federated_metrics[m]
        for m in metrics
    ]

    # Comparison Graph
    plt.figure(figsize=(10, 5))

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(
        x - width/2,
        centralized_values,
        width,
        label='Centralized'
    )

    plt.bar(
        x + width/2,
        federated_values,
        width,
        label='Federated'
    )

    plt.xticks(x, metrics)

    plt.ylabel('Score')

    plt.title('Centralized vs Federated IDS')

    plt.legend()

    plt.savefig('comparison_metrics.png')

    # Federated Training Graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(federated_accuracies)+1),
        federated_accuracies,
        marker='o'
    )

    plt.xlabel('Federated Round')

    plt.ylabel('Accuracy')

    plt.title('Federated Training Performance')

    plt.savefig('federated_training.png')

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    # Load Data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Centralized IDS
    centralized_model, centralized_metrics = centralized_training(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Partition Data Across Nodes
    client_data = partition_for_nodes(
        X_train,
        y_train,
        n_nodes=NUM_CLIENTS
    )

    # Federated IDS
    federated_model, federated_metrics, federated_accuracies = federated_training(
        client_data,
        X_test,
        y_test
    )

    # Visualization
    plot_results(
        centralized_metrics,
        federated_metrics,
        federated_accuracies
    )

    print("\nProject Execution Completed Successfully!")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    main()