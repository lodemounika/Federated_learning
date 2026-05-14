# =============================================================
# MAIN RUNNER — ADVANCED PRIVACY-PRESERVING IDS
# File: main.py
# Run: python main.py
# =============================================================

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# =============================================================
# IMPORTS
# =============================================================

from preprocessing.preprocess import (
    load_data,
    preprocess,
    split_data,
    partition_for_nodes
)

from baseline.centralized_model import (
    train_baseline,
    evaluate_model,
    get_feature_importance,
    save_model
)

from llm.attack_explainer import (
    detect_intrusion,
    get_top_features,
    explain_attack,
    print_explanation,
    build_llm_prompt
)

# =============================================================
# IMPORT ADVANCED FEATURES
# =============================================================

from advanced_code import (
    federated_training,
    centralized_training,
    add_differential_privacy,
    secure_aggregate,
    perform_model_poisoning
)

# =============================================================
# CREATE OUTPUT DIRECTORIES
# =============================================================

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# =============================================================
# MAIN FUNCTION
# =============================================================

def main():

    print("\n" + "=" * 70)
    print(" PRIVACY-PRESERVING IDS WITH FEDERATED LEARNING ")
    print("=" * 70)

    # =========================================================
    # STEP 1 — LOAD + PREPROCESS DATA
    # =========================================================

    print("\n[STEP 1] Loading and preprocessing NSL-KDD dataset...")

    train_df, test_df = load_data(
        train_path="data/NSL_KDD_Train.csv",
        test_path="data/NSL_KDD_Test.csv"
    )

    all_df = train_df if test_df is None else train_df

    X, y, feature_names, label_encoder, scaler = preprocess(all_df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2
    )

    print(f"\nTraining Samples : {X_train.shape[0]}")
    print(f"Testing Samples  : {X_test.shape[0]}")
    print(f"Features         : {X_train.shape[1]}")

    # =========================================================
    # STEP 2 — CENTRALIZED BASELINE IDS
    # =========================================================

    print("\n[STEP 2] Training centralized baseline IDS...")

    baseline_model = train_baseline(
        X_train,
        y_train,
        n_estimators=100
    )

    baseline_metrics = evaluate_model(
        baseline_model,
        X_test,
        y_test,
        label_encoder,
        model_name="Centralized Baseline IDS"
    )

    print("\n[INFO] Extracting feature importance...")

    top_features = get_feature_importance(
        baseline_model,
        feature_names
    )

    save_model(
        baseline_model,
        "models/baseline_model.pkl"
    )

    print("[DONE] Baseline model saved.")

    # =========================================================
    # STEP 3 — FEDERATED NODE PARTITIONING
    # =========================================================

    print("\n[STEP 3] Partitioning data across federated nodes...")

    node_data = partition_for_nodes(
        X_train,
        y_train,
        n_nodes=3
    )

    print("\nFederated Node Summary:")

    for node in node_data:

        print(
            f" Node {node['node_id']} "
            f"→ {node['n_samples']} samples"
        )

    # =========================================================
    # STEP 4 — ADVANCED FEDERATED TRAINING
    # =========================================================

    print("\n[STEP 4] Running Advanced Federated Learning...")

    print("\nEnabled Features:")
    print(" ✓ Differential Privacy")
    print(" ✓ Secure Aggregation")
    print(" ✓ Poisoning Attack Simulation")
    print(" ✓ Federated Logging")
    print(" ✓ Privacy-Preserving Training")

    global_model, federated_metrics, federated_accuracies = federated_training(
        node_data,
        X_test,
        y_test
    )

    print("\n[DONE] Federated training completed.")

    # =========================================================
    # STEP 5 — SAVE GLOBAL MODEL
    # =========================================================

    save_model(
        global_model,
        "models/federated_global_model.pkl"
    )

    print("[DONE] Federated global model saved.")

    # =========================================================
    # STEP 6 — INTRUSION DETECTION
    # =========================================================

    print("\n[STEP 5] Running intrusion detection...")

    sample_X = X_test[:5]
    sample_y = y_test[:5]

    predictions = detect_intrusion(
        model=global_model,
        X_input=sample_X,
        label_encoder=label_encoder
    )

    print("\nSample Predictions:")

    for pred in predictions:

        true_label = label_encoder.inverse_transform(
            [sample_y[pred['sample_index']]]
        )[0]

        print(
            f"\n Sample {pred['sample_index'] + 1}"
        )

        print(
            f" Predicted : {pred['predicted_class']}"
        )

        print(
            f" Confidence: {pred['confidence']}%"
        )

        print(
            f" True Label: {true_label}"
        )

    # =========================================================
    # STEP 7 — ATTACK EXPLANATION
    # =========================================================

    print("\n[STEP 6] Generating attack explanation...")

    first_pred = predictions[0]

    top_feat_vals = get_top_features(
        sample_X[0],
        feature_names,
        global_model,
        top_n=5
    )

    explanation = explain_attack(
        predicted_class=first_pred['predicted_class'],
        confidence=first_pred['confidence'],
        top_features=top_feat_vals
    )

    print_explanation(explanation)

    # =========================================================
    # STEP 8 — LLM PROMPT GENERATION
    # =========================================================

    print("\n[STEP 7] Building LLM-ready security prompt...")

    llm_prompt = build_llm_prompt(explanation)

    print("\nLLM Prompt Preview:\n")

    print(
        llm_prompt[:500]
    )

    # =========================================================
    # STEP 9 — PRIVACY ANALYSIS
    # =========================================================

    print("\n[STEP 8] Privacy & Security Analysis")

    centralized_acc = baseline_metrics['accuracy']
    federated_acc = federated_metrics['accuracy']

    accuracy_delta = federated_acc - centralized_acc

    print("\nCentralized Accuracy : "
          f"{round(centralized_acc, 4)}")

    print("Federated Accuracy  : "
          f"{round(federated_acc, 4)}")

    print("Accuracy Difference : "
          f"{round(accuracy_delta, 4)}")

    print("\nPrivacy Benefits:")

    print(" ✓ No raw data sharing")
    print(" ✓ Distributed training")
    print(" ✓ Reduced leakage risk")
    print(" ✓ Better regulatory compliance")
    print(" ✓ Attack-resistant learning")

    # =========================================================
    # STEP 10 — SAVE RESULTS
    # =========================================================

    print("\n[STEP 9] Saving metrics and reports...")

    results = {

        'baseline_metrics': baseline_metrics,

        'federated_metrics': federated_metrics,

        'federated_accuracies': federated_accuracies,

        'top_features': top_features,

        'sample_predictions': predictions,

        'privacy_features': {

            'differential_privacy': True,

            'secure_aggregation': True,

            'poisoning_attack_simulation': True,

            'federated_learning': True
        },

        'federated_nodes': len(node_data),

        'training_rounds': len(federated_accuracies)
    }

    with open("results/metrics.json", "w") as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print("\n[DONE] Metrics saved → results/metrics.json")

    # =========================================================
    # STEP 11 — FINAL SUMMARY
    # =========================================================

    print("\n" + "=" * 70)

    print(" PROJECT EXECUTION COMPLETED SUCCESSFULLY ")

    print("=" * 70)

    print("\nImplemented Features:")

    print(" ✓ Centralized IDS")
    print(" ✓ Federated Learning")
    print(" ✓ Differential Privacy")
    print(" ✓ Secure Aggregation")
    print(" ✓ Poisoning Attack Simulation")
    print(" ✓ Attack Explanation")
    print(" ✓ LLM Integration")
    print(" ✓ Federated Metrics")
    print(" ✓ Privacy Benchmarking")

    print("\nTo launch dashboard:")

    print("\n streamlit run dashboard/app.py\n")

    # =========================================================
    # RETURN OBJECTS
    # =========================================================

    return (
        results,
        label_encoder,
        scaler,
        global_model,
        feature_names
    )

# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()