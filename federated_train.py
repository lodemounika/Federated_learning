# =============================================================
# STEP 3 & 4: FEDERATED LEARNING — NODE + SERVER (FedAvg)
# File: federated/federated_train.py
# =============================================================

import numpy as np
import copy
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# NODE: Local training on each simulated client
# ─────────────────────────────────────────────────────────────

class FederatedNode:
    """
    Represents one federated client node.
    Each node:
      1. Receives the global model
      2. Trains on its LOCAL data only
      3. Returns updated model weights (never raw data)
    """

    def __init__(self, node_id, X, y, n_estimators=50):
        self.node_id      = node_id
        self.X            = X          # local data — stays on this node
        self.y            = y
        self.n_estimators = n_estimators
        self.local_model  = None

    def train_local(self, global_model=None):
        """
        Train a Random Forest on local data.
        If a global model is provided, we warm-start from it.
        """
        print(f"  [Node {self.node_id}] Training on {len(self.X)} local samples...")

        self.local_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            random_state=self.node_id * 7,   # different seed per node
            n_jobs=-1,
            warm_start=False
        )
        self.local_model.fit(self.X, self.y)

        # Local accuracy (self-reported — for monitoring only)
        y_pred    = self.local_model.predict(self.X)
        local_acc = accuracy_score(self.y, y_pred)
        print(f"  [Node {self.node_id}] Local accuracy: {local_acc*100:.2f}%")
        return local_acc

    def get_weights(self):
        """
        Return model 'weights' = the list of decision trees.
        Only this — not the raw training data — is sent to the server.
        """
        if self.local_model is None:
            raise ValueError(f"Node {self.node_id} has not been trained yet.")
        return {
            'node_id':      self.node_id,
            'estimators':   self.local_model.estimators_,
            'n_classes':    self.local_model.n_classes_,
            'n_features':   self.local_model.n_features_in_,
            'classes':      self.local_model.classes_,
            'n_samples':    len(self.X)
        }


# ─────────────────────────────────────────────────────────────
# SERVER: FedAvg aggregation
# ─────────────────────────────────────────────────────────────

class FederatedServer:
    """
    Central aggregation server.
    Collects weight updates from all nodes and applies FedAvg
    to produce an updated global model — without ever seeing
    the nodes' raw data.

    FedAvg formula:
        w_global = Σ (n_k / n_total) * w_k
    For Random Forest we implement this by pooling trees
    proportionally to each node's sample count.
    """

    def __init__(self, n_estimators=100):
        self.n_estimators  = n_estimators
        self.global_model  = None
        self.round_metrics = []   # history: accuracy per round

    def aggregate(self, node_weights_list):
        """
        FedAvg aggregation for Random Forest:
        - Collect all trees from all nodes
        - Weight each node's contribution by its sample count
        - Sample trees proportionally to form the global model
        """
        print("\n  [Server] Aggregating weights via FedAvg...")

        total_samples = sum(w['n_samples'] for w in node_weights_list)
        all_trees     = []
        all_weights   = []   # sampling weights per tree

        for node_w in node_weights_list:
            contribution = node_w['n_samples'] / total_samples
            n_trees_from_node = max(
                1,
                int(contribution * self.n_estimators)
            )
            # Sample trees from this node (or take all if fewer available)
            node_trees  = node_w['estimators']
            selected    = np.random.choice(
                len(node_trees),
                min(n_trees_from_node, len(node_trees)),
                replace=False
            )
            for idx in selected:
                all_trees.append(node_trees[idx])
                all_weights.append(contribution)

            print(f"  [Server] Node {node_w['node_id']}: "
                  f"contribution={contribution:.2f}, "
                  f"trees selected={len(selected)}")

        # ── Build the global model from aggregated trees ───────
        base_info = node_weights_list[0]
        global_rf = RandomForestClassifier(
            n_estimators=len(all_trees),
            random_state=42
        )
        # Manually set internals — sklearn allows this after fitting
        global_rf.estimators_   = all_trees
        global_rf.n_features_in_= base_info['n_features']
        global_rf.n_classes_    = base_info['n_classes']
        global_rf.classes_      = base_info['classes']
        global_rf.n_outputs_    = 1

        self.global_model = global_rf
        print(f"  [Server] Global model ready — {len(all_trees)} trees total")
        return self.global_model

    def evaluate_global(self, X_test, y_test):
        """Evaluate the current global model on the test set."""
        if self.global_model is None:
            return 0.0, 0.0
        y_pred = self.global_model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred) * 100
        f1     = f1_score(y_test, y_pred, average='weighted',
                          zero_division=0) * 100
        return round(acc, 2), round(f1, 2)

    def save_global_model(self, path="models/global_federated_model.pkl"):
        joblib.dump(self.global_model, path)
        print(f"  [Server] Global model saved → {path}")


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def run_federated_training(node_data_list, X_test, y_test,
                           n_rounds=5, n_estimators=100):
    """
    Full federated training loop across T rounds.

    Args:
        node_data_list : output of partition_for_nodes()
        X_test, y_test : held-out test set
        n_rounds       : number of federated rounds
        n_estimators   : trees in the global model

    Returns:
        global_model, round_metrics (list of dicts)
    """
    print("\n" + "="*55)
    print("  FEDERATED LEARNING — STARTING TRAINING")
    print("="*55)

    # ── Initialise nodes ──────────────────────────────────────
    nodes  = [
        FederatedNode(
            node_id=d['node_id'],
            X=d['X'],
            y=d['y'],
            n_estimators=max(10, n_estimators // len(node_data_list))
        )
        for d in node_data_list
    ]
    server = FederatedServer(n_estimators=n_estimators)
    round_metrics = []

    for rnd in range(1, n_rounds + 1):
        print(f"\n{'─'*55}")
        print(f"  ROUND {rnd} / {n_rounds}")
        print(f"{'─'*55}")

        # ── Each node trains locally ───────────────────────────
        node_weights = []
        local_accs   = []
        for node in nodes:
            local_acc = node.train_local(global_model=server.global_model)
            local_accs.append(local_acc)
            node_weights.append(node.get_weights())

        # ── Server aggregates weights ──────────────────────────
        server.aggregate(node_weights)

        # ── Evaluate global model ──────────────────────────────
        global_acc, global_f1 = server.evaluate_global(X_test, y_test)
        avg_local_acc = np.mean(local_accs) * 100

        round_metrics.append({
            'round':          rnd,
            'global_accuracy':global_acc,
            'global_f1':      global_f1,
            'avg_local_acc':  round(avg_local_acc, 2)
        })

        print(f"\n  [Round {rnd}] Global Accuracy : {global_acc}%")
        print(f"  [Round {rnd}] Global F1-Score : {global_f1}%")
        print(f"  [Round {rnd}] Avg Local Acc   : {avg_local_acc:.2f}%")

    # ── Save final global model ────────────────────────────────
    server.save_global_model()

    print("\n" + "="*55)
    print("  FEDERATED TRAINING COMPLETE")
    print("="*55)

    return server.global_model, round_metrics
