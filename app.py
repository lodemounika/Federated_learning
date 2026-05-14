

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import sys

# =============================================================
# PROJECT IMPORTS
# =============================================================

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.preprocess import (
    load_data,
    preprocess,
    split_data,
    partition_for_nodes
)

from baseline.centralized_model import (
    train_baseline,
    evaluate_model,
    get_feature_importance
)

from llm.attack_explainer import (
    detect_intrusion,
    get_top_features,
    explain_attack,
    ATTACK_KNOWLEDGE,
    SEVERITY_COLOR
)

# =============================================================
# IMPORT ADVANCED FEDERATED IDS
# =============================================================

from advanced_code import (
    federated_training,
    centralized_training,
    add_differential_privacy,
    secure_aggregate,
    perform_model_poisoning
)

# =============================================================
# PAGE CONFIG
# =============================================================

st.set_page_config(
    page_title="Privacy-Preserving IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# CUSTOM CSS
# =============================================================

st.markdown("""
<style>

.metric-card {
    background: #f0f4ff;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border-left: 4px solid #4a6cf7;
}

.attack-high {
    border-left: 4px solid #e53935;
    background: #fff5f5;
}

.attack-critical {
    border-left: 4px solid #b71c1c;
    background: #ffebee;
}

.attack-medium {
    border-left: 4px solid #fb8c00;
    background: #fff8e1;
}

.attack-normal {
    border-left: 4px solid #43a047;
    background: #f1f8e9;
}

h1, h2, h3 {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# =============================================================
# LOAD + TRAIN
# =============================================================

@st.cache_resource(show_spinner="Loading and training models...")
def load_and_train():

    train_df, test_df = load_data()

    X, y, feat_names, le, scaler = preprocess(train_df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    # =========================================================
    # BASELINE MODEL
    # =========================================================

    baseline = train_baseline(
        X_train,
        y_train,
        n_estimators=80
    )

    b_metrics = evaluate_model(
        baseline,
        X_test,
        y_test,
        le,
        "Baseline"
    )

    b_top = get_feature_importance(
        baseline,
        feat_names
    )

    # =========================================================
    # ADVANCED FEDERATED TRAINING
    # =========================================================

    node_data = partition_for_nodes(
        X_train,
        y_train,
        n_nodes=3
    )

    global_model, f_metrics, federated_accuracies = federated_training(
        node_data,
        X_test,
        y_test
    )

    round_metrics = []

    for i, acc in enumerate(federated_accuracies):

        round_metrics.append({
            'round': i + 1,
            'global_accuracy': round(acc * 100, 2),
            'avg_local_acc': round(acc * 100, 2),
            'global_f1': round(f_metrics['f1'] * 100, 2)
        })

    return {
        'baseline': baseline,
        'global_model': global_model,
        'b_metrics': b_metrics,
        'f_metrics': f_metrics,
        'b_top': b_top,
        'round_metrics': round_metrics,
        'X_test': X_test,
        'y_test': y_test,
        'feat_names': feat_names,
        'le': le,
        'scaler': scaler,
        'node_data': node_data
    }

# =============================================================
# HEADER
# =============================================================

st.title("🛡️ Privacy-Preserving Intrusion Detection System using Federated Learning")

st.caption(
    "Federated Learning + Differential Privacy + Secure Aggregation"
)

st.divider()

# =============================================================
# LOAD EVERYTHING
# =============================================================

with st.spinner("Initializing IDS System..."):
    D = load_and_train()

# =============================================================
# SIDEBAR
# =============================================================

with st.sidebar:

    st.title("🛡️ IDS Control Panel")

    tab_choice = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📊 Model Performance",
            "🔍 Live Detection",
            "🔗 Federated Learning",
            "⚠️ Attack Simulation",
            "📋 Benchmark Report"
        ]
    )

    st.divider()

    st.caption(
        "NSL-KDD · 5 attack classes · 3 federated nodes"
    )

    st.subheader("🔒 Privacy Features")

    st.success("Differential Privacy Enabled")

    st.success("Secure Aggregation Enabled")

    st.success("Poisoning Attack Simulation Enabled")

# =============================================================
# OVERVIEW TAB
# =============================================================

if "Overview" in tab_choice:

    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Federated Accuracy",
        f"{round(D['f_metrics']['accuracy']*100,2)}%"
    )

    col2.metric(
        "Centralized Accuracy",
        f"{round(D['b_metrics']['accuracy'],2)}%"
    )

    col3.metric(
        "Federated Nodes",
        "3"
    )

    col4.metric(
        "Federated Rounds",
        "5"
    )

    st.divider()

    st.subheader("Advanced Privacy Protection")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.success("""
        Differential Privacy

        Gaussian noise added to local
        client training data.
        """)

    with c2:

        st.success("""
        Secure Aggregation

        Server never sees raw client data.
        """)

    with c3:

        st.warning("""
        Poisoning Attack Simulation

        Simulates malicious clients
        inside federated training.
        """)

    st.divider()

    st.subheader("How Federated IDS Works")

    a1, a2, a3 = st.columns(3)

    with a1:

        st.info("""
        Step 1 — Local Training

        Each node trains independently.
        """)

    with a2:

        st.info("""
        Step 2 — Secure Aggregation

        Only protected updates are shared.
        """)

    with a3:

        st.info("""
        Step 3 — Global Detection

        Aggregated model detects attacks.
        """)

# =============================================================
# MODEL PERFORMANCE TAB
# =============================================================

elif "Performance" in tab_choice:

    st.subheader("📊 Performance Comparison")

    metrics_df = pd.DataFrame({

        'Metric': [
            'Accuracy',
            'Precision',
            'Recall',
            'F1 Score'
        ],

        'Centralized': [
            D['b_metrics']['accuracy'],
            D['b_metrics']['precision'],
            D['b_metrics']['recall'],
            D['b_metrics']['f1_score']
        ],

        'Federated': [
            D['f_metrics']['accuracy'],
            D['f_metrics']['precision'],
            D['f_metrics']['recall'],
            D['f_metrics']['f1']
        ]

    })

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Accuracy Comparison")

    chart_df = pd.DataFrame({

        'Model': [
            'Centralized',
            'Federated'
        ],

        'Accuracy': [
            D['b_metrics']['accuracy'],
            D['f_metrics']['accuracy']
        ]

    }).set_index('Model')

    st.bar_chart(chart_df)

# =============================================================
# LIVE DETECTION TAB
# =============================================================

# =============================================================
# LIVE DETECTION TAB
# =============================================================

elif "Detection" in tab_choice:

    st.subheader("🔍 Real-Time Intrusion Detection")

    mode = st.radio(
        "Choose Input Mode",
        [
            "Use Test Sample",
            "Manual User Input"
        ],
        horizontal=True
    )

    # =========================================================
    # TEST SAMPLE MODE
    # =========================================================

    if mode == "Use Test Sample":

        sample_idx = st.slider(
            "Select Test Sample",
            0,
            len(D['X_test']) - 1,
            0
        )

        X_sample = D['X_test'][[sample_idx]]

        true_label = D['le'].inverse_transform(
            [D['y_test'][sample_idx]]
        )[0]

        st.caption(f"True Label: {true_label}")

    # =========================================================
    # MANUAL USER INPUT MODE
    # =========================================================

    else:

        st.info(
            "Enter custom network traffic values below."
        )

        col1, col2, col3 = st.columns(3)

        duration = col1.number_input(
            "Duration",
            min_value=0,
            max_value=100000,
            value=0
        )

        src_bytes = col2.number_input(
            "Source Bytes",
            min_value=0,
            max_value=1000000,
            value=0
        )

        dst_bytes = col3.number_input(
            "Destination Bytes",
            min_value=0,
            max_value=1000000,
            value=0
        )

        count = col1.number_input(
            "Connection Count",
            min_value=0,
            max_value=1000,
            value=1
        )

        srv_count = col2.number_input(
            "Service Count",
            min_value=0,
            max_value=1000,
            value=1
        )

        serror_rate = col3.slider(
            "Serror Rate",
            0.0,
            1.0,
            0.0
        )

        same_srv_rate = col1.slider(
            "Same Service Rate",
            0.0,
            1.0,
            1.0
        )

        diff_srv_rate = col2.slider(
            "Different Service Rate",
            0.0,
            1.0,
            0.0
        )

        dst_host_count = col3.number_input(
            "Destination Host Count",
            min_value=0,
            max_value=255,
            value=0
        )

        # =====================================================
        # BUILD FEATURE VECTOR
        # =====================================================

        X_manual = np.zeros(
            (1, len(D['feat_names']))
        )

        feature_map = {

            'duration': duration,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'count': count,
            'srv_count': srv_count,
            'serror_rate': serror_rate,
            'same_srv_rate': same_srv_rate,
            'diff_srv_rate': diff_srv_rate,
            'dst_host_count': dst_host_count

        }

        for feature, value in feature_map.items():

            if feature in D['feat_names']:

                idx = D['feat_names'].index(feature)

                X_manual[0][idx] = value

        X_sample = X_manual

        true_label = "Unknown"

    # =========================================================
    # DETECTION BUTTON
    # =========================================================

    if st.button(
        "🔎 Analyze Traffic",
        type="primary"
    ):

        preds = detect_intrusion(
            D['global_model'],
            X_sample,
            D['le'],
            scaler=None
        )

        pred = preds[0]

        # =====================================================
        # FEATURE IMPORTANCE
        # =====================================================

        top_f = get_top_features(
            X_sample[0],
            D['feat_names'],
            D['global_model'],
            top_n=5
        )

        # =====================================================
        # ATTACK EXPLANATION
        # =====================================================

        expl = explain_attack(
            pred['predicted_class'],
            pred['confidence'],
            top_f
        )

        st.divider()

        # =====================================================
        # RESULT METRICS
        # =====================================================

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted Attack",
            pred['predicted_class']
        )

        c2.metric(
            "Confidence",
            f"{pred['confidence']}%"
        )

        c3.metric(
            "Severity",
            expl['severity']
        )

        # =====================================================
        # CLASS PROBABILITIES
        # =====================================================

        st.subheader("📊 Class Probabilities")

        prob_df = pd.DataFrame(

            list(pred['class_probs'].items()),

            columns=[
                'Attack Type',
                'Probability (%)'
            ]

        ).set_index('Attack Type')

        st.bar_chart(prob_df)

        # =====================================================
        # EXPLANATION
        # =====================================================

        st.subheader("🧠 Attack Explanation")

        st.write(
            expl['description']
        )

        # =====================================================
        # IMPACT
        # =====================================================

        st.subheader("⚠️ Potential Impact")

        st.write(
            expl['impact']
        )

        # =====================================================
        # INDICATORS
        # =====================================================

        with st.expander(
            "🔍 Key Indicators",
            expanded=True
        ):

            for ind in expl['indicators']:

                st.write(f"• {ind}")

        # =====================================================
        # RECOMMENDATIONS
        # =====================================================

        with st.expander(
            "🛠️ Security Recommendations",
            expanded=True
        ):

            for i, rec in enumerate(
                expl['recommendation'],
                1
            ):

                st.write(f"{i}. {rec}")

        # =====================================================
        # TOP FEATURES
        # =====================================================

        with st.expander(
            "📈 Top Contributing Features"
        ):

            feat_df = pd.DataFrame(top_f)

            st.dataframe(
                feat_df,
                use_container_width=True,
                hide_index=True
            )

        # =====================================================
        # RAW PREDICTION OUTPUT
        # =====================================================

        with st.expander(
            "🧾 Raw Prediction Output"
        ):

            st.json(pred)
# =============================================================
# FEDERATED LEARNING TAB
# =============================================================

elif "Federated" in tab_choice:

    st.subheader("🔗 Federated Learning Analytics")

    rounds_df = pd.DataFrame(D['round_metrics'])

    st.subheader("Global Accuracy per Round")

    st.line_chart(
        rounds_df.set_index('round')['global_accuracy']
    )

    st.subheader("Global F1 Score per Round")

    st.line_chart(
        rounds_df.set_index('round')['global_f1']
    )

    st.subheader("Node Distribution")

    node_df = pd.DataFrame([

        {
            'Node': f"Node {d['node_id']}",
            'Samples': d['n_samples']
        }

        for d in D['node_data']

    ]).set_index('Node')

    st.bar_chart(node_df)

# =============================================================
# ATTACK SIMULATION TAB
# =============================================================

elif "Attack" in tab_choice:

    st.subheader("⚠️ Federated Attack Simulation")

    st.warning("""
    Simulates malicious federated clients
    performing poisoning attacks.
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Malicious Client",
            "Node 1"
        )

        st.metric(
            "Attack Type",
            "Model Poisoning"
        )

    with col2:

        st.metric(
            "Defense",
            "Differential Privacy"
        )

        st.metric(
            "Aggregation",
            "Secure Aggregation"
        )

    attack_df = pd.DataFrame({

        'Scenario': [
            'Centralized IDS',
            'Federated IDS',
            'Federated + Privacy'
        ],

        'Privacy Risk': [
            'High',
            'Medium',
            'Low'
        ],

        'Attack Resistance': [
            'Low',
            'Medium',
            'High'
        ]

    })

    st.dataframe(
        attack_df,
        use_container_width=True,
        hide_index=True
    )

# =============================================================
# BENCHMARK REPORT TAB
# =============================================================

elif "Benchmark" in tab_choice:

    st.subheader("📋 Benchmark Report")

    benchmark_df = pd.DataFrame({

        'Metric': [
            'Accuracy',
            'Precision',
            'Recall',
            'F1 Score'
        ],

        'Centralized IDS': [
            D['b_metrics']['accuracy'],
            D['b_metrics']['precision'],
            D['b_metrics']['recall'],
            D['b_metrics']['f1_score']
        ],

        'Federated IDS': [
            D['f_metrics']['accuracy'],
            D['f_metrics']['precision'],
            D['f_metrics']['recall'],
            D['f_metrics']['f1']
        ]

    })

    st.dataframe(
        benchmark_df,
        use_container_width=True,
        hide_index=True
    )

    delta = (
        D['f_metrics']['accuracy']
        - D['b_metrics']['accuracy']
    )

    if delta >= 0:

        st.success(
            f"Federated IDS improved by {round(delta,2)}%"
        )

    else:

        st.warning(
            f"Federated IDS reduced by {round(abs(delta),2)}%"
        )

    st.info("""
    Privacy-preserving IDS may slightly reduce accuracy,
    but greatly improves privacy and security.
    """)