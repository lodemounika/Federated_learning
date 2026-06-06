

# Federated Learning Based Privacy-Preserving Intrusion Detection System

## Project Overview

This project presents a **Privacy-Preserving Intrusion Detection System (IDS)** using **Federated Learning** and **Machine Learning** techniques to detect cyber attacks while protecting sensitive network data.

Traditional IDS systems require centralized data collection, which can lead to:

* Privacy leakage
* Security risks
* High communication overhead
* Single point of failure

To overcome these problems, this project uses **Federated Learning**, where multiple clients train models locally and only share model parameters instead of raw data.

The system also includes:

* Differential Privacy concepts
* Secure model aggregation
* Attack explanation module
* Interactive Streamlit dashboard
* Performance visualization

---

# Features

 Federated Learning-based IDS
 Privacy-Preserving Training
 Centralized vs Federated Comparison
 Attack Detection & Classification
 AI-Based Attack Explanation
 Interactive Streamlit Dashboard
 Performance Metrics Visualization
 Model Training Graphs
 NSL-KDD Dataset Support
 Machine Learning-Based Detection

---

# Technologies Used

| Technology         | Purpose               |
| ------------------ | --------------------- |
| Python             | Core Programming      |
| Scikit-Learn       | Machine Learning      |
| Pandas             | Data Processing       |
| NumPy              | Numerical Computation |
| Streamlit          | Dashboard UI          |
| Matplotlib         | Data Visualization    |
| Federated Learning | Distributed Training  |
| Pickle             | Model Saving          |

---

#  Project Structure

```bash
Federated_learning/
│
├── main.py
├── advanced_code.py
├── federated_train.py
├── centralized_model.py
├── preprocess.py
├── attack_explainer.py
├── app.py
│
├── baseline_model.pkl
├── federated_global_model.pkl
├── global_federated_model.pkl
│
├── metrics.json
├── comparison_metrics.png
├── federated_training.png
│
└── README.md
```

---

# Installation

## 1️ Clone the Repository

```bash
git clone https://github.com/lodemounika/Federated_learning.git
cd Federated_learning
```

---

## 2️ Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run Main System

```bash
python main.py
```

---

## Run Federated Training

```bash
python federated_train.py
```

---

## Run Centralized Model

```bash
python centralized_model.py
```

---

## Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

# Workflow

```text
NSL-KDD Dataset
        ↓
Data Preprocessing
        ↓
Multiple Client Nodes
        ↓
Local Model Training
        ↓
Federated Aggregation
        ↓
Global Model Generation
        ↓
Intrusion Detection
        ↓
Attack Classification
        ↓
Dashboard Visualization
```

---

# Privacy-Preserving Techniques

## Federated Learning

* Data remains on local devices
* Only model parameters are shared
* Reduces data exposure risks
* Improves distributed learning

---

## Secure Aggregation

* Combines local client models securely
* Prevents direct access to client data
* Enhances system privacy

---

# Performance Metrics

The project evaluates performance using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

# Dashboard Features

The Streamlit dashboard provides:

* Real-time predictions
* Attack analysis
* Federated vs centralized comparison
* Performance graphs
* Privacy visualization
* Interactive UI

---

# Supported Attack Types

The IDS supports detection of:

* DoS Attacks
* Probe Attacks
* R2L Attacks
* U2R Attacks
* Normal Network Traffic

# Dataset Used

## NSL-KDD Dataset
The project uses the **NSL-KDD** dataset for training and testing.
Dataset includes:
* Network traffic features
* Normal activities
* Multiple cyber attack categories

## Federated Training Graph
```bash
federated_training.png
```
## Comparison Metrics
bash
comparison_metrics.png

# Future Improvements

* Deep Learning Integration
* Real-Time Packet Monitoring
* Blockchain-based Aggregation
* Explainable AI (XAI)
* Cloud Deployment
* Edge AI Integration
* Hybrid IDS Models

---

#  Contributing

Contributions are welcome.

Steps:

1. Fork the repository
2. Create a new branch
3. Commit changes
4. Push code
5. Create Pull Request

#  License

This project is developed for educational and research purposes.

Free to use and modify for learning.

#  Author

### L. Mounika



