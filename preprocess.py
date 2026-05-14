# =============================================================
# STEP 1: DATA PREPROCESSING
# File: preprocessing/preprocess.py
# =============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# ── Column names for NSL-KDD dataset ──────────────────────────
COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

# ── Attack type mapping (5 classes) ───────────────────────────
ATTACK_MAP = {
    'normal': 'Normal',
    # DoS attacks
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS',
    'worm':'DoS','mailbomb':'DoS',
    # Probe attacks
    'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe','satan':'Probe',
    'mscan':'Probe','saint':'Probe',
    # R2L attacks
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L',
    'phf':'R2L','spy':'R2L','warezclient':'R2L','warezmaster':'R2L',
    'sendmail':'R2L','named':'R2L','snmpgetattack':'R2L','snmpguess':'R2L',
    'xlock':'R2L','xsnoop':'R2L','httptunnel':'R2L',
    # U2R attacks
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R',
    'ps':'U2R','sqlattack':'U2R','xterm':'U2R'
}

def generate_sample_data(n_samples=5000):
    """
    Generate synthetic NSL-KDD-like data for demonstration.
    In real use, replace this with actual NSL-KDD CSV files.
    """
    np.random.seed(42)
    protocols  = ['tcp','udp','icmp']
    services   = ['http','ftp','smtp','ssh','dns','ftp_data','eco_i','other']
    flags      = ['SF','S0','REJ','RSTO','SH','S1','S2','S3','RSTOS0','OTH']
    labels     = ['normal','neptune','satan','ipsweep','smurf','portsweep',
                  'back','teardrop','buffer_overflow','guess_passwd']

    data = {
        'duration':          np.random.randint(0, 5000, n_samples),
        'protocol_type':     np.random.choice(protocols, n_samples),
        'service':           np.random.choice(services, n_samples),
        'flag':              np.random.choice(flags, n_samples),
        'src_bytes':         np.random.randint(0, 100000, n_samples),
        'dst_bytes':         np.random.randint(0, 100000, n_samples),
        'land':              np.random.randint(0, 2, n_samples),
        'wrong_fragment':    np.random.randint(0, 4, n_samples),
        'urgent':            np.random.randint(0, 3, n_samples),
        'hot':               np.random.randint(0, 30, n_samples),
        'num_failed_logins': np.random.randint(0, 5, n_samples),
        'logged_in':         np.random.randint(0, 2, n_samples),
        'num_compromised':   np.random.randint(0, 10, n_samples),
        'root_shell':        np.random.randint(0, 2, n_samples),
        'su_attempted':      np.random.randint(0, 2, n_samples),
        'num_root':          np.random.randint(0, 10, n_samples),
        'num_file_creations':np.random.randint(0, 10, n_samples),
        'num_shells':        np.random.randint(0, 5, n_samples),
        'num_access_files':  np.random.randint(0, 10, n_samples),
        'num_outbound_cmds': np.zeros(n_samples, dtype=int),
        'is_host_login':     np.random.randint(0, 2, n_samples),
        'is_guest_login':    np.random.randint(0, 2, n_samples),
        'count':             np.random.randint(0, 512, n_samples),
        'srv_count':         np.random.randint(0, 512, n_samples),
        'serror_rate':       np.random.uniform(0, 1, n_samples),
        'srv_serror_rate':   np.random.uniform(0, 1, n_samples),
        'rerror_rate':       np.random.uniform(0, 1, n_samples),
        'srv_rerror_rate':   np.random.uniform(0, 1, n_samples),
        'same_srv_rate':     np.random.uniform(0, 1, n_samples),
        'diff_srv_rate':     np.random.uniform(0, 1, n_samples),
        'srv_diff_host_rate':np.random.uniform(0, 1, n_samples),
        'dst_host_count':    np.random.randint(0, 256, n_samples),
        'dst_host_srv_count':np.random.randint(0, 256, n_samples),
        'dst_host_same_srv_rate':    np.random.uniform(0, 1, n_samples),
        'dst_host_diff_srv_rate':    np.random.uniform(0, 1, n_samples),
        'dst_host_same_src_port_rate':np.random.uniform(0, 1, n_samples),
        'dst_host_srv_diff_host_rate':np.random.uniform(0, 1, n_samples),
        'dst_host_serror_rate':      np.random.uniform(0, 1, n_samples),
        'dst_host_srv_serror_rate':  np.random.uniform(0, 1, n_samples),
        'dst_host_rerror_rate':      np.random.uniform(0, 1, n_samples),
        'dst_host_srv_rerror_rate':  np.random.uniform(0, 1, n_samples),
        'label':      np.random.choice(labels, n_samples,
                          p=[0.4,0.15,0.08,0.08,0.08,0.06,0.05,0.04,0.03,0.03]),
        'difficulty': np.random.randint(0, 21, n_samples)
    }
    return pd.DataFrame(data)


def load_data(train_path=None, test_path=None):
    """
    Load NSL-KDD data from CSV files.
    Falls back to synthetic data if files are not found.
    """
    if train_path and os.path.exists(train_path):
        print("Loading real NSL-KDD dataset...")
        train_df = pd.read_csv(train_path, names=COLUMNS)
        test_df  = pd.read_csv(test_path,  names=COLUMNS) \
                   if test_path and os.path.exists(test_path) else None
    else:
        print("NSL-KDD files not found — generating synthetic demo data...")
        train_df = generate_sample_data(5000)
        test_df  = generate_sample_data(1000)

    return train_df, test_df


def preprocess(df):
    """
    Full preprocessing pipeline:
      1. Map raw labels → 5-class attack categories
      2. Drop the difficulty column
      3. Label-encode categorical features
      4. MinMax-scale numerical features
    Returns: X (numpy array), y (numpy array), feature_names, label_encoder
    """
    df = df.copy()

    # ── 1. Map labels to 5 attack classes ─────────────────────
    df['label'] = df['label'].str.strip().str.lower()
    df['attack_type'] = df['label'].map(
        lambda x: ATTACK_MAP.get(x, 'Normal')
    )

    # ── 2. Drop unused columns ─────────────────────────────────
    df.drop(columns=['label', 'difficulty'], errors='ignore', inplace=True)

    # ── 3. Encode categorical columns ─────────────────────────
    cat_cols = ['protocol_type', 'service', 'flag']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # ── 4. Encode target label ─────────────────────────────────
    le_target = LabelEncoder()
    y = le_target.fit_transform(df['attack_type'])

    # ── 5. Separate features ───────────────────────────────────
    X = df.drop(columns=['attack_type'])
    feature_names = X.columns.tolist()

    # ── 6. Scale numerical features ───────────────────────────
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Preprocessing complete — samples: {X_scaled.shape[0]}, "
          f"features: {X_scaled.shape[1]}, classes: {le_target.classes_}")
    return X_scaled, y, feature_names, le_target, scaler


def split_data(X, y, test_size=0.2):
    """Split into train / test sets."""
    return train_test_split(X, y, test_size=test_size,
                            random_state=42, stratify=y)


def partition_for_nodes(X_train, y_train, n_nodes=3):
    """
    Partition training data across N simulated federated nodes.
    Each node gets a roughly equal, non-overlapping slice.
    """
    node_data = []
    indices   = np.random.permutation(len(X_train))
    splits    = np.array_split(indices, n_nodes)

    for i, idx in enumerate(splits):
        node_data.append({
            'node_id': i + 1,
            'X': X_train[idx],
            'y': y_train[idx],
            'n_samples': len(idx)
        })
        print(f"  Node {i+1}: {len(idx)} samples")

    return node_data
