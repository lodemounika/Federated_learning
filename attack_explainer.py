# =============================================================
# STEP 5 & 6: INTRUSION DETECTION + ATTACK EXPLANATION
# File: llm/attack_explainer.py
# =============================================================

import numpy as np

# ─────────────────────────────────────────────────────────────
# STEP 5: INTRUSION DETECTION (INFERENCE)
# ─────────────────────────────────────────────────────────────

def detect_intrusion(model, X_input, label_encoder, scaler=None):
    """
    Run inference on new network traffic samples.

    Args:
        model         : trained global / baseline model
        X_input       : raw feature array (n_samples, n_features)
        label_encoder : fitted LabelEncoder for decoding predictions
        scaler        : fitted MinMaxScaler (optional, if input is raw)

    Returns:
        predictions   : list of dicts with class + confidence
    """
    # ── Scale input if scaler provided ────────────────────────
    if scaler is not None:
        X_input = scaler.transform(X_input)

    # ── Predict class and confidence ──────────────────────────
    y_pred   = model.predict(X_input)
    y_proba  = model.predict_proba(X_input)

    results = []
    for i in range(len(y_pred)):
        class_name  = label_encoder.inverse_transform([y_pred[i]])[0]
        confidence  = round(float(np.max(y_proba[i])) * 100, 2)
        class_probs = {
            label_encoder.inverse_transform([j])[0]: round(float(p) * 100, 2)
            for j, p in enumerate(y_proba[i])
        }
        results.append({
            'sample_index':  i,
            'predicted_class': class_name,
            'confidence':    confidence,
            'class_probs':   class_probs
        })

    return results


def get_top_features(X_sample, feature_names, model, top_n=5):
    """
    Get top contributing features for a single prediction.
    Uses Random Forest feature importances as a proxy for
    per-sample explanation (simpler than SHAP for demo purposes).
    """
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:top_n]
    top = []
    for idx in top_indices:
        top.append({
            'feature':    feature_names[idx],
            'value':      round(float(X_sample[idx]), 4),
            'importance': round(float(importances[idx]), 4)
        })
    return top


# ─────────────────────────────────────────────────────────────
# STEP 6: ATTACK EXPLANATION (Rule-based + LLM-ready)
# ─────────────────────────────────────────────────────────────

# Detailed knowledge base for each attack class
ATTACK_KNOWLEDGE = {
    'Normal': {
        'severity':    'None',
        'description': 'This is normal, legitimate network traffic. '
                       'No malicious activity detected.',
        'impact':      'No impact. System is operating normally.',
        'indicators':  ['Regular connection patterns',
                        'Expected byte counts',
                        'No suspicious flags'],
        'recommendation': [
            'Continue monitoring as usual.',
            'Maintain current security posture.',
            'Log for baseline reference.'
        ]
    },
    'DoS': {
        'severity':    'HIGH',
        'description': 'Denial of Service (DoS) attack detected. '
                       'The attacker is flooding the target with '
                       'excessive requests to exhaust resources and '
                       'make services unavailable to legitimate users.',
        'impact':      'Service unavailability, system slowdown, '
                       'network bandwidth exhaustion, financial loss '
                       'due to downtime.',
        'indicators':  ['Very high packet count',
                        'Extremely high src_bytes or dst_bytes',
                        'High serror_rate',
                        'Same source IP with rapid connections'],
        'recommendation': [
            'Immediately block the source IP address.',
            'Enable rate limiting on the affected service.',
            'Activate DDoS protection / CDN scrubbing.',
            'Notify the network security team.',
            'Scale up resources temporarily if cloud-based.'
        ]
    },
    'Probe': {
        'severity':    'MEDIUM',
        'description': 'Probe / Reconnaissance attack detected. '
                       'The attacker is scanning the network to '
                       'discover open ports, services, and potential '
                       'vulnerabilities before launching a deeper attack.',
        'impact':      'Information disclosure, vulnerability mapping, '
                       'preparation for a future targeted attack.',
        'indicators':  ['Port scan patterns (portsweep / nmap)',
                        'Many different destination ports',
                        'High diff_srv_rate',
                        'Low bytes per connection'],
        'recommendation': [
            'Block the scanning IP address immediately.',
            'Review and close unnecessary open ports.',
            'Enable IDS alerting for port scan signatures.',
            'Audit firewall rules.',
            'Check for any services running with default credentials.'
        ]
    },
    'R2L': {
        'severity':    'HIGH',
        'description': 'Remote to Local (R2L) attack detected. '
                       'An attacker from outside the network is '
                       'attempting to gain unauthorized local access '
                       'to a system, often by exploiting services '
                       'like FTP or by guessing credentials.',
        'impact':      'Unauthorized account access, data exfiltration, '
                       'installation of backdoors.',
        'indicators':  ['Multiple failed login attempts',
                        'Unusual FTP / SMTP traffic',
                        'High num_failed_logins',
                        'Connections to sensitive services'],
        'recommendation': [
            'Immediately reset compromised account credentials.',
            'Enable multi-factor authentication (MFA).',
            'Block the external IP address.',
            'Audit all remote-access logs.',
            'Enable account lockout after failed login attempts.'
        ]
    },
    'U2R': {
        'severity':    'CRITICAL',
        'description': 'User to Root (U2R) attack detected. '
                       'A local or authenticated user is attempting '
                       'to escalate privileges to root/admin level '
                       'by exploiting system vulnerabilities such as '
                       'buffer overflows or kernel exploits.',
        'impact':      'Full system compromise, root-level control, '
                       'ability to install malware, create backdoors, '
                       'and exfiltrate all data.',
        'indicators':  ['root_shell flag = 1',
                        'su_attempted flag = 1',
                        'Abnormal num_root activity',
                        'Buffer overflow patterns'],
        'recommendation': [
            'IMMEDIATE: Isolate the affected system from the network.',
            'Terminate the suspicious user session.',
            'Patch the exploited vulnerability immediately.',
            'Conduct full forensic analysis of the system.',
            'Rotate ALL credentials on the affected system.',
            'Review sudoers and root access permissions.',
            'File a security incident report.'
        ]
    }
}

SEVERITY_COLOR = {
    'None':     '🟢',
    'LOW':      '🟡',
    'MEDIUM':   '🟠',
    'HIGH':     '🔴',
    'CRITICAL': '🚨'
}


def explain_attack(predicted_class, confidence, top_features=None):
    """
    Generate a structured, human-readable explanation for
    a detected attack class.

    Args:
        predicted_class : string — one of Normal/DoS/Probe/R2L/U2R
        confidence      : float  — model confidence %
        top_features    : list of dicts from get_top_features()

    Returns:
        explanation dict with all fields
    """
    info     = ATTACK_KNOWLEDGE.get(predicted_class, ATTACK_KNOWLEDGE['Normal'])
    severity = info['severity']
    icon     = SEVERITY_COLOR.get(severity, '⚪')

    explanation = {
        'attack_type':     predicted_class,
        'severity':        severity,
        'severity_icon':   icon,
        'confidence':      confidence,
        'description':     info['description'],
        'impact':          info['impact'],
        'indicators':      info['indicators'],
        'recommendation':  info['recommendation'],
        'top_features':    top_features or [],
        'summary': (
            f"{icon} [{severity}] {predicted_class} attack detected "
            f"with {confidence}% confidence. "
            f"{info['description']}"
        )
    }
    return explanation


def build_llm_prompt(explanation, feature_names=None):
    """
    Build a ready-to-send prompt for an LLM (GPT/Claude/LLaMA).
    Paste this into an LLM API call for richer explanations.
    """
    feat_str = ""
    if explanation['top_features']:
        feat_str = "\n".join([
            f"  - {f['feature']}: {f['value']} (importance: {f['importance']})"
            for f in explanation['top_features']
        ])

    prompt = f"""You are a cybersecurity expert analyzing network intrusion detection results.

The IDS model detected the following:
- Attack Type   : {explanation['attack_type']}
- Severity      : {explanation['severity']}
- Confidence    : {explanation['confidence']}%

Top contributing network features:
{feat_str if feat_str else "  Not available"}

Please provide:
1. A plain-English explanation of what this attack is
2. How it works technically
3. The potential impact on the organization
4. Step-by-step mitigation recommendations
5. How to prevent this attack in the future

Write your response for both technical and non-technical stakeholders.
Keep it concise, actionable, and prioritize the most critical steps first.
"""
    return prompt


def print_explanation(explanation):
    """Pretty-print explanation to console."""
    print(f"\n{'='*60}")
    print(f"  INTRUSION DETECTION REPORT")
    print(f"{'='*60}")
    print(f"  Attack Type : {explanation['severity_icon']} "
          f"{explanation['attack_type']}")
    print(f"  Severity    : {explanation['severity']}")
    print(f"  Confidence  : {explanation['confidence']}%")
    print(f"\n  Description:")
    print(f"  {explanation['description']}")
    print(f"\n  Impact:")
    print(f"  {explanation['impact']}")
    print(f"\n  Key Indicators:")
    for ind in explanation['indicators']:
        print(f"    • {ind}")
    print(f"\n  Recommendations:")
    for i, rec in enumerate(explanation['recommendation'], 1):
        print(f"    {i}. {rec}")
    if explanation['top_features']:
        print(f"\n  Top Contributing Features:")
        for f in explanation['top_features']:
            print(f"    • {f['feature']}: {f['value']} "
                  f"(importance: {f['importance']})")
    print(f"{'='*60}\n")
