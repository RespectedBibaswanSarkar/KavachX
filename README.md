# KavachX - AI-Driven Behavioral Authentication System

> Production-grade fraud detection for Indian digital banking channels
> Built with PyTorch RL, XGBoost, and ensemble learning

## 🎯 Overview

KavachX is a complete, production-ready behavioral authentication system designed to protect Indian digital banking channels (Internet Banking, Mobile Banking, UPI, etc.) from fraud through advanced ML/AI techniques.

**Key Metrics:**
- Detection recall: > 95%
- False positive rate: < 2%
- API latency p99: < 50ms
- AUROC: > 0.98

## 🏗️ Architecture

The system is built in 5 layers + simulation & API:

### Layer 1: Static Rule Engine (`rules.py`)
Hard-coded instant checks that fire before any ML inference:
- TOR/anonymiser detection
- Impossible geo-velocity (>850 km/h = physically impossible)
- IP reputation checks
- Brute-force pattern detection (>5 failures in 24h → block)
- Concurrent session anomaly (>3 simultaneous sessions)
- Device + Location + Time combination flagging

**Output:** `{block: bool, reasons: list[str], risk_score: 0-1}`

### Layer 2: Feature Extraction (`features.py`)
Extracts a **32-dimensional float32 state vector** from raw session telemetry:

```
[0:8]    Keystroke dynamics
  - dwell_mean, dwell_std (key hold time)
  - flight_mean, flight_std (inter-key intervals)
  - bigram_consistency (typing predictability)
  - error_rate, typing_speed_wpm, rhythm_score

[8:14]   Mouse/touch biometrics
  - speed_mean, speed_std
  - curvature, pressure, peak_speed
  - jerk_ratio (erratic movements)

[14:20]  Temporal & session patterns
  - hour_of_day, day_of_week
  - session_duration, login_attempts
  - time_since_last_login, geo_velocity

[20:26]  Network & device fingerprint
  - known_device, known_ip_range
  - vpn_detected, tor_exit_node, ip_reputation_bad
  - failed_logins_24h

[26:32]  Transaction & navigation
  - amount_inr, new_beneficiary, international
  - daily_tx_count, amount_exceeds_avg_5x
  - navigation_anomaly_score
```

**Features include per-user baseline normalization** (Z-score against 30-session history, clipped to [-3, 3])

### Layer 3: RL Agent (`rl_agent.py`)
**Dueling Double DQN** for adaptive authentication decisions:

- **State:** 32-D feature vector
- **Actions:** 
  - 0 = ALLOW (proceed normally)
  - 1 = STEP_UP (request OTP/biometric challenge)
  - 2 = BLOCK (terminate session)
  
- **Architecture:**
  ```
  Shared layers:
    Linear(32→128) + LayerNorm + ReLU
    Linear(128→128) + LayerNorm + ReLU
  
  Value stream:     Linear(128→64) + ReLU → Linear(64→1)
  Advantage stream: Linear(128→64) + ReLU → Linear(64→3)
  
  Q(s,a) = V(s) + A(s,a) - mean(A(s,·))  [Dueling combination]
  ```

- **Training:** Experience replay (50k buffer), batch 64, gamma 0.95, epsilon decay
- **Asymmetric Rewards** (reflects real banking costs):
  ```
  ALLOW on legitimate:    +10
  ALLOW on fraudster:     -20  (false negative = very costly)
  BLOCK on fraudster:     +10
  BLOCK on legitimate:    -15  (false positive = UX damage)
  STEP_UP (various):      ±5 to ±12 based on outcome
  ```

### Layer 4: Classical ML Pipeline (`classical_ml.py`)
Three complementary models:

**A. Isolation Forest** (per-user anomaly detection)
- Contamination: 0.05, n_estimators: 200
- Personalized to each user's history
- Returns anomaly_score in [0, 1]

**B. XGBoost Classifier** (supervised fraud detection)
- n_estimators: 300, max_depth: 6, learning_rate: 0.05
- Handles class imbalance with scale_pos_weight
- Returns fraud_probability in [0, 1]
- SHAP explanations for top-5 features
- **Fallback:** RandomForest if XGBoost unavailable

**C. Phishing URL Detector** (18-feature URL analyzer)
- No external ML required - heuristic features:
  - URL length, subdomain depth, IP as host
  - HTTPS presence, suspicious TLDs
  - Brand impersonation patterns
  - Banking keywords + redirect tricks
- Returns: phishing_probability, is_phishing, risk_level

### Layer 5: Ensemble Fusion (`fusion.py`)
Weighted combination of all signals with hard overrides:

```
composite_risk = 
  0.15 * static_rules +
  0.35 * rl_q_risk +
  0.30 * xgboost_fraud +
  0.20 * isolation_forest_anomaly

Decision thresholds:
  composite >= 0.70 → BLOCK
  composite >= 0.40 → STEP_UP
  composite <  0.40 → ALLOW

Hard overrides:
  - Static rules say BLOCK → immediate BLOCK
  - Phishing URL detected → immediate BLOCK
```

## 🎓 Simulation & Training (`simulation.py`)

Synthetic session generator for RL training:

- **User Profiles:** Behavioral characteristics (typing speed, typical hours, locations)
- **Legitimate Sessions:** Realistic human behavior
- **Fraud Attacks:**
  - Account takeover (stolen credentials)
  - Credential stuffing (rapid attempts)
  - SIM swap (victim's phone taken)
  - Phishing (tricked into fake site)
  - Insider threat (employee backdoor)

**Training Loop:**
- 10,000 episodes
- 30% fraud rate
- 30-session baseline per user
- Metrics: avg reward, accuracy, loss

## 🚀 API Server (`api_server.py`)

FastAPI application with endpoints:

```python
# Main authentication decision
POST /api/v1/authenticate
  Input: Full session telemetry (keystrokes, mouse, network, transaction)
  Output: {decision, risk_score, component_scores, reasons, explanation}
  Target latency: <50ms p99

# Phishing URL scanning
POST /api/v1/phishing/scan
  Input: {url: str}
  Output: {phishing_probability, is_phishing, risk_level}

# Online learning feedback
POST /api/v1/feedback
  Input: {session_id, true_label}  (0=legitimate, 1=fraud)

# Health check
GET /api/v1/health
  Returns: Model status, uptime, system health

# Interactive docs
GET /docs
  Swagger UI for testing endpoints
```

### Running the API

```bash
# Standalone demo (no server)
python api_server.py

# Start FastAPI server
python api_server.py server
# Access at http://localhost:8000/docs
```

## 📦 Installation

### Requirements

```bash
# Core (always required)
pip install numpy scipy

# ML/AI (optional but recommended)
pip install torch scikit-learn xgboost shap

# Web API (optional but recommended)
pip install fastapi uvicorn pydantic

# All at once
pip install -r requirements.txt
```

### Graceful Degradation

The system works even if optional libraries are missing:
- **Without PyTorch:** Uses NumPy fallback for RL (reduced performance)
- **Without XGBoost:** Uses RandomForest instead
- **Without FastAPI:** CLI demo mode only
- **Without SHAP:** Falls back to feature importance

## 🏃 Quick Start

### 1. Run System Tests
```bash
python run_tests.py
```

This tests all 5 layers individually with sample data.

### 2. Run Standalone Demo
```bash
python api_server.py
```

Runs 4 authentication scenarios:
1. **Legitimate user** - all signals green
2. **Suspicious user** - medium risk (new device + location)
3. **Likely fraud** - bot-like behavior + impossible velocity
4. **Phishing attack** - malicious URLs detected

Expected output shows decision, risk score, latency, and reasoning for each.

### 3. Start API Server (if FastAPI installed)
```bash
python api_server.py server
```

Then visit http://localhost:8000/docs for interactive API documentation.

## 📊 System Behavior

### Example 1: Legitimate Login
```
Input:
  - Known device, known IP
  - Normal typing pattern
  - Same city, reasonable velocity
  - Small transaction to known beneficiary
  - Business hours (9 AM)

Output:
  Decision: ALLOW
  Risk Score: 0.08
  Latency: 18ms
  Source: Ensemble (Low Risk)
```

### Example 2: Suspicious Activity
```
Input:
  - New device, VPN detected
  - New location (200 km away)
  - 2 failed attempts before this
  - Large transaction (50k) to new beneficiary
  - Odd timing (2 AM)

Output:
  Decision: STEP_UP
  Risk Score: 0.45
  Latency: 22ms
  Reasons:
    - New device + new location
    - VPN usage detected
    - Amount 5x user average
  → Requires OTP/biometric challenge
```

### Example 3: Clear Fraud
```
Input:
  - TOR exit node IP
  - Impossible geo-velocity (1500 km/h)
  - Bot-like typing (too fast, too consistent)
  - Large international transfer (500k to new account)
  - Failed logins detected

Output:
  Decision: BLOCK
  Risk Score: 0.92
  Latency: 15ms
  Source: Static Rule Engine (hard override)
  Reasons:
    - Impossible geo-velocity
    - TOR exit node detected
    - IP has high threat reputation
  → Session terminated immediately
```

## 🔐 Security Considerations

### What KavachX Protects Against
✓ Account takeover (stolen credentials)
✓ Brute-force attacks
✓ Phishing-based fraud
✓ SIM swap attacks
✓ Insider threats
✓ Bot/automated attacks
✓ Unusual transaction patterns
✓ Behavioral anomalies

### What KavachX Does NOT Protect Against
✗ Zero-day vulnerabilities (network layer)
✗ SSL/TLS downgrade attacks (network layer)
✗ Session hijacking after login (application layer)
✗ Database breaches (data layer)

**→ Use KavachX as part of a defense-in-depth strategy**

## 📈 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Detection Recall | > 95% | ✓ Achievable |
| False Positive Rate | < 2% | ✓ Achievable |
| API Latency p99 | < 50ms | ✓ On-target |
| AUROC | > 0.98 | ✓ Achievable with training |
| Step-up rate (legit) | < 3% | ✓ Configurable |

## 🧩 Integration Points

KavachX can be integrated with banking systems at login time:

```python
# Before allowing user into account
auth_response = api_client.post(
    "https://your-bank.com/api/v1/authenticate",
    json=session_telemetry
)

if auth_response.decision == "BLOCK":
    # Terminate session, log security event
    notify_fraud_team(auth_response.reasons)
elif auth_response.decision == "STEP_UP":
    # Require additional authentication (OTP, biometric)
    request_otp(user_id)
else:  # ALLOW
    # Grant access
    create_session(user_id)
```

## 📚 Key Design Decisions

1. **Asymmetric Reward Shaping:** False negatives (missed fraud) cost more than false positives (UX friction)
2. **Per-user Baselines:** Z-score normalization personalizes detection
3. **Hard Rules First:** Static engine filters obvious threats before expensive ML
4. **Ensemble Voting:** Multiple signals reduce single-point-of-failure
5. **Graceful Degradation:** Works without optional libraries
6. **Explainability:** Every decision includes reasoning and component scores

## 🛠️ Development

### File Structure
```
KavachX/
├── rules.py              # Layer 1: Static rules
├── features.py           # Layer 2: Feature extraction
├── rl_agent.py          # Layer 3: RL agent (Dueling DQN)
├── classical_ml.py      # Layer 4: Classical ML pipeline
├── fusion.py            # Layer 5: Ensemble fusion
├── simulation.py        # Simulation engine
├── api_server.py        # REST API server
├── requirements.txt     # Python dependencies
├── run_tests.py        # Integration tests
└── README.md           # This file
```

### Adding Custom Rules

```python
from rules import StaticRuleEngine

engine = StaticRuleEngine()

# Add custom rule
def check_custom_pattern(session_data):
    if some_condition(session_data):
        return True, "Custom pattern detected"
    return False, ""

# Use it
is_triggered, reason = check_custom_pattern(session_data)
```

### Training on Custom Data

```python
from features import FeatureExtractor
from rl_agent import DuelingDQNAgent
from classical_ml import XGBoostFraudClassifier

# Extract features from your data
extractor = FeatureExtractor()
X_train, y_train = [], []  # Load your data

for session, label in your_sessions:
    _, features = extractor.extract(session['user_id'], session)
    X_train.append(features)
    y_train.append(label)

# Train classifier
classifier = XGBoostFraudClassifier()
classifier.fit(np.array(X_train), np.array(y_train))
```

## 🎓 References

The system uses these techniques:
- **Dueling DQN:** Efficient RL architecture separating value and advantage
- **Double DQN:** Reduces overestimation in Q-learning
- **Experience Replay:** Breaks correlation in sequential data
- **Layer Normalization:** Stabilizes deep network training
- **Isolation Forest:** Anomaly detection without explicit modeling
- **XGBoost:** Gradient boosting for supervised learning
- **Ensemble Methods:** Combines diverse models for robustness
- **Behavioral Biometrics:** Keystroke, mouse patterns for user authentication

## 📞 Support

For questions or issues:
1. Check the README section above
2. Review demo scenarios in `api_server.py`
3. Run `python run_tests.py` to verify system integrity
4. Check component demo functions:
   - `demo_static_rules()` in `rules.py`
   - `demo_feature_extraction()` in `features.py`
   - `demo_rl_agent()` in `rl_agent.py`
   - `demo_classical_ml()` in `classical_ml.py`
   - `demo_ensemble_fusion()` in `fusion.py`

## 📄 License

This system is part of the PSB Hackathon project.

---

**Built with ❤️ for Indian Digital Banking Security**
