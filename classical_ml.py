"""
LAYER 4: Classical ML Pipeline
Three complementary models:
  A. Isolation Forest (per-user anomaly detection)
  B. XGBoost Fraud Classifier (with SHAP explanations)
  C. Phishing URL Detector

All gracefully degrade if libraries unavailable.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import re
from urllib.parse import urlparse, parse_qs
import json

# Try sklearn
try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Try xgboost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    # Fallback to RandomForest
    if HAS_SKLEARN:
        from sklearn.ensemble import RandomForestClassifier

# Try SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class IsolationForestAnomalyDetector:
    """
    Per-user anomaly detection using Isolation Forest.
    Trained on each user's historical sessions.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200):
        """
        Initialize detector.

        Args:
            contamination: Expected proportion of anomalies (0-1)
            n_estimators: Number of isolation trees
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.models: Dict[str, IsolationForest] = {}  # user_id -> model

    def fit(self, user_id: str, feature_vectors: List[np.ndarray]):
        """
        Fit isolation forest on user's historical sessions.

        Args:
            user_id: User identifier
            feature_vectors: List of 32-D feature vectors (historical sessions)
        """
        if not HAS_SKLEARN:
            return

        if len(feature_vectors) < 10:
            return  # Need minimum samples

        X = np.array(feature_vectors, dtype=np.float32)
        model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
        )
        model.fit(X)
        self.models[user_id] = model

    def score(self, user_id: str, feature_vector: np.ndarray) -> float:
        """
        Compute anomaly score for a feature vector.

        Args:
            user_id: User identifier
            feature_vector: 32-D feature vector

        Returns:
            Anomaly score in [0, 1] (0=normal, 1=anomalous)
        """
        if not HAS_SKLEARN or user_id not in self.models:
            return 0.0  # No model available

        model = self.models[user_id]
        # anomaly_score ranges from -1 to 1 (negative = normal, positive = anomaly)
        raw_score = model.score_samples(feature_vector.reshape(1, -1))[0]
        # Normalize to [0, 1]
        anomaly_score = 1.0 / (1.0 + np.exp(raw_score))  # Sigmoid
        return float(np.clip(anomaly_score, 0.0, 1.0))


class XGBoostFraudClassifier:
    """
    XGBoost fraud classifier with SHAP explanations.
    Falls back to RandomForest if XGBoost unavailable.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 20.0,  # Handle class imbalance
    ):
        """
        Initialize classifier.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Tree depth
            learning_rate: Boosting learning rate
            scale_pos_weight: Weight for minority class (fraud)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.model = None
        self.explainer = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train classifier.

        Args:
            X: (N, 32) feature matrix
            y: (N,) binary labels (0=legit, 1=fraud)
        """
        if HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                scale_pos_weight=self.scale_pos_weight,
                objective="binary:logistic",
                random_state=42,
                eval_metric="auc",
            )
            self.model.fit(X, y, verbose=False)

            # Initialize SHAP explainer if available
            if HAS_SHAP:
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except Exception:
                    pass
        elif HAS_SKLEARN:
            # Fallback to RandomForest
            from sklearn.ensemble import RandomForestClassifier

            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
            )
            self.model.fit(X, y)
        else:
            # No ML library available
            pass

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        """
        Predict fraud probability for a single sample.

        Args:
            feature_vector: 32-D feature vector

        Returns:
            Fraud probability in [0, 1]
        """
        if self.model is None:
            return 0.5

        try:
            X = feature_vector.reshape(1, -1)
            if HAS_XGBOOST:
                proba = self.model.predict_proba(X)[0][1]
            else:
                proba = self.model.predict_proba(X)[0][1]
            return float(np.clip(proba, 0.0, 1.0))
        except Exception:
            return 0.5

    def explain(self, feature_vector: np.ndarray, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Explain prediction with SHAP values.

        Args:
            feature_vector: 32-D feature vector
            top_n: Number of top features to return

        Returns:
            List of (feature_index, importance) tuples
        """
        if not HAS_SHAP or self.explainer is None or self.model is None:
            # Fallback: use feature importance from model
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                top_indices = np.argsort(importances)[-top_n:][::-1]
                return [
                    (f"feature_{idx}", float(importances[idx]))
                    for idx in top_indices
                ]
            return []

        try:
            shap_values = self.explainer.shap_values(feature_vector.reshape(1, -1))[0]
            top_indices = np.argsort(np.abs(shap_values))[-top_n:][::-1]
            return [
                (f"feature_{idx}", float(shap_values[idx]))
                for idx in top_indices
            ]
        except Exception:
            return []


class PhishingURLDetector:
    """
    Phishing URL detector using 18 engineered features.
    No external ML libraries required.
    """

    # Suspicious TLDs commonly used in phishing
    SUSPICIOUS_TLDS = {
        ".xyz", ".top", ".click", ".tk", ".ml", ".ga", ".cf", ".pw",
        ".online", ".website", ".site", ".space", ".download"
    }

    # Brand names to check for impersonation
    BANKING_BRANDS = {
        "sbi", "hdfc", "icici", "axis", "kotak", "pnb", "bob",
        "paytm", "phonepe", "googlepay", "whatsapp", "gpay",
        "upi", "npci", "rbi", "uidai"
    }

    # Banking/sensitive keywords
    BANKING_KEYWORDS = {
        "login", "signin", "verify", "kyc", "otp", "password",
        "netbanking", "onlinebanking", "secure", "confirm",
        "update", "validate", "authenticate"
    }

    def extract_features(self, url: str) -> np.ndarray:
        """
        Extract 18 features from URL for phishing detection.

        Args:
            url: URL string

        Returns:
            18-D feature vector
        """
        features = np.zeros(18, dtype=np.float32)

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            query = parsed.query.lower()
            scheme = parsed.scheme.lower()

            # [0] URL length
            features[0] = min(1.0, len(url) / 100.0)

            # [1] Subdomain depth (more = suspicious)
            subdomain_count = domain.count(".")
            features[1] = min(1.0, subdomain_count / 5.0)

            # [2] IP address as host (highly suspicious)
            if re.match(r"^\d+\.\d+\.\d+\.\d+$", domain):
                features[2] = 1.0

            # [3] HTTPS present
            features[3] = 1.0 if scheme == "https" else 0.0

            # [4] Suspicious TLD
            for tld in self.SUSPICIOUS_TLDS:
                if domain.endswith(tld):
                    features[4] = 1.0
                    break

            # [5] Dot count (many = suspicious)
            features[5] = min(1.0, domain.count(".") / 3.0)

            # [6] Hyphen count (many = suspicious)
            features[6] = min(1.0, domain.count("-") / 3.0)

            # [7] Brand impersonation in domain
            for brand in self.BANKING_BRANDS:
                if brand in domain and domain != brand:
                    features[7] = 1.0
                    break

            # [8] Brand impersonation in path
            for brand in self.BANKING_BRANDS:
                if brand in path:
                    features[8] = 1.0
                    break

            # [9] Banking keywords in path
            keyword_count = sum(1 for kw in self.BANKING_KEYWORDS if kw in path)
            features[9] = min(1.0, keyword_count / 3.0)

            # [10] Banking keywords in query
            keyword_count = sum(1 for kw in self.BANKING_KEYWORDS if kw in query)
            features[10] = min(1.0, keyword_count / 3.0)

            # [11] Percent-encoding in URL (tricks to bypass filters)
            features[11] = min(1.0, url.count("%") / 5.0)

            # [12] Double slash tricks
            features[12] = 1.0 if "//" in url[8:] else 0.0  # Skip scheme://

            # [13] Query string length
            features[13] = min(1.0, len(query) / 200.0)

            # [14] Number of @ symbols (email-like tricks)
            features[14] = min(1.0, url.count("@") / 2.0)

            # [15] Suspicious domain patterns
            if re.search(r"(verify|confirm|update|login|signin).*bank", domain):
                features[15] = 1.0

            # [16] Query parameter count
            features[16] = min(1.0, len(parse_qs(query)) / 5.0)

            # [17] Redirect patterns (e.g., ?url=, ?redirect=)
            if "url=" in query or "redirect=" in query or "goto=" in query:
                features[17] = 1.0

        except Exception:
            pass  # Return zeros on parsing error

        return features

    def score(self, url: str) -> Dict[str, any]:
        """
        Compute phishing risk score for URL.

        Args:
            url: URL string

        Returns:
            Dict with phishing_probability, is_phishing, risk_level
        """
        features = self.extract_features(url)

        # Weighted combination of features
        # IP address + no HTTPS + brand impersonation + banking keywords = high risk
        weights = np.array([
            0.02,  # [0] URL length
            0.05,  # [1] Subdomain depth
            0.15,  # [2] IP as host (VERY suspicious)
            0.10,  # [3] HTTPS present (inverse)
            0.08,  # [4] Suspicious TLD
            0.03,  # [5] Dot count
            0.03,  # [6] Hyphen count
            0.12,  # [7] Brand impersonation in domain
            0.10,  # [8] Brand impersonation in path
            0.08,  # [9] Banking keywords in path
            0.06,  # [10] Banking keywords in query
            0.05,  # [11] Percent-encoding
            0.05,  # [12] Double slash tricks
            0.03,  # [13] Query length
            0.04,  # [14] @ symbols
            0.10,  # [15] Suspicious domain patterns
            0.02,  # [16] Query param count
            0.07,  # [17] Redirect patterns
        ], dtype=np.float32)

        # Adjust feature [3] (HTTPS) - invert so no-HTTPS increases risk
        features[3] = 1.0 - features[3]

        phishing_prob = float(np.dot(features, weights))
        phishing_prob = np.clip(phishing_prob, 0.0, 1.0)

        # Risk level
        if phishing_prob > 0.7:
            risk_level = "CRITICAL"
        elif phishing_prob > 0.4:
            risk_level = "HIGH"
        elif phishing_prob > 0.2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        is_phishing = phishing_prob > 0.5

        return {
            "phishing_probability": phishing_prob,
            "is_phishing": is_phishing,
            "risk_level": risk_level,
            "features": features,
        }


def demo_classical_ml():
    """Demonstrate classical ML pipeline."""
    print("\n" + "=" * 70)
    print("LAYER 4: CLASSICAL ML PIPELINE DEMO")
    print("=" * 70)

    # 1. Isolation Forest
    print("\n[1] Isolation Forest (Anomaly Detection)")
    if HAS_SKLEARN:
        iso_detector = IsolationForestAnomalyDetector()

        # Generate dummy historical sessions for user
        hist_sessions = [np.random.randn(32).astype(np.float32) * 0.5 for _ in range(30)]
        iso_detector.fit("user_123", hist_sessions)

        # Score a normal session
        normal_session = np.random.randn(32).astype(np.float32) * 0.5
        anomaly_score = iso_detector.score("user_123", normal_session)
        print(f"  Normal session anomaly score: {anomaly_score:.3f}")

        # Score an anomalous session (outlier)
        anomalous_session = np.ones(32, dtype=np.float32) * 10.0
        anomaly_score_high = iso_detector.score("user_123", anomalous_session)
        print(f"  Anomalous session anomaly score: {anomaly_score_high:.3f}")
    else:
        print("  (sklearn not available)")

    # 2. XGBoost Classifier
    print("\n[2] XGBoost Fraud Classifier")
    if HAS_XGBOOST or HAS_SKLEARN:
        classifier = XGBoostFraudClassifier()

        # Generate dummy training data
        X_train = np.random.randn(200, 32).astype(np.float32)
        y_train = np.concatenate([
            np.zeros(150, dtype=int),  # 150 legitimate
            np.ones(50, dtype=int),    # 50 fraudulent
        ])
        classifier.fit(X_train, y_train)

        # Predict on dummy samples
        legit_sample = np.random.randn(32).astype(np.float32) * 0.5
        fraud_proba = classifier.predict_proba(legit_sample)
        print(f"  Legitimate sample fraud probability: {fraud_proba:.3f}")

        explanations = classifier.explain(legit_sample, top_n=3)
        if explanations:
            print(f"  Top 3 important features:")
            for feat_name, importance in explanations:
                print(f"    - {feat_name}: {importance:.3f}")
    else:
        print("  (xgboost and sklearn not available)")

    # 3. Phishing Detector
    print("\n[3] Phishing URL Detector")
    detector = PhishingURLDetector()

    test_urls = [
        ("https://www.sbi.co.in/login", "Legitimate SBI"),
        ("http://192.168.1.1/login?verify=otp", "Fake IP-based"),
        ("http://sbi-secure-verify.click/login", "Phishing domain"),
        ("https://secure.kotak.com/update", "Legitimate Kotak"),
        ("http://update-hdfc-kyc-verify.tk/login", "Phishing with keywords"),
    ]

    for url, description in test_urls:
        result = detector.score(url)
        print(f"  {description}")
        print(f"    URL: {url}")
        print(f"    Phishing prob: {result['phishing_probability']:.3f}, "
              f"Risk: {result['risk_level']}, Is phishing: {result['is_phishing']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_classical_ml()
