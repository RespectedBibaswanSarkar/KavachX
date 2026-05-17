"""
LAYER 5: Ensemble Fusion
Combines all signals into a final authentication decision.

Weights:
  - RL agent Q-risk:          35%
  - XGBoost fraud prob:       30%
  - Isolation Forest anomaly: 20%
  - Static rule engine risk:  15%

Thresholds:
  - composite >= 0.70 → BLOCK
  - composite >= 0.40 → STEP_UP
  - composite <  0.40 → ALLOW

Hard overrides:
  - Static rule engine says BLOCK → immediate BLOCK
  - Phishing URL detected → immediate BLOCK
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime


@dataclass
class ComponentScores:
    """Component scores from each layer."""
    rule_engine_risk: float = 0.0
    rl_q_risk: float = 0.0
    xgboost_fraud: float = 0.0
    isolation_forest_anomaly: float = 0.0
    phishing_detected: bool = False


@dataclass
class FusionOutput:
    """Final ensemble decision and explanation."""
    session_id: str
    decision: str = "ALLOW"  # "ALLOW", "STEP_UP", "BLOCK"
    risk_score: float = 0.0  # 0.0-1.0
    component_scores: ComponentScores = field(default_factory=ComponentScores)
    source: str = ""  # Which layer made the decision
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)


class EnsembleFusion:
    """
    Fuse all authentication signals into a final decision.
    Implements weighted ensemble voting + hard overrides.
    """

    # Decision thresholds
    THRESHOLD_BLOCK = 0.70
    THRESHOLD_STEP_UP = 0.40

    # Weights
    WEIGHT_RULES = 0.15
    WEIGHT_RL = 0.35
    WEIGHT_XGBOOST = 0.30
    WEIGHT_ISO_FOREST = 0.20

    DECISION_NAMES = {
        "ALLOW": "Proceed normally",
        "STEP_UP": "Request OTP / biometric challenge",
        "BLOCK": "Terminate session immediately",
    }

    def __init__(self):
        """Initialize fusion layer."""
        pass

    def convert_q_values_to_risk(self, q_values: Dict) -> float:
        """
        Convert RL Q-values to risk score.

        Args:
            q_values: Dict with keys "ALLOW", "STEP_UP", "BLOCK"

        Returns:
            Risk score 0-1 (high Q for BLOCK = high risk)
        """
        allow_q = q_values.get("ALLOW", 0.0)
        step_up_q = q_values.get("STEP_UP", 0.0)
        block_q = q_values.get("BLOCK", 0.0)

        # Normalize Q-values to [0, 1] range
        q_min = min(allow_q, step_up_q, block_q)
        q_max = max(allow_q, step_up_q, block_q)
        q_range = max(q_max - q_min, 1.0)

        # Higher Q for BLOCK indicates higher perceived risk
        normalized_block_q = (block_q - q_min) / q_range
        return float(np.clip(normalized_block_q, 0.0, 1.0))

    def fuse(
        self,
        session_id: str,
        rule_engine_output: Dict,  # {block, risk_score, reasons}
        rl_prediction: Dict,  # {action, q_values, confidence}
        xgboost_fraud_prob: float,
        isolation_forest_anomaly: float,
        phishing_detected: bool = False,
        verbose: bool = False,
    ) -> FusionOutput:
        """
        Fuse all components into final decision.

        Args:
            session_id: Session identifier
            rule_engine_output: From StaticRuleEngine.evaluate()
            rl_prediction: From DuelingDQNAgent.predict()
            xgboost_fraud_prob: Fraud probability 0-1
            isolation_forest_anomaly: Anomaly score 0-1
            phishing_detected: Is phishing URL detected?
            verbose: Log reasoning?

        Returns:
            FusionOutput with decision and explanations
        """
        output = FusionOutput(session_id=session_id)

        # Component scores
        rule_risk = rule_engine_output.get("risk_score", 0.0)
        rl_risk = self.convert_q_values_to_risk(rl_prediction.get("q_values", {}))

        output.component_scores = ComponentScores(
            rule_engine_risk=rule_risk,
            rl_q_risk=rl_risk,
            xgboost_fraud=xgboost_fraud_prob,
            isolation_forest_anomaly=isolation_forest_anomaly,
            phishing_detected=phishing_detected,
        )

        # Hard override 1: Static rules say BLOCK
        if rule_engine_output.get("block", False):
            output.decision = "BLOCK"
            output.source = "Static Rule Engine"
            output.reasons = rule_engine_output.get("reasons", ["Static rules triggered BLOCK"])
            output.risk_score = min(1.0, rule_risk)
            output.confidence = 0.95
            if verbose:
                print(f"  [OVERRIDE] Static rules detected threat: {output.reasons}")
            return output

        # Hard override 2: Phishing detected
        if phishing_detected:
            output.decision = "BLOCK"
            output.source = "Phishing Detector"
            output.reasons = ["Phishing URL detected in session"]
            output.risk_score = 1.0
            output.confidence = 0.99
            if verbose:
                print(f"  [OVERRIDE] Phishing URL blocked")
            return output

        # Weighted ensemble of ML signals
        composite_risk = (
            self.WEIGHT_RULES * rule_risk
            + self.WEIGHT_RL * rl_risk
            + self.WEIGHT_XGBOOST * xgboost_fraud_prob
            + self.WEIGHT_ISO_FOREST * isolation_forest_anomaly
        )

        output.risk_score = float(np.clip(composite_risk, 0.0, 1.0))

        # Decision thresholds
        if composite_risk >= self.THRESHOLD_BLOCK:
            output.decision = "BLOCK"
            output.source = "Ensemble (High Risk)"
        elif composite_risk >= self.THRESHOLD_STEP_UP:
            output.decision = "STEP_UP"
            output.source = "Ensemble (Medium Risk)"
        else:
            output.decision = "ALLOW"
            output.source = "Ensemble (Low Risk)"

        # Build explanation
        high_risk_components = []
        if rule_risk > 0.5:
            high_risk_components.append(f"Rules ({rule_risk:.2f})")
        if rl_risk > 0.5:
            high_risk_components.append(f"RL ({rl_risk:.2f})")
        if xgboost_fraud_prob > 0.5:
            high_risk_components.append(f"XGBoost ({xgboost_fraud_prob:.2f})")
        if isolation_forest_anomaly > 0.5:
            high_risk_components.append(f"AnomalyDetector ({isolation_forest_anomaly:.2f})")

        if high_risk_components:
            output.reasons = [
                f"Composite risk: {composite_risk:.3f}",
                f"High-risk signals: {', '.join(high_risk_components)}",
            ]
        else:
            output.reasons = [f"All signals nominal (composite: {composite_risk:.3f})"]

        # Confidence: how certain are we?
        # Consensus among components increases confidence
        risk_variance = np.var([rule_risk, rl_risk, xgboost_fraud_prob, isolation_forest_anomaly])
        output.confidence = float(1.0 - risk_variance)  # High variance = low confidence

        if verbose:
            print(f"  Component scores: Rules={rule_risk:.3f}, RL={rl_risk:.3f}, "
                  f"XGB={xgboost_fraud_prob:.3f}, IsoForest={isolation_forest_anomaly:.3f}")
            print(f"  Composite: {composite_risk:.3f}")
            print(f"  Decision: {output.decision} (confidence: {output.confidence:.3f})")

        return output

    def get_decision_explanation(self, output: FusionOutput) -> str:
        """
        Generate human-readable explanation of decision.

        Args:
            output: FusionOutput from fuse()

        Returns:
            Formatted explanation string
        """
        lines = [
            f"Session: {output.session_id}",
            f"Decision: {output.decision} ({self.DECISION_NAMES.get(output.decision, '')})",
            f"Risk Score: {output.risk_score:.2%}",
            f"Confidence: {output.confidence:.2%}",
            f"Source: {output.source}",
            "",
            "Component Breakdown:",
            f"  • Static Rules:      {output.component_scores.rule_engine_risk:.2%} (weight: {self.WEIGHT_RULES:.0%})",
            f"  • RL Agent:          {output.component_scores.rl_q_risk:.2%} (weight: {self.WEIGHT_RL:.0%})",
            f"  • XGBoost:           {output.component_scores.xgboost_fraud:.2%} (weight: {self.WEIGHT_XGBOOST:.0%})",
            f"  • Anomaly Detector:  {output.component_scores.isolation_forest_anomaly:.2%} (weight: {self.WEIGHT_ISO_FOREST:.0%})",
            f"  • Phishing Detected: {'YES' if output.component_scores.phishing_detected else 'NO'}",
            "",
            "Reasoning:",
        ]
        for reason in output.reasons:
            lines.append(f"  • {reason}")

        return "\n".join(lines)


def demo_ensemble_fusion():
    """Demonstrate ensemble fusion."""
    print("\n" + "=" * 70)
    print("LAYER 5: ENSEMBLE FUSION DEMO")
    print("=" * 70)

    fusion = EnsembleFusion()

    # Scenario 1: Legitimate user (low risk across all signals)
    print("\n[SCENARIO 1] Legitimate user (all signals green)")
    rule_output_1 = {
        "block": False,
        "risk_score": 0.1,
        "reasons": [],
    }
    rl_pred_1 = {
        "action": 0,
        "q_values": {"ALLOW": 5.0, "STEP_UP": 2.0, "BLOCK": 1.0},
    }
    result_1 = fusion.fuse(
        session_id="sess_001",
        rule_engine_output=rule_output_1,
        rl_prediction=rl_pred_1,
        xgboost_fraud_prob=0.05,
        isolation_forest_anomaly=0.1,
        phishing_detected=False,
        verbose=True,
    )
    print(f"\n{fusion.get_decision_explanation(result_1)}")

    # Scenario 2: Suspicious user (medium risk)
    print("\n[SCENARIO 2] Suspicious user (medium risk)")
    rule_output_2 = {
        "block": False,
        "risk_score": 0.3,
        "reasons": ["New device + new location"],
    }
    rl_pred_2 = {
        "action": 1,
        "q_values": {"ALLOW": 2.0, "STEP_UP": 4.5, "BLOCK": 2.5},
    }
    result_2 = fusion.fuse(
        session_id="sess_002",
        rule_engine_output=rule_output_2,
        rl_prediction=rl_pred_2,
        xgboost_fraud_prob=0.45,
        isolation_forest_anomaly=0.35,
        phishing_detected=False,
        verbose=True,
    )
    print(f"\n{fusion.get_decision_explanation(result_2)}")

    # Scenario 3: Fraud detected (high risk across multiple signals)
    print("\n[SCENARIO 3] Likely fraud (high risk)")
    rule_output_3 = {
        "block": False,
        "risk_score": 0.7,
        "reasons": ["TOR exit node detected", "Impossible geo-velocity"],
    }
    rl_pred_3 = {
        "action": 2,
        "q_values": {"ALLOW": 1.0, "STEP_UP": 2.0, "BLOCK": 7.5},
    }
    result_3 = fusion.fuse(
        session_id="sess_003",
        rule_engine_output=rule_output_3,
        rl_prediction=rl_pred_3,
        xgboost_fraud_prob=0.88,
        isolation_forest_anomaly=0.85,
        phishing_detected=False,
        verbose=True,
    )
    print(f"\n{fusion.get_decision_explanation(result_3)}")

    # Scenario 4: Phishing URL detected (hard override)
    print("\n[SCENARIO 4] Phishing URL (hard override)")
    rule_output_4 = {
        "block": False,
        "risk_score": 0.2,
        "reasons": [],
    }
    rl_pred_4 = {
        "action": 0,
        "q_values": {"ALLOW": 4.0, "STEP_UP": 1.5, "BLOCK": 1.0},
    }
    result_4 = fusion.fuse(
        session_id="sess_004",
        rule_engine_output=rule_output_4,
        rl_prediction=rl_pred_4,
        xgboost_fraud_prob=0.1,
        isolation_forest_anomaly=0.2,
        phishing_detected=True,
        verbose=True,
    )
    print(f"\n{fusion.get_decision_explanation(result_4)}")

    # Scenario 5: Static rule override (hard block)
    print("\n[SCENARIO 5] Static rule override (brute-force)")
    rule_output_5 = {
        "block": True,
        "risk_score": 0.95,
        "reasons": ["Brute-force pattern: 8 failures in last 24h"],
    }
    rl_pred_5 = {
        "action": 0,
        "q_values": {"ALLOW": 3.0, "STEP_UP": 2.0, "BLOCK": 1.0},
    }
    result_5 = fusion.fuse(
        session_id="sess_005",
        rule_engine_output=rule_output_5,
        rl_prediction=rl_pred_5,
        xgboost_fraud_prob=0.3,
        isolation_forest_anomaly=0.2,
        phishing_detected=False,
        verbose=True,
    )
    print(f"\n{fusion.get_decision_explanation(result_5)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_ensemble_fusion()
