"""
LAYER: REST API SERVER
FastAPI application with endpoints for authentication decision and phishing scanning.

Endpoints:
  POST /api/v1/authenticate - Main decision endpoint
  POST /api/v1/phishing/scan - URL phishing check
  POST /api/v1/feedback - Online learning feedback
  GET /api/v1/health - Health check
  GET /docs - Interactive API documentation

Target latency: <50ms p99
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
import logging

# Core imports
import numpy as np

# Import all system layers
from rules import StaticRuleEngine, RuleEngineOutput
from features import FeatureExtractor, FeatureVector
from rl_agent import DuelingDQNAgent
from classical_ml import IsolationForestAnomalyDetector, XGBoostFraudClassifier, PhishingURLDetector
from fusion import EnsembleFusion, FusionOutput
from simulation import SessionGenerator, UserProfile, SimulationTrainingLoop

# Try to import FastAPI
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models (Input/Output Schemas)
# ============================================================================

class KeystrokeData(BaseModel):
    """Keystroke telemetry."""
    key: str
    dwell: float = Field(..., description="Key hold time (ms)")
    flight: float = Field(..., description="Inter-key interval (ms)")


class MouseEventData(BaseModel):
    """Mouse/touch event."""
    x: float
    y: float
    pressure: float = Field(..., ge=0.0, le=1.0, description="Touch pressure (0-1)")


class AuthenticationRequest(BaseModel):
    """Full authentication request with session telemetry."""
    session_id: str
    user_id: str
    keystrokes: List[KeystrokeData] = Field(default_factory=list)
    mouse_events: List[MouseEventData] = Field(default_factory=list)
    current_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    session_duration_s: float = 0.0
    login_attempts: int = 1
    last_login_time: Optional[str] = None
    current_lat: float = 0.0
    current_lon: float = 0.0
    last_lat: Optional[float] = None
    last_lon: Optional[float] = None
    geo_velocity_kmh: float = 0.0
    known_device: bool = False
    known_ip_range: bool = False
    vpn_detected: bool = False
    tor_exit_node: bool = False
    ip_reputation_bad: bool = False
    failed_logins_24h: int = 0
    amount_inr: float = 0.0
    new_beneficiary: bool = False
    international: bool = False
    daily_tx_count: int = 0
    user_avg_amount_inr: float = 10000.0
    urls_visited: List[str] = Field(default_factory=list)
    ip: str = "203.0.113.1"  # Client IP
    ip_threat_level: float = Field(default=0.0, ge=0.0, le=1.0)


class AuthenticationResponse(BaseModel):
    """Authentication decision response."""
    session_id: str
    decision: str  # ALLOW, STEP_UP, BLOCK
    risk_score: float = Field(..., ge=0.0, le=1.0)
    component_scores: Dict[str, float] = {}
    reasons: List[str] = []
    explanation: str = ""
    latency_ms: float
    timestamp: str


class PhishingRequest(BaseModel):
    """Phishing scan request."""
    url: str


class PhishingResponse(BaseModel):
    """Phishing scan response."""
    url: str
    phishing_probability: float = Field(..., ge=0.0, le=1.0)
    is_phishing: bool
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL


class FeedbackRequest(BaseModel):
    """User feedback for online learning."""
    session_id: str
    true_label: int = Field(..., ge=0, le=1, description="0=legitimate, 1=fraud")
    feedback_text: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    uptime_seconds: float
    models_loaded: Dict[str, bool] = {}


# ============================================================================
# Core Authentication System
# ============================================================================

class KavachXAuthenticationSystem:
    """
    Complete AI-Driven Behavioral Authentication System.
    Integrates all 5 layers + RL training.
    """

    def __init__(self):
        """Initialize all system components."""
        self.start_time = time.time()

        logger.info("Initializing KavachX Authentication System...")

        # Layer 1: Static Rules
        self.rule_engine = StaticRuleEngine()
        logger.info("✓ Static Rule Engine initialized")

        # Layer 2: Feature Extraction
        self.feature_extractor = FeatureExtractor()
        logger.info("✓ Feature Extractor initialized")

        # Layer 3: RL Agent
        self.rl_agent = DuelingDQNAgent()
        logger.info("✓ RL Agent (Dueling Double DQN) initialized")

        # Layer 4: Classical ML
        self.iso_forest = IsolationForestAnomalyDetector()
        self.xgboost_classifier = XGBoostFraudClassifier()
        self.phishing_detector = PhishingURLDetector()
        logger.info("✓ Classical ML Pipeline initialized (IsoForest, XGBoost, Phishing)")

        # Layer 5: Ensemble Fusion
        self.fusion = EnsembleFusion()
        logger.info("✓ Ensemble Fusion Layer initialized")

        logger.info("All systems online! System ready for authentication.")

    def authenticate(self, request: AuthenticationRequest) -> AuthenticationResponse:
        """
        Main authentication decision pipeline.
        Target latency: <50ms

        Args:
            request: Authentication request with full session telemetry

        Returns:
            Authentication decision response
        """
        start_time = time.time()

        try:
            # Prepare session data dict
            session_data = {
                "keystrokes": [dict(ks) for ks in request.keystrokes],
                "mouse_events": [dict(me) for me in request.mouse_events],
                "current_time": datetime.fromisoformat(request.current_time) if request.current_time else datetime.now(),
                "session_duration_s": request.session_duration_s,
                "login_attempts": request.login_attempts,
                "last_login_time": datetime.fromisoformat(request.last_login_time) if request.last_login_time else None,
                "geo_velocity_kmh": request.geo_velocity_kmh,
                "known_device": request.known_device,
                "known_ip_range": request.known_ip_range,
                "vpn_detected": request.vpn_detected,
                "tor_exit_node": request.tor_exit_node,
                "ip_reputation_bad": request.ip_reputation_bad,
                "failed_logins_24h": request.failed_logins_24h,
                "amount_inr": request.amount_inr,
                "new_beneficiary": request.new_beneficiary,
                "international": request.international,
                "daily_tx_count": request.daily_tx_count,
                "user_avg_amount_inr": request.user_avg_amount_inr,
                "urls_visited": request.urls_visited,
                "current_lat": request.current_lat,
                "current_lon": request.current_lon,
                "last_lat": request.last_lat,
                "last_lon": request.last_lon,
            }

            # LAYER 1: Static Rules
            rule_session = {
                "ip": request.ip,
                "ip_threat_level": request.ip_threat_level,
                "current_lat": request.current_lat,
                "current_lon": request.current_lon,
                "current_time": session_data["current_time"],
                "last_lat": request.last_lat,
                "last_lon": request.last_lon,
                "last_time": session_data["last_login_time"],
                "is_new_device": not request.known_device,
                "is_new_location": not request.known_ip_range,
                "hour_of_day": session_data["current_time"].hour,
                "known_device": request.known_device,
                "known_location": request.known_ip_range,
            }
            rule_output = self.rule_engine.evaluate(request.user_id, rule_session)

            # LAYER 2: Feature Extraction
            features, feature_array = self.feature_extractor.extract(
                request.user_id, session_data, normalize=True
            )

            # LAYER 3: RL Agent
            rl_prediction = self.rl_agent.predict(feature_array)

            # LAYER 4: Classical ML
            iso_anomaly = self.iso_forest.score(request.user_id, feature_array)
            xgb_fraud = self.xgboost_classifier.predict_proba(feature_array)

            # Phishing detection
            phishing_detected = False
            for url in request.urls_visited:
                phishing_result = self.phishing_detector.score(url)
                if phishing_result["is_phishing"]:
                    phishing_detected = True
                    break

            # LAYER 5: Fusion
            fusion_output = self.fusion.fuse(
                session_id=request.session_id,
                rule_engine_output={
                    "block": rule_output.block,
                    "risk_score": rule_output.risk_score,
                    "reasons": rule_output.reasons,
                },
                rl_prediction=rl_prediction,
                xgboost_fraud_prob=xgb_fraud,
                isolation_forest_anomaly=iso_anomaly,
                phishing_detected=phishing_detected,
            )

            latency_ms = (time.time() - start_time) * 1000.0

            # Build response
            response = AuthenticationResponse(
                session_id=request.session_id,
                decision=fusion_output.decision,
                risk_score=fusion_output.risk_score,
                component_scores={
                    "rule_engine": rule_output.risk_score,
                    "rl_agent": fusion_output.component_scores.rl_q_risk,
                    "xgboost": fusion_output.component_scores.xgboost_fraud,
                    "anomaly_detector": fusion_output.component_scores.isolation_forest_anomaly,
                },
                reasons=fusion_output.reasons,
                explanation=self.fusion.get_decision_explanation(fusion_output),
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
            )

            logger.info(
                f"Session {request.session_id}: {response.decision} "
                f"(risk: {response.risk_score:.3f}, latency: {latency_ms:.1f}ms)"
            )

            return response

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise

    def scan_phishing(self, url: str) -> PhishingResponse:
        """Scan URL for phishing."""
        result = self.phishing_detector.score(url)
        return PhishingResponse(
            url=url,
            phishing_probability=result["phishing_probability"],
            is_phishing=result["is_phishing"],
            risk_level=result["risk_level"],
        )

    def process_feedback(self, feedback: FeedbackRequest):
        """
        Process user feedback for online learning.
        In production, this would trigger model retraining.
        """
        logger.info(
            f"Feedback received: session={feedback.session_id}, "
            f"label={'FRAUD' if feedback.true_label else 'LEGITIMATE'}"
        )
        # In production: store in database, trigger async retraining
        pass

    def get_health(self) -> HealthResponse:
        """Get system health status."""
        uptime = time.time() - self.start_time
        return HealthResponse(
            status="HEALTHY",
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime,
            models_loaded={
                "rule_engine": True,
                "feature_extractor": True,
                "rl_agent": True,
                "iso_forest": True,
                "xgboost": True,
                "phishing_detector": True,
                "fusion": True,
            },
        )


# ============================================================================
# FastAPI Application
# ============================================================================

if HAS_FASTAPI:
    app = FastAPI(
        title="KavachX - AI-Driven Behavioral Authentication",
        description="Production-grade fraud detection for digital banking",
        version="1.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize system
    auth_system = KavachXAuthenticationSystem()

    @app.post("/api/v1/authenticate", response_model=AuthenticationResponse)
    async def authenticate(request: AuthenticationRequest) -> AuthenticationResponse:
        """
        Main authentication endpoint.
        Processes full session telemetry and returns fraud risk assessment.
        """
        return auth_system.authenticate(request)

    @app.post("/api/v1/phishing/scan", response_model=PhishingResponse)
    async def scan_phishing(request: PhishingRequest) -> PhishingResponse:
        """Scan URL for phishing indicators."""
        return auth_system.scan_phishing(request.url)

    @app.post("/api/v1/feedback")
    async def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
        """Submit user feedback for online learning."""
        background_tasks.add_task(auth_system.process_feedback, feedback)
        return {"status": "feedback_received", "session_id": feedback.session_id}

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return auth_system.get_health()

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "KavachX",
            "status": "online",
            "docs_url": "/docs",
        }


# ============================================================================
# Standalone Demo Mode
# ============================================================================

def demo_standalone():
    """
    Standalone CLI demo without FastAPI.
    Runs 4 authentication scenarios.
    """
    print("\n" + "=" * 80)
    print("KAVACHX - AI-Driven Behavioral Authentication System")
    print("=" * 80)
    print("Standalone Demo Mode (No FastAPI required)\n")

    # Initialize system
    auth_system = KavachXAuthenticationSystem()

    # Scenario 1: Legitimate User
    print("\n" + "-" * 80)
    print("[SCENARIO 1] LEGITIMATE USER - Normal login during business hours")
    print("-" * 80)
    legit_request = AuthenticationRequest(
        session_id="sess_legitimate_001",
        user_id="user_john_doe",
        keystrokes=[
            KeystrokeData(key="j", dwell=110, flight=80),
            KeystrokeData(key="o", dwell=120, flight=75),
            KeystrokeData(key="h", dwell=105, flight=85),
        ] * 10,  # Realistic typing pattern
        mouse_events=[
            MouseEventData(x=100, y=200, pressure=0.6),
            MouseEventData(x=110, y=210, pressure=0.65),
            MouseEventData(x=120, y=220, pressure=0.6),
        ] * 5,
        session_duration_s=180.0,
        login_attempts=1,
        current_lat=28.7041,
        current_lon=77.1025,
        last_lat=28.7041,
        last_lon=77.1025,
        geo_velocity_kmh=10.0,
        known_device=True,
        known_ip_range=True,
        vpn_detected=False,
        tor_exit_node=False,
        ip_reputation_bad=False,
        failed_logins_24h=0,
        amount_inr=5000.0,
        new_beneficiary=False,
        international=False,
        daily_tx_count=1,
        user_avg_amount_inr=10000.0,
        urls_visited=["https://www.sbi.co.in/netbanking/login"],
        ip="203.0.113.42",
        ip_threat_level=0.1,
    )

    response_legit = auth_system.authenticate(legit_request)
    print(f"\nDecision: {response_legit.decision}")
    print(f"Risk Score: {response_legit.risk_score:.1%}")
    print(f"Latency: {response_legit.latency_ms:.1f}ms")
    print(f"Reasons: {', '.join(response_legit.reasons[:2])}")

    # Scenario 2: Suspicious User
    print("\n" + "-" * 80)
    print("[SCENARIO 2] SUSPICIOUS USER - New device + new location")
    print("-" * 80)
    suspicious_request = AuthenticationRequest(
        session_id="sess_suspicious_002",
        user_id="user_jane_smith",
        keystrokes=[KeystrokeData(key=c, dwell=180, flight=150) for c in "abcdefgh"],
        mouse_events=[
            MouseEventData(x=float(100 + i * 10), y=float(200 + i * 5), pressure=0.7)
            for i in range(5)
        ],
        session_duration_s=90.0,
        login_attempts=2,
        current_lat=19.0760,
        current_lon=72.8777,  # Mumbai (different from last login)
        last_lat=28.7041,
        last_lon=77.1025,  # Delhi
        geo_velocity_kmh=200.0,
        known_device=False,
        known_ip_range=False,
        vpn_detected=True,
        tor_exit_node=False,
        ip_reputation_bad=False,
        failed_logins_24h=1,
        amount_inr=50000.0,
        new_beneficiary=True,
        international=False,
        daily_tx_count=3,
        user_avg_amount_inr=10000.0,
        urls_visited=["https://www.sbi.co.in/netbanking/login"],
        ip="198.51.100.99",
        ip_threat_level=0.4,
    )

    response_suspicious = auth_system.authenticate(suspicious_request)
    print(f"\nDecision: {response_suspicious.decision}")
    print(f"Risk Score: {response_suspicious.risk_score:.1%}")
    print(f"Latency: {response_suspicious.latency_ms:.1f}ms")
    print(f"Reasons: {', '.join(response_suspicious.reasons[:2])}")

    # Scenario 3: Likely Fraud
    print("\n" + "-" * 80)
    print("[SCENARIO 3] LIKELY FRAUD - Bot-like behavior + impossible velocity")
    print("-" * 80)
    fraud_request = AuthenticationRequest(
        session_id="sess_fraud_003",
        user_id="user_victim_account",
        keystrokes=[KeystrokeData(key=c, dwell=50, flight=30) for c in "abcdefghij"],  # Too fast
        mouse_events=[
            MouseEventData(x=float(100 + i * 50), y=float(200 + i * 50), pressure=0.5)
            for i in range(3)
        ],
        session_duration_s=30.0,
        login_attempts=4,
        current_lat=1.3521,  # Singapore
        current_lon=103.8198,
        last_lat=28.7041,  # Delhi
        last_lon=77.1025,
        geo_velocity_kmh=1500.0,  # Impossible speed!
        known_device=False,
        known_ip_range=False,
        vpn_detected=True,
        tor_exit_node=True,  # TOR exit node!
        ip_reputation_bad=True,
        failed_logins_24h=5,
        amount_inr=500000.0,  # 50x average
        new_beneficiary=True,
        international=True,
        daily_tx_count=8,
        user_avg_amount_inr=10000.0,
        urls_visited=["http://192.168.1.1/login", "http://sbi-verify.click/kyc"],
        ip="192.0.2.99",
        ip_threat_level=0.95,
    )

    response_fraud = auth_system.authenticate(fraud_request)
    print(f"\nDecision: {response_fraud.decision}")
    print(f"Risk Score: {response_fraud.risk_score:.1%}")
    print(f"Latency: {response_fraud.latency_ms:.1f}ms")
    print(f"Reasons: {', '.join(response_fraud.reasons[:2])}")

    # Scenario 4: Phishing Attack
    print("\n" + "-" * 80)
    print("[SCENARIO 4] PHISHING ATTACK - Malicious URLs in session")
    print("-" * 80)
    phishing_request = AuthenticationRequest(
        session_id="sess_phishing_004",
        user_id="user_phishing_victim",
        keystrokes=[KeystrokeData(key=c, dwell=120, flight=80) for c in "password123"],
        mouse_events=[MouseEventData(x=100.0, y=200.0, pressure=0.6)],
        session_duration_s=45.0,
        login_attempts=1,
        current_lat=28.7041,
        current_lon=77.1025,
        known_device=True,
        known_ip_range=True,
        vpn_detected=False,
        tor_exit_node=False,
        ip_reputation_bad=False,
        failed_logins_24h=0,
        amount_inr=0.0,
        urls_visited=[
            "http://sbi-secure-verify.click/login?verify=otp",  # Phishing URL
            "https://www.google.com",
        ],
        ip="203.0.113.50",
    )

    response_phishing = auth_system.authenticate(phishing_request)
    print(f"\nDecision: {response_phishing.decision}")
    print(f"Risk Score: {response_phishing.risk_score:.1%}")
    print(f"Latency: {response_phishing.latency_ms:.1f}ms")
    print(f"Reasons: {', '.join(response_phishing.reasons[:2])}")

    # Health check
    print("\n" + "-" * 80)
    print("SYSTEM HEALTH CHECK")
    print("-" * 80)
    health = auth_system.get_health()
    print(f"Status: {health.status}")
    print(f"Uptime: {health.uptime_seconds:.1f}s")
    print(f"Models: {json.dumps(health.models_loaded, indent=2)}")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    if HAS_FASTAPI and len(sys.argv) > 1 and sys.argv[1] == "server":
        # Start server: python api_server.py server
        print("Starting KavachX API Server...")
        print("Documentation available at: http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    else:
        # Run standalone demo
        demo_standalone()
