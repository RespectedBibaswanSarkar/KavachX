"""
LAYER 2: Feature Extraction
Extracts a 32-dimensional float32 state vector from raw session telemetry.
Implements per-user baseline normalization (Z-score with [-3, 3] clipping).

Feature structure (32D):
  [0:8]    Keystroke dynamics (dwell, flight, consistency, speed, rhythm)
  [8:14]   Mouse/touch biometrics (speed, curvature, pressure, jerk)
  [14:20]  Temporal & session patterns
  [20:26]  Network & device fingerprints
  [26:32]  Transaction & navigation context
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta


@dataclass
class KeystrokeMetrics:
    """Keystroke dynamics from typing pattern."""
    dwell_mean: float = 0.0  # Average key hold time (ms)
    dwell_std: float = 0.0   # Std dev of dwell times
    flight_mean: float = 0.0  # Average inter-key interval (ms)
    flight_std: float = 0.0   # Std dev of flight times
    bigram_consistency: float = 0.5  # Regularity score 0-1
    error_rate: float = 0.0   # Fraction of backspaces (0-1)
    typing_speed_wpm: float = 0.5  # Normalized words per minute (0-1)
    rhythm_score: float = 0.5  # Inverse coefficient of variation (0-1)


@dataclass
class MouseBiometrics:
    """Mouse/touch biometrics from interaction pattern."""
    speed_mean: float = 0.0   # Average mouse speed (pixels/ms)
    speed_std: float = 0.0    # Std dev of speeds
    curvature: float = 0.0    # Std dev of movement angles (0-1)
    pressure: float = 0.0     # Average touch pressure (0-1, normalized)
    peak_speed: float = 0.0   # Peak speed normalized (0-1)
    jerk_ratio: float = 0.0   # Fraction of high-acceleration movements (0-1)


@dataclass
class TemporalPatterns:
    """Temporal and session characteristics."""
    hour_of_day: float = 0.5   # Current hour normalized (0-1, 0=midnight)
    day_of_week: float = 0.5   # Day normalized (0-1, 0=Monday)
    session_duration_s: float = 0.0  # Session length in seconds
    login_attempts: float = 0.0      # Number of login attempts so far
    time_since_last_login_h: float = 0.0  # Hours since last login
    geo_velocity_kmh: float = 0.0    # Speed between last location (0-1 normalized)


@dataclass
class NetworkDeviceFingerprint:
    """Network and device identity markers."""
    known_device: float = 0.0       # Is device in trusted list? (0.0 or 1.0)
    known_ip_range: float = 0.0     # Is IP in known ranges? (0.0 or 1.0)
    vpn_detected: float = 0.0       # VPN usage detected? (0.0 or 1.0)
    tor_exit_node: float = 0.0      # TOR exit node? (0.0 or 1.0)
    ip_reputation_bad: float = 0.0  # High threat reputation? (0.0 or 1.0)
    failed_logins_24h: float = 0.0  # Count of failures normalized (0-1)


@dataclass
class TransactionContext:
    """Transaction and navigation context."""
    amount_inr: float = 0.0          # Amount normalized (0-1, max 1M)
    new_beneficiary: float = 0.0     # New recipient? (0.0 or 1.0)
    international: float = 0.0       # International txn? (0.0 or 1.0)
    daily_tx_count: float = 0.0      # Daily txns normalized (0-1, max 20)
    amount_exceeds_avg_5x: float = 0.0  # Amount >5x user avg? (0.0 or 1.0)
    navigation_anomaly_score: float = 0.0  # Unusual nav flow (0-1)


@dataclass
class FeatureVector:
    """Complete 32-dimensional feature state."""
    keystroke: KeystrokeMetrics = field(default_factory=KeystrokeMetrics)
    mouse: MouseBiometrics = field(default_factory=MouseBiometrics)
    temporal: TemporalPatterns = field(default_factory=TemporalPatterns)
    network: NetworkDeviceFingerprint = field(default_factory=NetworkDeviceFingerprint)
    transaction: TransactionContext = field(default_factory=TransactionContext)

    def to_numpy(self) -> np.ndarray:
        """Convert to 32-D numpy array (float32)."""
        return np.array([
            # [0:8] Keystroke dynamics
            self.keystroke.dwell_mean,
            self.keystroke.dwell_std,
            self.keystroke.flight_mean,
            self.keystroke.flight_std,
            self.keystroke.bigram_consistency,
            self.keystroke.error_rate,
            self.keystroke.typing_speed_wpm,
            self.keystroke.rhythm_score,
            # [8:14] Mouse biometrics
            self.mouse.speed_mean,
            self.mouse.speed_std,
            self.mouse.curvature,
            self.mouse.pressure,
            self.mouse.peak_speed,
            self.mouse.jerk_ratio,
            # [14:20] Temporal patterns
            self.temporal.hour_of_day,
            self.temporal.day_of_week,
            self.temporal.session_duration_s,
            self.temporal.login_attempts,
            self.temporal.time_since_last_login_h,
            self.temporal.geo_velocity_kmh,
            # [20:26] Network & device
            self.network.known_device,
            self.network.known_ip_range,
            self.network.vpn_detected,
            self.network.tor_exit_node,
            self.network.ip_reputation_bad,
            self.network.failed_logins_24h,
            # [26:32] Transaction & navigation
            self.transaction.amount_inr,
            self.transaction.new_beneficiary,
            self.transaction.international,
            self.transaction.daily_tx_count,
            self.transaction.amount_exceeds_avg_5x,
            self.transaction.navigation_anomaly_score,
        ], dtype=np.float32)


class FeatureExtractor:
    """
    Extract 32-D feature vector from raw session telemetry.
    Maintains per-user baseline statistics for Z-score normalization.
    """

    def __init__(self):
        """Initialize feature extractor with empty baseline storage."""
        self.user_baselines: Dict[str, Dict] = {}  # user_id -> {mean, std, session_count}
        self.baseline_window_size = 30  # Use last 30 sessions for baseline

    def update_baseline(self, user_id: str, feature_vector: np.ndarray):
        """
        Update per-user baseline with new feature observation.

        Args:
            user_id: User identifier
            feature_vector: 32-D numpy array
        """
        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = {
                "vectors": [],
                "mean": np.zeros(32, dtype=np.float32),
                "std": np.ones(32, dtype=np.float32),
            }

        baseline = self.user_baselines[user_id]
        baseline["vectors"].append(feature_vector)

        # Keep only last N sessions
        if len(baseline["vectors"]) > self.baseline_window_size:
            baseline["vectors"] = baseline["vectors"][-self.baseline_window_size :]

        # Recompute statistics
        vectors_array = np.array(baseline["vectors"])
        baseline["mean"] = np.mean(vectors_array, axis=0, dtype=np.float32)
        baseline["std"] = np.std(vectors_array, axis=0, dtype=np.float32)
        baseline["std"] = np.maximum(baseline["std"], 1e-6)  # Avoid division by zero

    def normalize_to_user_baseline(
        self,
        user_id: str,
        feature_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Z-score normalize feature vector against user's baseline.
        Clamp to [-3, 3] to cap outliers.

        Args:
            user_id: User identifier
            feature_vector: 32-D numpy array

        Returns:
            Normalized 32-D numpy array (float32)
        """
        if user_id not in self.user_baselines or len(self.user_baselines[user_id]["vectors"]) < 3:
            # Not enough baseline data yet - return unnormalized but clipped
            return np.clip(feature_vector, -3.0, 3.0).astype(np.float32)

        baseline = self.user_baselines[user_id]
        mean = baseline["mean"]
        std = baseline["std"]

        # Z-score: (x - mean) / std
        normalized = (feature_vector - mean) / std
        # Clip to [-3, 3]
        normalized = np.clip(normalized, -3.0, 3.0)

        return normalized.astype(np.float32)

    def extract_keystroke_dynamics(
        self,
        keystrokes: List[Dict],  # [{"key": "a", "dwell": 120, "flight": 85}, ...]
    ) -> KeystrokeMetrics:
        """
        Extract keystroke dynamics from raw keystroke events.

        Args:
            keystrokes: List of keystroke dicts with 'dwell' and 'flight' in ms

        Returns:
            KeystrokeMetrics object
        """
        metrics = KeystrokeMetrics()

        if not keystrokes or len(keystrokes) < 3:
            return metrics

        dwells = [k.get("dwell", 100) for k in keystrokes]
        flights = [k.get("flight", 50) for k in keystrokes if "flight" in k]

        metrics.dwell_mean = float(np.mean(dwells))
        metrics.dwell_std = float(np.std(dwells))
        metrics.flight_mean = float(np.mean(flights)) if flights else 0.0
        metrics.flight_std = float(np.std(flights)) if flights else 0.0

        # Bigram consistency: measure repetition of digraph patterns
        # Higher = more regular/predictable typing
        if len(keystrokes) >= 2:
            bigrams = [
                (keystrokes[i].get("key"), keystrokes[i + 1].get("key"))
                for i in range(len(keystrokes) - 1)
            ]
            unique_bigrams = len(set(bigrams))
            total_bigrams = len(bigrams)
            metrics.bigram_consistency = 1.0 - (unique_bigrams / max(total_bigrams, 1))

        # Error rate: fraction of backspaces
        error_count = sum(1 for k in keystrokes if k.get("key") == "backspace")
        metrics.error_rate = error_count / max(len(keystrokes), 1)

        # Typing speed: normalize to 0-1 (0=very slow, 1=very fast)
        # Assume avg human: 40 WPM = ~240 chars/min = ~4 chars/sec
        # Very fast bot: 1000+ chars/sec
        chars_per_sec = 60.0 / (metrics.dwell_mean + metrics.flight_mean + 1e-6)
        metrics.typing_speed_wpm = min(1.0, chars_per_sec / 10.0)

        # Rhythm score: inverse coefficient of variation of flight times
        if metrics.flight_mean > 0:
            cv = metrics.flight_std / metrics.flight_mean
            metrics.rhythm_score = 1.0 / (1.0 + cv)  # 0-1 scale

        return metrics

    def extract_mouse_biometrics(
        self,
        mouse_events: List[Dict],  # [{"x": 100, "y": 200, "pressure": 0.5}, ...]
    ) -> MouseBiometrics:
        """
        Extract mouse/touch biometrics from movement events.

        Args:
            mouse_events: List of mouse/touch events with x, y, pressure

        Returns:
            MouseBiometrics object
        """
        metrics = MouseBiometrics()

        if not mouse_events or len(mouse_events) < 2:
            return metrics

        # Calculate speeds between consecutive points
        speeds = []
        angles = []
        for i in range(len(mouse_events) - 1):
            p1 = mouse_events[i]
            p2 = mouse_events[i + 1]
            dx = p2.get("x", 0) - p1.get("x", 0)
            dy = p2.get("y", 0) - p1.get("y", 0)
            distance = np.sqrt(dx**2 + dy**2)
            # Assume ~20ms per event
            speed = distance / 0.02
            speeds.append(speed)

            # Curvature: angle change between consecutive movements
            if i > 0:
                prev_dx = p1.get("x", 0) - mouse_events[i - 1].get("x", 0)
                prev_dy = p1.get("y", 0) - mouse_events[i - 1].get("y", 0)
                angle = np.arctan2(dy, dx) - np.arctan2(prev_dy, prev_dx)
                angles.append(angle)

        metrics.speed_mean = float(np.mean(speeds)) if speeds else 0.0
        metrics.speed_std = float(np.std(speeds)) if speeds else 0.0
        metrics.peak_speed = min(1.0, float(np.max(speeds) / 500.0)) if speeds else 0.0

        # Curvature: std of angle changes, normalized to 0-1
        if angles:
            metrics.curvature = min(1.0, float(np.std(angles) / np.pi))

        # Pressure: average touch pressure (0-1)
        pressures = [e.get("pressure", 0.5) for e in mouse_events]
        metrics.pressure = float(np.mean(pressures))

        # Jerk ratio: fraction of movements with high acceleration
        if len(speeds) > 1:
            accelerations = np.abs(np.diff(speeds))
            high_accel = np.sum(accelerations > np.mean(speeds))
            metrics.jerk_ratio = high_accel / len(accelerations)

        return metrics

    def extract_temporal_patterns(
        self,
        current_time: datetime,
        session_duration_s: float,
        login_attempts: int,
        last_login_time: Optional[datetime] = None,
        geo_velocity_kmh: float = 0.0,
    ) -> TemporalPatterns:
        """
        Extract temporal and session characteristics.

        Args:
            current_time: Current login timestamp
            session_duration_s: Session duration in seconds
            login_attempts: Number of login attempts so far
            last_login_time: Previous login timestamp
            geo_velocity_kmh: Travel speed between locations

        Returns:
            TemporalPatterns object
        """
        metrics = TemporalPatterns()

        # Hour of day (0-1, 0=midnight)
        metrics.hour_of_day = current_time.hour / 23.0

        # Day of week (0-1, 0=Monday)
        metrics.day_of_week = current_time.weekday() / 6.0

        # Session duration normalized (max 1 hour = 3600s)
        metrics.session_duration_s = min(1.0, session_duration_s / 3600.0)

        # Login attempts normalized (max 5)
        metrics.login_attempts = min(1.0, login_attempts / 5.0)

        # Time since last login (max 168 hours = 1 week)
        if last_login_time:
            hours_since = (current_time - last_login_time).total_seconds() / 3600.0
            metrics.time_since_last_login_h = min(1.0, hours_since / 168.0)

        # Geo velocity normalized (max 900 km/h = speed limit for flag)
        metrics.geo_velocity_kmh = min(1.0, geo_velocity_kmh / 900.0)

        return metrics

    def extract_network_fingerprint(
        self,
        known_device: bool,
        known_ip_range: bool,
        vpn_detected: bool,
        tor_exit_node: bool,
        ip_reputation_bad: bool,
        failed_logins_24h: int,
    ) -> NetworkDeviceFingerprint:
        """
        Extract network and device fingerprinting features.

        Args:
            known_device: Device in trusted list?
            known_ip_range: IP in known ranges?
            vpn_detected: VPN usage detected?
            tor_exit_node: TOR exit node?
            ip_reputation_bad: High threat reputation?
            failed_logins_24h: Failed login count in last 24h

        Returns:
            NetworkDeviceFingerprint object
        """
        metrics = NetworkDeviceFingerprint()
        metrics.known_device = 1.0 if known_device else 0.0
        metrics.known_ip_range = 1.0 if known_ip_range else 0.0
        metrics.vpn_detected = 1.0 if vpn_detected else 0.0
        metrics.tor_exit_node = 1.0 if tor_exit_node else 0.0
        metrics.ip_reputation_bad = 1.0 if ip_reputation_bad else 0.0
        metrics.failed_logins_24h = min(1.0, failed_logins_24h / 10.0)  # Max 10 failures
        return metrics

    def extract_transaction_context(
        self,
        amount_inr: float = 0.0,
        new_beneficiary: bool = False,
        international: bool = False,
        daily_tx_count: int = 0,
        user_avg_amount_inr: float = 10000.0,
        urls_visited: List[str] = None,
    ) -> TransactionContext:
        """
        Extract transaction and navigation context.

        Args:
            amount_inr: Transaction amount in INR
            new_beneficiary: First time sending to this recipient?
            international: International transaction?
            daily_tx_count: Number of transactions today
            user_avg_amount_inr: User's average transaction amount
            urls_visited: List of URLs visited in session

        Returns:
            TransactionContext object
        """
        metrics = TransactionContext()

        # Amount normalized (max 1M INR)
        metrics.amount_inr = min(1.0, amount_inr / 1_000_000.0)

        metrics.new_beneficiary = 1.0 if new_beneficiary else 0.0
        metrics.international = 1.0 if international else 0.0

        # Daily tx count normalized (max 20)
        metrics.daily_tx_count = min(1.0, daily_tx_count / 20.0)

        # Amount exceeds 5x user average?
        exceeds_5x = amount_inr > (user_avg_amount_inr * 5.0)
        metrics.amount_exceeds_avg_5x = 1.0 if exceeds_5x else 0.0

        # Navigation anomaly: check for suspicious URL patterns
        if urls_visited:
            phishing_indicators = 0
            for url in urls_visited:
                # Simple heuristics
                if any(x in url.lower() for x in ["login", "verify", "otp", "kyc"]):
                    if "http" not in url[:8]:  # No HTTPS
                        phishing_indicators += 1
            metrics.navigation_anomaly_score = min(1.0, phishing_indicators / 3.0)

        return metrics

    def extract(
        self,
        user_id: str,
        session_data: Dict,
        normalize: bool = True,
    ) -> Tuple[FeatureVector, np.ndarray]:
        """
        Complete feature extraction pipeline.

        Args:
            user_id: User identifier
            session_data: Dict with session telemetry:
                - keystrokes: List[Dict]
                - mouse_events: List[Dict]
                - current_time: datetime
                - session_duration_s: float
                - login_attempts: int
                - last_login_time: Optional[datetime]
                - geo_velocity_kmh: float
                - known_device, known_ip_range, vpn_detected, tor_exit_node, etc.
                - amount_inr: float
                - new_beneficiary: bool
                - international: bool
                - daily_tx_count: int
                - user_avg_amount_inr: float
                - urls_visited: List[str]
            normalize: Apply per-user baseline normalization?

        Returns:
            Tuple of (FeatureVector object, 32-D normalized numpy array)
        """
        features = FeatureVector(
            keystroke=self.extract_keystroke_dynamics(session_data.get("keystrokes", [])),
            mouse=self.extract_mouse_biometrics(session_data.get("mouse_events", [])),
            temporal=self.extract_temporal_patterns(
                session_data.get("current_time", datetime.now()),
                session_data.get("session_duration_s", 0.0),
                session_data.get("login_attempts", 0),
                session_data.get("last_login_time"),
                session_data.get("geo_velocity_kmh", 0.0),
            ),
            network=self.extract_network_fingerprint(
                session_data.get("known_device", False),
                session_data.get("known_ip_range", False),
                session_data.get("vpn_detected", False),
                session_data.get("tor_exit_node", False),
                session_data.get("ip_reputation_bad", False),
                session_data.get("failed_logins_24h", 0),
            ),
            transaction=self.extract_transaction_context(
                session_data.get("amount_inr", 0.0),
                session_data.get("new_beneficiary", False),
                session_data.get("international", False),
                session_data.get("daily_tx_count", 0),
                session_data.get("user_avg_amount_inr", 10000.0),
                session_data.get("urls_visited", []),
            ),
        )

        # Convert to numpy array
        feature_array = features.to_numpy()

        # Normalize against user baseline and update baseline
        if normalize:
            feature_array = self.normalize_to_user_baseline(user_id, feature_array)
        self.update_baseline(user_id, feature_array)

        return features, feature_array


def demo_feature_extraction():
    """Demonstrate feature extraction on sample telemetry."""
    print("\n" + "=" * 70)
    print("LAYER 2: FEATURE EXTRACTION DEMO")
    print("=" * 70)

    extractor = FeatureExtractor()

    # Sample keystroke events
    keystrokes = [
        {"key": "a", "dwell": 120, "flight": 85},
        {"key": "d", "dwell": 130, "flight": 80},
        {"key": "m", "dwell": 110, "flight": 90},
        {"key": "i", "dwell": 125, "flight": 75},
        {"key": "n", "dwell": 115, "flight": 88},
    ]

    # Sample mouse events
    mouse_events = [
        {"x": 100, "y": 200, "pressure": 0.6},
        {"x": 105, "y": 210, "pressure": 0.7},
        {"x": 115, "y": 220, "pressure": 0.65},
        {"x": 130, "y": 235, "pressure": 0.7},
    ]

    # Session data
    session_data = {
        "keystrokes": keystrokes,
        "mouse_events": mouse_events,
        "current_time": datetime.now(),
        "session_duration_s": 120,
        "login_attempts": 1,
        "last_login_time": datetime.now() - timedelta(days=1),
        "geo_velocity_kmh": 50.0,
        "known_device": True,
        "known_ip_range": True,
        "vpn_detected": False,
        "tor_exit_node": False,
        "ip_reputation_bad": False,
        "failed_logins_24h": 0,
        "amount_inr": 5000.0,
        "new_beneficiary": False,
        "international": False,
        "daily_tx_count": 2,
        "user_avg_amount_inr": 10000.0,
        "urls_visited": ["https://www.sbi.co.in/login"],
    }

    print("\n[1] Extracting features for legitimate user session...")
    features, feature_array = extractor.extract("user_legit", session_data, normalize=False)
    print(f"  Keystroke dwell_mean: {features.keystroke.dwell_mean:.1f} ms")
    print(f"  Keystroke typing_speed: {features.keystroke.typing_speed_wpm:.2f}")
    print(f"  Mouse speed_mean: {features.mouse.speed_mean:.1f} px/ms")
    print(f"  Temporal hour_of_day: {features.temporal.hour_of_day:.2f}")
    print(f"  Network known_device: {features.network.known_device:.1f}")
    print(f"  Transaction amount (norm): {features.transaction.amount_inr:.2f}")
    print(f"  Feature vector shape: {feature_array.shape}, dtype: {feature_array.dtype}")
    print(f"  Feature vector (first 8 elements): {feature_array[:8]}")

    # Simulate suspicious session
    suspicious_data = session_data.copy()
    suspicious_data["known_device"] = False
    suspicious_data["known_ip_range"] = False
    suspicious_data["amount_inr"] = 100000.0  # 10x average
    suspicious_data["new_beneficiary"] = True
    suspicious_data["international"] = True

    print("\n[2] Extracting features for SUSPICIOUS session...")
    features2, feature_array2 = extractor.extract("user_suspicious", suspicious_data, normalize=False)
    print(f"  Keystroke typing_speed: {features2.keystroke.typing_speed_wpm:.2f}")
    print(f"  Network known_device: {features2.network.known_device:.1f}")
    print(f"  Network known_ip_range: {features2.network.known_ip_range:.1f}")
    print(f"  Transaction amount (norm): {features2.transaction.amount_inr:.2f}")
    print(f"  Transaction new_beneficiary: {features2.transaction.new_beneficiary:.1f}")
    print(f"  Transaction international: {features2.transaction.international:.1f}")

    # Build baseline with 5 similar sessions
    print("\n[3] Building baseline from multiple sessions...")
    for i in range(5):
        extractor.extract("user_legit", session_data, normalize=False)

    print("\n[4] Normalizing against baseline...")
    features3, feature_array_norm = extractor.extract("user_legit", session_data, normalize=True)
    print(f"  Feature vector (first 8 elements, after normalization): {feature_array_norm[:8]}")
    print(f"  Baseline mean: {extractor.user_baselines['user_legit']['mean'][:3]}")
    print(f"  Baseline std: {extractor.user_baselines['user_legit']['std'][:3]}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_feature_extraction()
