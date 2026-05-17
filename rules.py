"""
LAYER 1: Static Rule Engine
Hard-coded instant checks that fire before any ML inference.
Returns risk assessment without requiring feature extraction or model inference.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import math


@dataclass
class RuleEngineOutput:
    """Output from static rule engine."""
    block: bool
    reasons: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0 to 1.0


class StaticRuleEngine:
    """
    Hard-coded fraud detection rules that execute before any ML.
    Designed for instant blocking of obvious threats.
    """

    def __init__(
        self,
        tor_exit_nodes: set = None,
        bad_ip_ranges: set = None,
        max_velocity_kmh: float = 850.0,
        max_concurrent_sessions: int = 3,
        max_failures_24h: int = 5,
        failure_window_hours: int = 24,
    ):
        """
        Initialize static rule engine with threat lists.

        Args:
            tor_exit_nodes: Set of known TOR exit node IPs
            bad_ip_ranges: Set of known malicious IP prefixes
            max_velocity_kmh: Max geo-velocity before flagging (850 km/h = physically impossible)
            max_concurrent_sessions: Max simultaneous sessions allowed
            max_failures_24h: Max failed login attempts in 24h before block
            failure_window_hours: Time window for counting failures
        """
        self.tor_exit_nodes = tor_exit_nodes or set()
        self.bad_ip_ranges = bad_ip_ranges or set()
        self.max_velocity_kmh = max_velocity_kmh
        self.max_concurrent_sessions = max_concurrent_sessions
        self.max_failures_24h = max_failures_24h
        self.failure_window_hours = failure_window_hours

        # In-memory tracking for this session
        self.login_failures: Dict[str, List[datetime]] = {}  # user_id -> [timestamps]
        self.active_sessions: Dict[str, List[Dict]] = {}  # user_id -> [session_data]

    def check_tor_exit_node(self, ip: str) -> Tuple[bool, str]:
        """Detect if IP is known TOR exit node."""
        if ip in self.tor_exit_nodes:
            return True, "IP is known TOR exit node"
        return False, ""

    def check_anonymiser(self, ip: str) -> Tuple[bool, str]:
        """Detect if IP is from known anonymiser service."""
        # Check IP prefix ranges
        for prefix in self.bad_ip_ranges:
            if ip.startswith(prefix):
                return True, f"IP from known anonymiser range: {prefix}"
        return False, ""

    def check_ip_reputation(self, ip: str, ip_threat_level: float = 0.0) -> Tuple[bool, str]:
        """
        Check IP against threat intelligence.

        Args:
            ip: IP address to check
            ip_threat_level: Threat level 0.0-1.0 from external feed (0=safe, 1=confirmed malicious)
        """
        if ip_threat_level > 0.7:
            return True, f"IP has high threat reputation ({ip_threat_level:.2f})"
        return False, ""

    def check_geo_velocity(
        self,
        user_id: str,
        current_lat: float,
        current_lon: float,
        current_time: datetime,
        last_lat: float = None,
        last_lon: float = None,
        last_time: datetime = None,
    ) -> Tuple[bool, str]:
        """
        Detect impossible geo-velocity (physically impossible travel between logins).
        Haversine distance formula for great-circle distance.

        Args:
            user_id: User identifier
            current_lat, current_lon: Current geolocation
            current_time: Current login time
            last_lat, last_lon: Previous geolocation
            last_time: Previous login time
        """
        if last_lat is None or last_lon is None or last_time is None:
            return False, ""  # No previous location to compare

        # Haversine distance
        R = 6371.0  # Earth radius in km
        lat1, lon1 = math.radians(last_lat), math.radians(last_lon)
        lat2, lon2 = math.radians(current_lat), math.radians(current_lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        distance_km = R * c

        # Time difference in hours
        time_diff = (current_time - last_time).total_seconds() / 3600.0
        if time_diff <= 0:
            time_diff = 0.001  # Avoid division by zero

        velocity_kmh = distance_km / time_diff

        if velocity_kmh > self.max_velocity_kmh:
            return (
                True,
                f"Impossible geo-velocity: {velocity_kmh:.1f} km/h (max: {self.max_velocity_kmh})",
            )
        return False, ""

    def check_brute_force_pattern(self, user_id: str) -> Tuple[bool, str]:
        """
        Detect >N failed logins in 24h window (brute-force attempt).

        Args:
            user_id: User identifier
        """
        if user_id not in self.login_failures:
            return False, ""

        now = datetime.now()
        cutoff = now - timedelta(hours=self.failure_window_hours)

        # Filter out old failures
        recent_failures = [
            t for t in self.login_failures[user_id] if t > cutoff
        ]
        self.login_failures[user_id] = recent_failures

        if len(recent_failures) > self.max_failures_24h:
            return (
                True,
                f"Brute-force pattern: {len(recent_failures)} failures in last {self.failure_window_hours}h",
            )
        return False, ""

    def record_login_failure(self, user_id: str, timestamp: datetime = None):
        """Record a failed login attempt for brute-force detection."""
        if timestamp is None:
            timestamp = datetime.now()
        if user_id not in self.login_failures:
            self.login_failures[user_id] = []
        self.login_failures[user_id].append(timestamp)

    def check_concurrent_sessions(self, user_id: str) -> Tuple[bool, str]:
        """
        Detect >N simultaneous sessions (concurrent session anomaly).

        Args:
            user_id: User identifier
        """
        if user_id not in self.active_sessions:
            return False, ""

        active_count = len(self.active_sessions[user_id])
        if active_count > self.max_concurrent_sessions:
            return (
                True,
                f"Concurrent session anomaly: {active_count} simultaneous sessions (max: {self.max_concurrent_sessions})",
            )
        return False, ""

    def register_session(
        self,
        user_id: str,
        session_id: str,
        device_id: str,
        location: Tuple[float, float],
        timestamp: datetime = None,
    ):
        """Register a new session for concurrent session tracking."""
        if timestamp is None:
            timestamp = datetime.now()
        if user_id not in self.active_sessions:
            self.active_sessions[user_id] = []
        self.active_sessions[user_id].append(
            {
                "session_id": session_id,
                "device_id": device_id,
                "location": location,
                "timestamp": timestamp,
            }
        )

    def deregister_session(self, user_id: str, session_id: str):
        """Deregister a session (logout/timeout)."""
        if user_id in self.active_sessions:
            self.active_sessions[user_id] = [
                s for s in self.active_sessions[user_id]
                if s["session_id"] != session_id
            ]

    def check_device_location_time_combo(
        self,
        user_id: str,
        is_new_device: bool,
        is_new_location: bool,
        hour_of_day: int,
        known_device: bool,
        known_location: bool,
    ) -> Tuple[bool, str]:
        """
        Detect suspicious combination: new device + new location + off-hours.

        Args:
            user_id: User identifier
            is_new_device: Device not seen before for this user
            is_new_location: Location not seen before for this user
            hour_of_day: Current hour (0-23)
            known_device: Device is in user's trusted list
            known_location: Location is in user's trusted list
        """
        off_hours = hour_of_day < 5 or hour_of_day > 23  # Midnight to 5 AM + late night

        if is_new_device and is_new_location and off_hours:
            return (
                True,
                "Suspicious combo: new device + new location + off-hours (midnight-5am or >11pm)",
            )

        # Also flag: new device + new location (even during business hours, worth noting)
        if is_new_device and is_new_location:
            return True, "Risky combo: new device + new location (even during business hours)"

        return False, ""

    def evaluate(
        self,
        user_id: str,
        session_data: Dict,
    ) -> RuleEngineOutput:
        """
        Run all static checks on a session.

        Args:
            user_id: User identifier
            session_data: Dict with keys:
                - ip: Client IP address
                - ip_threat_level: 0.0-1.0 threat rating
                - current_lat, current_lon: Current geolocation
                - current_time: datetime of login
                - last_lat, last_lon, last_time: Previous login location/time (optional)
                - is_new_device, is_new_location: Bool flags
                - hour_of_day: 0-23
                - known_device, known_location: Bool flags
                - session_id: Session identifier
                - device_id: Device identifier
                - recent_failed_logins: Count of failures in last 24h

        Returns:
            RuleEngineOutput with block decision and reasons
        """
        output = RuleEngineOutput(block=False, reasons=[], risk_score=0.0)

        # 1. TOR/Anonymiser check
        is_tor, reason = self.check_tor_exit_node(session_data.get("ip", ""))
        if is_tor:
            output.reasons.append(reason)
            output.risk_score += 0.25

        is_anon, reason = self.check_anonymiser(session_data.get("ip", ""))
        if is_anon:
            output.reasons.append(reason)
            output.risk_score += 0.20

        # 2. IP Reputation
        is_bad_ip, reason = self.check_ip_reputation(
            session_data.get("ip", ""),
            session_data.get("ip_threat_level", 0.0),
        )
        if is_bad_ip:
            output.reasons.append(reason)
            output.risk_score += 0.30

        # 3. Geo-velocity check
        is_impossible_velocity, reason = self.check_geo_velocity(
            user_id,
            session_data.get("current_lat", 0),
            session_data.get("current_lon", 0),
            session_data.get("current_time", datetime.now()),
            session_data.get("last_lat"),
            session_data.get("last_lon"),
            session_data.get("last_time"),
        )
        if is_impossible_velocity:
            output.reasons.append(reason)
            output.risk_score += 0.25

        # 4. Brute-force pattern
        is_brute_force, reason = self.check_brute_force_pattern(user_id)
        if is_brute_force:
            output.reasons.append(reason)
            output.risk_score += 0.30
            output.block = True  # Hard block on brute-force

        # 5. Concurrent sessions
        is_concurrent, reason = self.check_concurrent_sessions(user_id)
        if is_concurrent:
            output.reasons.append(reason)
            output.risk_score += 0.25

        # 6. Device + Location + Time combination
        is_suspicious_combo, reason = self.check_device_location_time_combo(
            user_id,
            session_data.get("is_new_device", False),
            session_data.get("is_new_location", False),
            session_data.get("hour_of_day", 12),
            session_data.get("known_device", False),
            session_data.get("known_location", False),
        )
        if is_suspicious_combo:
            output.reasons.append(reason)
            output.risk_score += 0.15

        # Hard blocks for obvious threats
        if output.risk_score > 0.65:
            output.block = True

        # Clamp risk score to [0, 1]
        output.risk_score = min(1.0, max(0.0, output.risk_score))

        return output


# Demo/testing function
def demo_static_rules():
    """Demonstrate static rule engine on sample data."""
    print("\n" + "=" * 70)
    print("LAYER 1: STATIC RULE ENGINE DEMO")
    print("=" * 70)

    engine = StaticRuleEngine(
        tor_exit_nodes={"192.0.2.1", "198.51.100.5"},
        bad_ip_ranges={"10.0.0", "172.16"},
    )

    # Scenario 1: Legitimate login
    print("\n[SCENARIO 1] Legitimate login - normal params")
    result1 = engine.evaluate(
        "user_123",
        {
            "ip": "203.0.113.42",
            "ip_threat_level": 0.1,
            "current_lat": 28.7041,
            "current_lon": 77.1025,
            "current_time": datetime.now(),
            "is_new_device": False,
            "is_new_location": False,
            "hour_of_day": 10,
            "known_device": True,
            "known_location": True,
            "session_id": "sess_001",
            "device_id": "dev_001",
        },
    )
    print(f"  Block: {result1.block}, Risk: {result1.risk_score:.2f}")
    print(f"  Reasons: {result1.reasons or ['None - all checks passed']}")

    # Scenario 2: TOR exit node + suspicious IP combo
    print("\n[SCENARIO 2] TOR exit node detected")
    result2 = engine.evaluate(
        "user_456",
        {
            "ip": "192.0.2.1",
            "ip_threat_level": 0.85,
            "current_lat": 28.7041,
            "current_lon": 77.1025,
            "current_time": datetime.now(),
            "is_new_device": True,
            "is_new_location": True,
            "hour_of_day": 2,
            "known_device": False,
            "known_location": False,
            "session_id": "sess_002",
            "device_id": "dev_002",
        },
    )
    print(f"  Block: {result2.block}, Risk: {result2.risk_score:.2f}")
    print(f"  Reasons:")
    for r in result2.reasons:
        print(f"    - {r}")

    # Scenario 3: Impossible geo-velocity
    print("\n[SCENARIO 3] Impossible geo-velocity (Delhi to Mumbai in 30 min)")
    now = datetime.now()
    result3 = engine.evaluate(
        "user_789",
        {
            "ip": "203.0.113.42",
            "ip_threat_level": 0.0,
            "current_lat": 19.0760,
            "current_lon": 72.8777,  # Mumbai
            "current_time": now,
            "last_lat": 28.7041,
            "last_lon": 77.1025,  # Delhi
            "last_time": now - timedelta(minutes=30),
            "is_new_device": False,
            "is_new_location": True,
            "hour_of_day": 15,
            "known_device": True,
            "known_location": False,
            "session_id": "sess_003",
            "device_id": "dev_001",
        },
    )
    print(f"  Block: {result3.block}, Risk: {result3.risk_score:.2f}")
    print(f"  Reasons:")
    for r in result3.reasons:
        print(f"    - {r}")

    # Scenario 4: Brute-force pattern
    print("\n[SCENARIO 4] Brute-force pattern (7 failures in 2 hours)")
    for i in range(7):
        engine.record_login_failure("user_brute", datetime.now() - timedelta(minutes=i*15))
    result4 = engine.evaluate(
        "user_brute",
        {
            "ip": "203.0.113.99",
            "ip_threat_level": 0.0,
            "current_lat": 28.7041,
            "current_lon": 77.1025,
            "current_time": datetime.now(),
            "is_new_device": False,
            "is_new_location": False,
            "hour_of_day": 14,
            "known_device": True,
            "known_location": True,
            "session_id": "sess_004",
            "device_id": "dev_001",
        },
    )
    print(f"  Block: {result4.block}, Risk: {result4.risk_score:.2f}")
    print(f"  Reasons:")
    for r in result4.reasons:
        print(f"    - {r}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_static_rules()
