"""
SIMULATION & TRAINING
Synthetic session generator for RL training.

Generates realistic legitimate and fraudulent sessions.
Implements 10k episode training loop with RL agent.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
import random

# Import all layers
from features import FeatureExtractor, FeatureVector
from rl_agent import DuelingDQNAgent


@dataclass
class UserProfile:
    """User behavioral profile for realistic session generation."""
    user_id: str
    dwell_mean: float = 100.0  # ms
    dwell_std: float = 20.0    # ms
    flight_mean: float = 70.0  # ms inter-key interval
    flight_std: float = 15.0   # ms
    typing_wpm: float = 60.0   # typical typing speed
    typical_login_hour: int = 9  # preferred login time (0-23)
    typical_days: List[int] = None  # preferred weekdays [0=Mon, 6=Sun]
    device_id: str = "device_001"
    ip_prefix: str = "203.0.113"  # ISP/location prefix
    avg_tx_amount_inr: float = 5000.0
    known_locations: List[Tuple[float, float]] = None  # (lat, lon)

    def __post_init__(self):
        """Initialize defaults."""
        if self.typical_days is None:
            self.typical_days = [0, 1, 2, 3, 4]  # Weekdays
        if self.known_locations is None:
            self.known_locations = [(28.7041, 77.1025)]  # Delhi


class SessionGenerator:
    """Generate synthetic legitimate and fraudulent sessions."""

    def __init__(self, feature_extractor: FeatureExtractor):
        """
        Initialize generator.

        Args:
            feature_extractor: FeatureExtractor instance for feature generation
        """
        self.feature_extractor = feature_extractor

    def generate_legitimate_session(
        self,
        user_profile: UserProfile,
        num_sessions_so_far: int = 0,
    ) -> Dict:
        """
        Generate realistic legitimate session.

        Args:
            user_profile: User behavioral profile
            num_sessions_so_far: Number of sessions user has had

        Returns:
            Session dict with telemetry
        """
        session = {}

        # 1. Keystroke dynamics (realistic human typing)
        keystrokes = []
        for i in range(random.randint(20, 40)):
            dwell = max(30, np.random.normal(user_profile.dwell_mean, user_profile.dwell_std))
            flight = max(20, np.random.normal(user_profile.flight_mean, user_profile.flight_std))
            is_error = np.random.random() < 0.02  # 2% error rate
            keystrokes.append({
                "key": "backspace" if is_error else chr(ord('a') + i % 26),
                "dwell": float(dwell),
                "flight": float(flight),
            })
        session["keystrokes"] = keystrokes

        # 2. Mouse events (natural, smooth movements)
        mouse_events = []
        x, y = 500, 300
        for _ in range(random.randint(10, 30)):
            dx = np.random.normal(0, 20)
            dy = np.random.normal(0, 20)
            x += dx
            y += dy
            x = np.clip(x, 0, 1000)
            y = np.clip(y, 0, 800)
            mouse_events.append({
                "x": float(x),
                "y": float(y),
                "pressure": 0.5 + np.random.normal(0, 0.1),  # Natural pressure variation
            })
        session["mouse_events"] = mouse_events

        # 3. Temporal patterns (realistic timing)
        now = datetime.now()
        # Slight preference for typical hour, but with variance
        login_hour = user_profile.typical_login_hour + np.random.randint(-3, 4)
        login_hour = login_hour % 24
        session["current_time"] = now.replace(hour=login_hour, minute=random.randint(0, 59))

        # Realistic login day (weekday preference)
        days_until_pref = min([
            (d - now.weekday()) % 7
            for d in user_profile.typical_days
        ])
        login_time = session["current_time"] + timedelta(days=days_until_pref)
        session["current_time"] = login_time

        session["session_duration_s"] = float(random.randint(60, 300))  # 1-5 minutes
        session["login_attempts"] = 1  # First attempt (legit users usually succeed on first try)

        # Previous login (1-30 days ago)
        session["last_login_time"] = now - timedelta(days=random.randint(1, 30))

        # 4. Geo info (same location, or known secondary location)
        location = random.choice(user_profile.known_locations)
        session["current_lat"] = location[0] + np.random.normal(0, 0.01)
        session["current_lon"] = location[1] + np.random.normal(0, 0.01)
        session["last_lat"] = location[0] + np.random.normal(0, 0.02)
        session["last_lon"] = location[1] + np.random.normal(0, 0.02)
        session["geo_velocity_kmh"] = float(random.randint(0, 50))  # Local movement

        # 5. Device & network (known/trusted)
        session["known_device"] = True
        session["known_ip_range"] = True
        session["vpn_detected"] = False
        session["tor_exit_node"] = False
        session["ip_reputation_bad"] = False
        session["failed_logins_24h"] = 0

        # 6. Transaction context
        if random.random() < 0.3:  # 30% include transactions
            session["amount_inr"] = float(user_profile.avg_tx_amount_inr * np.random.lognormal(0, 0.3))
            session["new_beneficiary"] = random.random() < 0.1  # 10%
            session["international"] = random.random() < 0.05  # 5%
        else:
            session["amount_inr"] = 0.0
            session["new_beneficiary"] = False
            session["international"] = False

        session["daily_tx_count"] = random.randint(0, 5)
        session["user_avg_amount_inr"] = user_profile.avg_tx_amount_inr
        session["urls_visited"] = ["https://www.sbi.co.in/netbanking/login"]  # Legitimate URL

        return session

    def generate_fraudulent_session(
        self,
        user_profile: UserProfile,
        attack_type: str = "account_takeover",
    ) -> Dict:
        """
        Generate fraudulent session with attack-specific signals.

        Args:
            user_profile: User profile (attacker impersonating this user)
            attack_type: One of:
              - account_takeover: Stolen credentials
              - credential_stuffing: Rapid attempts from same IP
              - sim_swap: Victim changes phone, attacker gets SIM
              - phishing: User tricked to enter credentials on fake site
              - insider_threat: Bank employee with backdoor access

        Returns:
            Session dict with telemetry
        """
        session = {}

        # 1. Keystroke dynamics (bot-like or unfamiliar)
        keystrokes = []
        if attack_type == "account_takeover":
            # Scripted bot typing: too fast, too consistent
            for i in range(random.randint(15, 25)):
                dwell = 60.0 + np.random.normal(0, 2)  # Very fast, very consistent
                flight = 40.0 + np.random.normal(0, 2)
                keystrokes.append({
                    "key": chr(ord('a') + i % 26),
                    "dwell": float(dwell),
                    "flight": float(flight),
                })
        elif attack_type == "sim_swap":
            # Unfamiliar keyboard: slow and irregular
            for i in range(random.randint(20, 40)):
                dwell = 200.0 + np.random.normal(0, 50)
                flight = 150.0 + np.random.normal(0, 40)
                keystrokes.append({
                    "key": chr(ord('a') + i % 26),
                    "dwell": float(dwell),
                    "flight": float(flight),
                })
        else:
            # Other attacks: relatively normal
            for i in range(random.randint(20, 40)):
                dwell = user_profile.dwell_mean + np.random.normal(0, 20)
                flight = user_profile.flight_mean + np.random.normal(0, 15)
                keystrokes.append({
                    "key": chr(ord('a') + i % 26),
                    "dwell": float(dwell),
                    "flight": float(flight),
                })
        session["keystrokes"] = keystrokes

        # 2. Mouse events (jerky, erratic movements)
        mouse_events = []
        x, y = 100, 100
        for _ in range(random.randint(5, 15)):
            # High acceleration, jerky movements
            dx = np.random.normal(0, 50)
            dy = np.random.normal(0, 50)
            x += dx
            y += dy
            x = np.clip(x, 0, 1000)
            y = np.clip(y, 0, 800)
            mouse_events.append({
                "x": float(x),
                "y": float(y),
                "pressure": np.random.uniform(0, 1),  # Inconsistent pressure
            })
        session["mouse_events"] = mouse_events

        # 3. Temporal patterns (suspicious timing)
        now = datetime.now()
        if attack_type == "account_takeover" or attack_type == "credential_stuffing":
            # Off-hours: 2 AM
            login_hour = random.randint(0, 5)
        else:
            login_hour = random.randint(0, 23)

        session["current_time"] = now.replace(hour=login_hour, minute=random.randint(0, 59))
        session["session_duration_s"] = float(random.randint(30, 120))
        session["login_attempts"] = random.randint(2, 8) if attack_type == "credential_stuffing" else 1
        session["last_login_time"] = now - timedelta(hours=random.randint(1, 24))

        # 4. Geo info (impossible velocity or foreign location)
        if attack_type == "sim_swap" or attack_type == "account_takeover":
            # Geo-velocity > 850 km/h (impossible)
            session["current_lat"] = 19.0760  # Mumbai
            session["current_lon"] = 72.8777
            session["last_lat"] = 28.7041  # Delhi
            session["last_lon"] = 77.1025
            session["geo_velocity_kmh"] = float(random.randint(900, 2000))  # Impossible
        else:
            # Unknown location
            session["current_lat"] = float(np.random.uniform(-90, 90))
            session["current_lon"] = float(np.random.uniform(-180, 180))
            session["last_lat"] = user_profile.known_locations[0][0]
            session["last_lon"] = user_profile.known_locations[0][1]
            session["geo_velocity_kmh"] = float(random.randint(100, 300))

        # 5. Device & network (suspicious)
        session["known_device"] = False
        session["known_ip_range"] = False
        if attack_type == "account_takeover":
            session["vpn_detected"] = True
            session["tor_exit_node"] = random.random() < 0.3
        else:
            session["vpn_detected"] = random.random() < 0.3
            session["tor_exit_node"] = False
        session["ip_reputation_bad"] = True
        session["failed_logins_24h"] = random.randint(0, 3) if attack_type == "credential_stuffing" else 0

        # 6. Transaction context (suspicious amounts)
        session["amount_inr"] = float(user_profile.avg_tx_amount_inr * random.randint(8, 12))  # 8-12x avg
        session["new_beneficiary"] = True
        session["international"] = random.random() < 0.7  # 70% international
        session["daily_tx_count"] = random.randint(5, 20)  # Many attempts
        session["user_avg_amount_inr"] = user_profile.avg_tx_amount_inr

        # Phishing URL for phishing attacks
        if attack_type == "phishing":
            session["urls_visited"] = ["http://sbi-secure-verify.click/login"]  # Phishing URL
        else:
            session["urls_visited"] = ["https://www.sbi.co.in/netbanking/login"]

        return session


class SimulationTrainingLoop:
    """Training loop for RL agent on simulated sessions."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        rl_agent: DuelingDQNAgent,
    ):
        """
        Initialize training loop.

        Args:
            feature_extractor: Feature extractor
            rl_agent: RL agent to train
        """
        self.feature_extractor = feature_extractor
        self.rl_agent = rl_agent
        self.session_gen = SessionGenerator(feature_extractor)
        self.metrics = {
            "episode_rewards": [],
            "episode_accuracies": [],
            "episode_losses": [],
        }

    def run(
        self,
        num_episodes: int = 10000,
        fraud_rate: float = 0.30,
        baseline_sessions_per_user: int = 30,
        num_users: int = 10,
        verbose: bool = True,
    ):
        """
        Run training loop.

        Args:
            num_episodes: Number of episodes
            fraud_rate: Fraction of fraudulent sessions (0-1)
            baseline_sessions_per_user: Sessions to build per-user baseline
            num_users: Number of simulated users
            verbose: Print progress?
        """
        # Create user profiles
        users = [
            UserProfile(f"user_{i}", typing_wpm=40 + i * 5)
            for i in range(num_users)
        ]

        print("\n" + "=" * 70)
        print("SIMULATION TRAINING LOOP")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Episodes: {num_episodes}")
        print(f"  Fraud rate: {fraud_rate:.1%}")
        print(f"  Baseline sessions/user: {baseline_sessions_per_user}")
        print(f"  Users: {num_users}")

        # Build baselines for each user
        print(f"\nBuilding baselines ({baseline_sessions_per_user} sessions per user)...")
        for user in users:
            for _ in range(baseline_sessions_per_user):
                legit_session = self.session_gen.generate_legitimate_session(user)
                _, features = self.feature_extractor.extract(user.user_id, legit_session, normalize=True)

        print(f"Baselines built.")

        # Training loop
        print(f"\nRunning {num_episodes} episodes...")
        for episode in range(num_episodes):
            episode_reward = 0.0
            episode_correct = 0
            episode_total = 0
            episode_loss_sum = 0.0

            # Generate batch of sessions for this episode
            batch_size = random.randint(8, 16)
            for _ in range(batch_size):
                # Pick random user
                user = random.choice(users)

                # Decide: legitimate or fraudulent?
                is_fraud = np.random.random() < fraud_rate

                # Generate session
                if is_fraud:
                    attack_types = [
                        "account_takeover",
                        "credential_stuffing",
                        "sim_swap",
                        "phishing",
                        "insider_threat",
                    ]
                    attack_type = random.choice(attack_types)
                    session = self.session_gen.generate_fraudulent_session(user, attack_type)
                else:
                    session = self.session_gen.generate_legitimate_session(user)

                # Extract features
                _, features = self.feature_extractor.extract(user.user_id, session, normalize=True)

                # Get action from agent
                action = self.rl_agent.select_action(features, training=True)

                # Compute reward (ground truth available in simulation)
                reward = DuelingDQNAgent.compute_reward(action, is_fraud)

                # Generate next state (simplified: random next state)
                next_features = self.feature_extractor.extract(user.user_id, session, normalize=True)[1]

                # Store in replay buffer
                self.rl_agent.remember(features, action, reward, next_features, False)

                # Train step
                loss = self.rl_agent.train_step()
                if loss is not None:
                    episode_loss_sum += loss

                # Metrics
                episode_reward += reward
                episode_total += 1

                # Check correctness
                correct_action = 2 if is_fraud else 0  # BLOCK if fraud, ALLOW if legit
                if action == correct_action:
                    episode_correct += 1

            episode_accuracy = episode_correct / max(episode_total, 1)
            avg_loss = episode_loss_sum / max(batch_size, 1)

            self.metrics["episode_rewards"].append(episode_reward)
            self.metrics["episode_accuracies"].append(episode_accuracy)
            self.metrics["episode_losses"].append(avg_loss)

            # Progress logging
            if verbose and (episode + 1) % 1000 == 0:
                print(f"  Episode {episode+1}/{num_episodes}")
                print(f"    Avg Reward: {np.mean(self.metrics['episode_rewards'][-100:]):.2f}")
                print(f"    Avg Accuracy: {np.mean(self.metrics['episode_accuracies'][-100:]):.2%}")
                print(f"    Avg Loss: {np.mean(self.metrics['episode_losses'][-100:]) if self.metrics['episode_losses'] else 0:.4f}")
                print(f"    Epsilon: {self.rl_agent.epsilon:.4f}")

        print(f"\nTraining complete!")
        print(f"Final metrics (last 100 episodes):")
        print(f"  Avg Reward: {np.mean(self.metrics['episode_rewards'][-100:]):.2f}")
        print(f"  Avg Accuracy: {np.mean(self.metrics['episode_accuracies'][-100:]):.2%}")
        print(f"  Avg Loss: {np.mean([l for l in self.metrics['episode_losses'][-100:] if l > 0]) if any(self.metrics['episode_losses'][-100:]) else 0:.4f}")


def demo_simulation():
    """Demonstrate simulation on small scale."""
    print("\n" + "=" * 70)
    print("SIMULATION DEMO (Quick Test - 100 episodes)")
    print("=" * 70)

    from features import FeatureExtractor
    from rl_agent import DuelingDQNAgent

    feature_extractor = FeatureExtractor()
    rl_agent = DuelingDQNAgent()

    training_loop = SimulationTrainingLoop(feature_extractor, rl_agent)
    training_loop.run(
        num_episodes=100,
        fraud_rate=0.3,
        baseline_sessions_per_user=10,
        num_users=3,
        verbose=True,
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_simulation()
