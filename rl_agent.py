"""
LAYER 3: Reinforcement Learning Agent
Dueling Double DQN for adaptive authentication decisions.

State: 32-D feature vector
Actions: 0=ALLOW, 1=STEP_UP, 2=BLOCK

Architecture:
  Shared: Linear(32→128) + LayerNorm + ReLU → Linear(128→128) + LayerNorm + ReLU
  Value stream: Linear(128→64) + ReLU → Linear(64→1)
  Advantage stream: Linear(128→64) + ReLU → Linear(64→3)
  Output: V(s) + A(s,a) - mean(A(s,·))  [Dueling combination]

Asymmetric reward shaping:
  ALLOW on legit: +10
  ALLOW on fraud: -20 (false negative, very costly)
  BLOCK on fraud: +10
  BLOCK on legit: -15 (false positive, UX damage)
  STEP_UP → legit passes: +5
  STEP_UP → fraud fails: +8
  STEP_UP → legit fails: -12
  STEP_UP → fraud passes: -10
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
from collections import deque
import json
import os

# Try to import PyTorch, fallback to numpy
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DuelingDQNNetwork(nn.Module):
    """
    Dueling Double DQN network architecture (PyTorch).

    Separates value and advantage streams, then combines them:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    """

    def __init__(self, state_dim: int = 32, action_dim: int = 3, hidden_dim: int = 128):
        """
        Initialize network.

        Args:
            state_dim: Input feature dimension (32)
            action_dim: Number of actions (3: ALLOW, STEP_UP, BLOCK)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Value stream: outputs scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Advantage stream: outputs A(s,a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute Q-values for all actions.

        Args:
            state: Tensor of shape (batch_size, 32) or (32,)

        Returns:
            Q-values of shape (batch_size, 3) or (3,)
        """
        # Handle both batched and single inputs
        if state.dim() == 1:
            state = state.unsqueeze(0)

        shared_out = self.shared(state)
        value = self.value_stream(shared_out)
        advantage = self.advantage_stream(shared_out)

        # Dueling combination: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # Remove batch dim if input was single state
        if q_values.shape[0] == 1 and state.shape[0] == 1:
            q_values = q_values.squeeze(0)

        return q_values


class ExperienceReplayBuffer:
    """Experience replay buffer for off-policy RL training."""

    def __init__(self, capacity: int = 50000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DuelingDQNAgent:
    """
    Dueling Double DQN RL Agent for adaptive authentication.

    Actions:
      0 = ALLOW (proceed normally)
      1 = STEP_UP (request OTP/biometric)
      2 = BLOCK (terminate session)
    """

    # Reward shaping (asymmetric - real banking costs)
    REWARD_ALLOW_LEGIT = 10.0
    REWARD_ALLOW_FRAUD = -20.0  # False negative = very costly
    REWARD_BLOCK_FRAUD = 10.0
    REWARD_BLOCK_LEGIT = -15.0  # False positive = UX damage
    REWARD_STEP_UP_LEGIT_PASS = 5.0
    REWARD_STEP_UP_FRAUD_FAIL = 8.0
    REWARD_STEP_UP_LEGIT_FAIL = -12.0
    REWARD_STEP_UP_FRAUD_PASS = -10.0

    ACTION_NAMES = {0: "ALLOW", 1: "STEP_UP", 2: "BLOCK"}

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 3,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 2000,
        batch_size: int = 64,
        update_target_freq: int = 100,
        buffer_capacity: int = 50000,
    ):
        """
        Initialize Dueling DQN agent.

        Args:
            state_dim: State dimension (32)
            action_dim: Number of actions (3)
            hidden_dim: Hidden layer dimension
            learning_rate: Adam learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon
            batch_size: Replay buffer batch size
            update_target_freq: Steps between target network updates
            buffer_capacity: Replay buffer capacity
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.training_steps = 0

        self.replay_buffer = ExperienceReplayBuffer(buffer_capacity)

        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Policy and target networks
            self.policy_net = DuelingDQNNetwork(state_dim, action_dim, hidden_dim).to(
                self.device
            )
            self.target_net = DuelingDQNNetwork(state_dim, action_dim, hidden_dim).to(
                self.device
            )
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()
        else:
            # NumPy fallback: simple tabular Q-learning
            self.q_table = np.zeros((state_dim, action_dim), dtype=np.float32)

        self.metrics = {
            "episode_rewards": [],
            "losses": [],
            "q_values": [],
        }

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: 32-D feature vector
            training: If True, use exploration; if False, greedy

        Returns:
            Action (0, 1, or 2)
        """
        # Decay epsilon
        progress = min(1.0, self.training_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start - (
            self.epsilon_start - self.epsilon_end
        ) * progress

        # Epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)

        if HAS_TORCH:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().to(self.device)
                q_values = self.policy_net(state_tensor)
                return int(torch.argmax(q_values).item())
        else:
            # NumPy fallback: simple lookup (won't work well, just for demo)
            return int(np.argmax(np.dot(state, self.q_table)))

    def predict(self, state: np.ndarray) -> Dict:
        """
        Greedy inference: return action, confidence, and Q-value info.

        Args:
            state: 32-D feature vector

        Returns:
            Dict with keys:
              - action: Action (0, 1, 2)
              - action_name: "ALLOW", "STEP_UP", "BLOCK"
              - confidence: 0.0-1.0
              - q_values: Q-values for all actions
              - best_q: Best Q-value
        """
        if HAS_TORCH:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().to(self.device)
                q_values = self.policy_net(state_tensor)
                q_numpy = q_values.cpu().numpy()

                best_action = int(np.argmax(q_numpy))
                best_q = float(np.max(q_numpy))

                # Confidence: softmax over Q-values
                q_softmax = np.exp(q_numpy - np.max(q_numpy))
                q_softmax /= q_softmax.sum()
                confidence = float(q_softmax[best_action])

                return {
                    "action": best_action,
                    "action_name": self.ACTION_NAMES[best_action],
                    "confidence": confidence,
                    "q_values": {
                        "ALLOW": float(q_numpy[0]),
                        "STEP_UP": float(q_numpy[1]),
                        "BLOCK": float(q_numpy[2]),
                    },
                    "best_q": best_q,
                }
        else:
            # NumPy fallback
            best_action = 0
            return {
                "action": best_action,
                "action_name": self.ACTION_NAMES[best_action],
                "confidence": 0.5,
                "q_values": {"ALLOW": 0.0, "STEP_UP": 0.0, "BLOCK": 0.0},
                "best_q": 0.0,
            }

    def train_step(self) -> Optional[float]:
        """
        Execute one training step (Double DQN with Dueling architecture).

        Returns:
            Loss value, or None if not enough buffer samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        if not HAS_TORCH:
            return None  # Can't train without PyTorch in this implementation

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        # Current Q-values
        q_pred = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: action selection from policy net, value from target net
        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(next_states_t), dim=1)
            next_q_values = self.target_net(next_states_t).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards_t + (self.gamma * next_q_values * (1 - dones_t))

        # MSE loss
        loss = self.criterion(q_pred, target_q)

        # Backprop with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Store transition in replay buffer.

        Args:
            state: 32-D feature vector
            action: Action taken (0, 1, 2)
            reward: Reward received
            next_state: 32-D feature vector of next state
            done: Episode done?
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    @staticmethod
    def compute_reward(
        action: int,
        is_fraudulent: bool,
        step_up_passed: bool = None,
    ) -> float:
        """
        Compute reward for an action based on ground truth label.

        Args:
            action: Action taken (0=ALLOW, 1=STEP_UP, 2=BLOCK)
            is_fraudulent: Is the session actually fraudulent?
            step_up_passed: For STEP_UP, did user pass the challenge? (None if not STEP_UP)

        Returns:
            Reward value
        """
        if action == 0:  # ALLOW
            return (
                DuelingDQNAgent.REWARD_ALLOW_FRAUD
                if is_fraudulent
                else DuelingDQNAgent.REWARD_ALLOW_LEGIT
            )
        elif action == 1:  # STEP_UP
            if step_up_passed is None:
                step_up_passed = not is_fraudulent  # Assume legit users pass
            if is_fraudulent:
                return (
                    DuelingDQNAgent.REWARD_STEP_UP_FRAUD_PASS
                    if step_up_passed
                    else DuelingDQNAgent.REWARD_STEP_UP_FRAUD_FAIL
                )
            else:
                return (
                    DuelingDQNAgent.REWARD_STEP_UP_LEGIT_PASS
                    if step_up_passed
                    else DuelingDQNAgent.REWARD_STEP_UP_LEGIT_FAIL
                )
        elif action == 2:  # BLOCK
            return (
                DuelingDQNAgent.REWARD_BLOCK_FRAUD
                if is_fraudulent
                else DuelingDQNAgent.REWARD_BLOCK_LEGIT
            )
        return 0.0

    def save(self, path: str):
        """
        Save model to disk.

        Args:
            path: File path to save to
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if HAS_TORCH:
            torch.save(
                {
                    "policy_state_dict": self.policy_net.state_dict(),
                    "target_state_dict": self.target_net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "training_steps": self.training_steps,
                    "epsilon": self.epsilon,
                },
                path,
            )
        else:
            # NumPy fallback
            np.save(path, self.q_table)

    def load(self, path: str):
        """
        Load model from disk.

        Args:
            path: File path to load from
        """
        if HAS_TORCH:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.training_steps = checkpoint["training_steps"]
            self.epsilon = checkpoint["epsilon"]
        else:
            self.q_table = np.load(path)


def demo_rl_agent():
    """Demonstrate RL agent on dummy states."""
    print("\n" + "=" * 70)
    print("LAYER 3: RL AGENT (Dueling Double DQN) DEMO")
    print("=" * 70)

    if not HAS_TORCH:
        print("  (PyTorch not available - using NumPy fallback)")

    agent = DuelingDQNAgent()

    print("\n[1] Testing action selection (epsilon-greedy exploration)")
    dummy_state = np.random.randn(32).astype(np.float32)
    for i in range(5):
        action = agent.select_action(dummy_state, training=True)
        print(f"  Step {i+1}: epsilon={agent.epsilon:.3f}, action={agent.ACTION_NAMES[action]}")

    print("\n[2] Testing greedy inference (no exploration)")
    prediction = agent.predict(dummy_state)
    print(f"  Action: {prediction['action_name']}")
    print(f"  Confidence: {prediction['confidence']:.3f}")
    print(f"  Q-values: ALLOW={prediction['q_values']['ALLOW']:.3f}, "
          f"STEP_UP={prediction['q_values']['STEP_UP']:.3f}, "
          f"BLOCK={prediction['q_values']['BLOCK']:.3f}")

    print("\n[3] Testing reward computation")
    print(f"  ALLOW + legit: {DuelingDQNAgent.compute_reward(0, False)}")
    print(f"  ALLOW + fraud: {DuelingDQNAgent.compute_reward(0, True)}")
    print(f"  BLOCK + fraud: {DuelingDQNAgent.compute_reward(2, True)}")
    print(f"  BLOCK + legit: {DuelingDQNAgent.compute_reward(2, False)}")
    print(f"  STEP_UP + legit + pass: {DuelingDQNAgent.compute_reward(1, False, True)}")
    print(f"  STEP_UP + fraud + fail: {DuelingDQNAgent.compute_reward(1, True, False)}")

    print("\n[4] Testing training loop (if PyTorch available)")
    if HAS_TORCH:
        for step in range(200):
            # Add random experiences
            for _ in range(4):
                s = np.random.randn(32).astype(np.float32)
                a = np.random.randint(0, 3)
                r = np.random.randn()
                s_next = np.random.randn(32).astype(np.float32)
                done = np.random.random() < 0.1
                agent.remember(s, a, r, s_next, done)

            loss = agent.train_step()
            if loss is not None and (step + 1) % 50 == 0:
                print(f"  Step {step+1}: loss={loss:.4f}, epsilon={agent.epsilon:.4f}")
    else:
        print("  (Skipped - PyTorch required for training)")

    print("\n[5] Testing save/load")
    agent.save("rl_agent_test.pt")
    print(f"  Saved to rl_agent_test.pt")
    if os.path.exists("rl_agent_test.pt"):
        agent2 = DuelingDQNAgent()
        agent2.load("rl_agent_test.pt")
        print(f"  Loaded from rl_agent_test.pt")
        pred2 = agent2.predict(dummy_state)
        print(f"  Restored agent prediction: {pred2['action_name']}")
        os.remove("rl_agent_test.pt")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_rl_agent()
