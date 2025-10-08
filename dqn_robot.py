import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tensorflow import keras
import gymnasium as gym

# -----------------------
# Environment Pendulum-V1 (dari Gymnasium)
# -----------------------
# Pendulum-V1 adalah inverted pendulum yang harus diayunkan ke posisi upright (180 derajat).
# Action space continuous (-2 to 2 torque), tapi kita discretisasi menjadi 3 actions untuk DQN:
# 0: torque = -2.0 (putar kiri), 1: torque = 0.0 (tidak ada aksi), 2: torque = 2.0 (putar kanan)

env = gym.make('Pendulum-v1')
state_size = env.observation_space.shape[0]  # 3: [cos(theta), sin(theta), theta_dot]
action_size = 3  # Discrete actions: 0 (left), 1 (no action), 2 (right)

# -----------------------
# DQN Agent (adaptasi untuk state size 3 dan discrete actions)
# -----------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation="relu"),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model = keras.models.load_model(name)

# -----------------------
# Training
# -----------------------
agent = DQNAgent(state_size, action_size)

episodes = 2000
scores = []
max_steps = 200  # Limit steps per episode (mirip custom env asli)

for e in range(episodes):
    state, _ = env.reset()  # Gymnasium returns state and info
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(max_steps):
        action = agent.act(state)
        
        # Map discrete action ke continuous torque untuk Pendulum-V1
        if action == 0:
            torque = -2.0
        elif action == 1:
            torque = 0.0
        else:  # action == 2
            torque = 2.0
        
        next_state, reward, terminated, truncated, _ = env.step([torque])  # Step dengan action continuous
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if done:
            break

    agent.replay()
    scores.append(total_reward)
    print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

# -----------------------
# Save Model
# -----------------------
save_path = r"G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 5\intelligent-control-week5\dqn_pendulum_v1.keras"
agent.model.save(save_path)
print(f"Training selesai! Model disimpan di {save_path}")

# -----------------------
# Plot hasil training
# -----------------------
window = 20
moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10,5))
plt.plot(scores, label="Reward per Episode", alpha=0.5)
plt.plot(range(window-1, len(moving_avg)+window-1), moving_avg, color="red", label=f"Moving Avg ({window} eps)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training DQN - Pengendalian Pendulum-V1 (Robot Lengan Balancing)")
plt.legend()
plt.grid(True)
plt.show()

# Close environment
env.close()