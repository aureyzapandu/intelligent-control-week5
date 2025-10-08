import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import gymnasium as gym

# -----------------------
# Environment: Pendulum-v1
# -----------------------
env = gym.make('Pendulum-v1', render_mode="human")  # Tampilkan animasi
state_size = env.observation_space.shape[0]
action_size = 3  # Sama seperti saat training (discretized actions)

# -----------------------
# Load Model
# -----------------------
model_path = r"G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 5\intelligent-control-week5\dqn_pendulum_v1.keras"
model = keras.models.load_model(model_path)
print(f"Model berhasil dimuat dari: {model_path}")

# -----------------------
# Fungsi untuk memilih aksi
# -----------------------
def act(state):
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# -----------------------
# Testing
# -----------------------
test_episodes = 10
max_steps = 200
scores = []

for e in range(test_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for step in range(max_steps):
        # Pilih aksi berdasarkan Q-value (tanpa eksplorasi)
        action = act(state)

        # Mapping discrete action ke continuous torque
        if action == 0:
            torque = -2.0
        elif action == 1:
            torque = 0.0
        else:
            torque = 2.0

        next_state, reward, terminated, truncated, _ = env.step([torque])
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])

        total_reward += reward
        state = next_state

        if done:
            break

    scores.append(total_reward)
    print(f"Episode {e+1}/{test_episodes} - Total Reward: {total_reward:.2f}")

# -----------------------
# Plot hasil testing
# -----------------------
plt.figure(figsize=(8,4))
plt.plot(scores, marker='o', label="Reward per Episode")
plt.title("Hasil Pengujian DQN pada Pendulum-V1")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend()
plt.show()

env.close()
