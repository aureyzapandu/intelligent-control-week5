import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt

# Inisialisasi lingkungan OpenAI Gym (misalnya, CartPole)
env = gym.make("CartPole-v1")

# Parameter DRL
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)

# Membangun model Deep Q-Network (DQN)
model = keras.Sequential([
    keras.layers.Dense(24, input_shape=(state_size,), activation="relu"),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(action_size, activation="linear")
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

# Fungsi memilih aksi berdasarkan eksplorasi dan eksploitasi
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)  # Eksplorasi
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])  # Eksploitasi

# Proses training
episodes = 1000  
scores = []  # simpan skor tiap episode

for episode in range(episodes):
    state, _ = env.reset()  # versi terbaru Gym return (obs, info)
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(500):
        # Pilih aksi
        action = select_action(state, epsilon)

        # Lakukan aksi
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated  # gabungkan kondisi done
        next_state = np.reshape(next_state, [1, state_size])

        # Simpan ke memory
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {episode+1}/{episodes}, Score: {time+1}, Epsilon: {epsilon:.2f}")
            scores.append(time+1)  # simpan skor episode
            break

    # Update jaringan saraf (Training)
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += gamma * np.amax(model.predict(next_state, verbose=0)[0])
            target_f = model.predict(state, verbose=0)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)

    # Kurangi epsilon (exploration decay)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Simpan model hasil training
model.save("dqn_cartpole.keras")
print("Training selesai! Model disimpan sebagai dqn_cartpole.keras")

# Plot grafik skor per episode
plt.figure(figsize=(10,5))
plt.plot(scores, label="Score per Episode")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Perkembangan Training DQN pada CartPole-v1")
plt.legend()
plt.grid(True)
plt.show()
