import gym
import numpy as np 
env = gym.make("FrozenLake-v0")

Q = np.zeros([env.observation_space.n, env.action_space.n])
print(Q.shape)

lr = 0.98
y = 0.95
num_episodes = 2000
num_ones = 0
best_rate = 0
for lr in np.arange(0.0, 1.0, 0.02):
    rewardsList = []
    for i in range(num_episodes):
        s = env.reset()
        totalRewards = 0
        d = False
        j = 0

        while j < 99:
            j += 1 
            a = np.argmax(Q[s,:] + np.random.rand(1, env.action_space.n) * (1./(i+1)))
            new_state, reward, d, _ = env.step(a)
            Q[s,a] = Q[s,a] + lr*(reward + y*np.max(Q[new_state,:]) - Q[s,a])
            totalRewards += reward
            s = new_state
            if d == True:
                break

        rewardsList.append(totalRewards)
        current_ones = rewardsList.count(1.0)
        if current_ones > num_ones:
            best_rate = lr
            num_ones = current_ones
# print(rewardsList.count(1.0))

print(best_rate, num_ones)