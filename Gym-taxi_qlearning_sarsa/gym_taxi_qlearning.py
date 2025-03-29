import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import pickle
import warnings
warnings.simplefilter("ignore")


#@title Functions 
def RandomBased(EPISODE_NUM):
  env = gym.make("Taxi-v3").env
  Reward = np.zeros(EPISODE_NUM)
  Step_size = np.zeros(EPISODE_NUM)
  for episode in range(EPISODE_NUM):
    env.reset()
    while True:
      action = env.action_space.sample()
      state, reward, done, info, _ = env.step(action)
      #state, reward, done, truncated, info
      Reward[episode] += reward
      Step_size[episode] += 1
      if done: break
  return Reward, Step_size

def Result_showing(EPISODE_NUM, Reward, Step_size):
  window_size = 20
  Avg_Reward = Reward / Step_size
  #plt.figure(figsize=(12, 15)) 
  plt.subplot(3, 1, 1)
  plt.plot(Reward)
  plt.xlabel('episode')
  plt.ylabel('Reward')
  plt.title('Reward per episode')
  plt.subplot(3, 1, 2)
  plt.plot(Step_size)
  plt.xlabel('episode')
  plt.ylabel('Step size')
  plt.title('Step size per episode')
  plt.subplot(3, 1, 3)
  plt.plot(Avg_Reward)
  plt.xlabel('episode')
  plt.ylabel('Average Reward')
  plt.title('Average Reward (Reward / Step-size) per episode')
  plt.tight_layout()
  plt.show()
def epsilon_greedy(a, epsilon):
    n = len(a)
    if random.random() < epsilon:
        # 以ε的概率随机选择一个行为
        return random.randint(0, n-1)
    else:
        # 以1-ε的概率选择当前最优的行为
        return np.argmax(a)

def Q_update(ps, a, ns, reward, Q, Q_ns, gamma, alpha):
  new_Q = Q + alpha * (reward + gamma * np.max(Q_ns)  - Q)
  return new_Q

def Q_Learning(EPISODE_NUM, gamma, alpha, epsilon):
  Q_table = np.zeros([500, 6])
  env = gym.make("Taxi-v3").env
  Reward = np.zeros(EPISODE_NUM)
  Step_size = np.zeros(EPISODE_NUM)
  for episode in range(EPISODE_NUM):
    s = env.reset()[0]
    #print(f"Type of s: {type(s)}, Value of s: {s}") 
    while True:
      # epsilon greedy
      a = epsilon_greedy(Q_table[s], epsilon)
      # next state 
      next_state, reward, done, _,info = env.step(a)
      Reward[episode] += reward
      Step_size[episode] += 1

      # update Q(s,a)
      Q_table[s,a] = Q_update(s, a, next_state, reward, Q_table[s,a], Q_table[next_state], gamma, alpha)
      s = next_state
      if (done == 1): break
    epsilon *= 0.9
    alpha *= 0.999
  return Q_table, Reward, Step_size

def Repeated_Q_Learning(EPISODE_NUM, gamma, alpha, Repeat_Num, epsilon):
  Q_table = np.zeros([500, 6])
  Reward = np.zeros(EPISODE_NUM)
  Step_size = np.zeros(EPISODE_NUM)
  for i in range(Repeat_Num):
    Q_table_, Reward_, Step_size_ = Q_Learning(EPISODE_NUM, gamma, alpha, epsilon)
    Q_table += Q_table_ / Repeat_Num
    Reward += Reward_ / Repeat_Num
    Step_size += Step_size_ / Repeat_Num
  return Q_table, Reward, Step_size

def save_Q_table(Q_table):
    fw = open('taxi_qlearning', 'wb')
    pickle.dump(Q_table, fw)
    fw.close()

#using Q-learning training:
EPISODE_NUM = 1000
Q_table, Reward_QL, Step_size_QL = Q_Learning(EPISODE_NUM, gamma = 0.9, alpha = 1, epsilon = 1)
Result_showing(EPISODE_NUM, Reward_QL, Step_size_QL)
plt.suptitle('Results using Q-learning', fontsize = 15)
avg_reward_rand= pd.Series(Reward_QL[round(EPISODE_NUM/2):]) / pd.Series(Step_size_QL[round(EPISODE_NUM/2):])
print("\n q learning average reward after half >0=",((avg_reward_rand > 0).sum())*100/round(EPISODE_NUM/2),"%\n")




#@title Results using Improved Q-learning
EPISODE_NUM = 1000
Q_table, Reward_QL_rp, Step_size_QL_rp = Repeated_Q_Learning(EPISODE_NUM, gamma = 0.9, alpha = 1, Repeat_Num = 20, epsilon = 1)
Result_showing(EPISODE_NUM, Reward_QL_rp, Step_size_QL_rp)
plt.suptitle('Results using Q-learning', fontsize = 15)
avg_reward_rand_rp= pd.Series(Reward_QL_rp[round(EPISODE_NUM/2):]) / pd.Series(Step_size_QL_rp[round(EPISODE_NUM/2):])
print("\nrepeat q learning average reward after half >0=",(avg_reward_rand_rp > 0).sum()*100/round(EPISODE_NUM/2),"%\n")

save_Q_table(Q_table)
