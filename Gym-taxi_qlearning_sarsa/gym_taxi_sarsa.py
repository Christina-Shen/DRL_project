import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import warnings
import math
warnings.simplefilter("ignore")
# del Q_table
# del Reward_Sarsa,Step_size_Sarsa,Reward_Sarsa_rp, Step_size_Sarsa_rp
#@title Functions 
import pickle
import torch


filename = f"taxi_sarsa.pth"

env = gym.make("Taxi-v3").env

def Result_showing(Reward, Step_size):
  Avg_Reward = Reward / Step_size
  plt.figure(figsize=(16, 10)) 
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
def epsilon_greedy(a, epsilon):
    n = len(a)
    if random.random() < epsilon:
        # 以ε的概率随机选择一个行为
        return random.randint(0, n-1)
    else:
        # 以1-ε的概率选择当前最优的行为
        return np.argmax(a)
def Q_update(reward, Q, Q_ns_na, gamma, alpha):
  # Q_ns_na is Q(S', A') in Sarsa, not the max Q as in Q-Learning
  new_Q = Q + alpha * (reward + gamma * Q_ns_na - Q)
  return new_Q
def get_action(a,t,episode):
#     print get_explore_rate(t)
    if episode>round(EPISODE_NUM *0.5):
        return np.argmax(a)
    r=np.random.random()
    if r<max(0.01, min(0.1, 1.0 - math.log10((t+1)/500.))) :
        return env.action_space.sample()
    else:
        return np.argmax(a)
    

def Sarsa(EPISODE_NUM, gamma, alpha, epsilon):
  Q_table = np.zeros([500, 6])
  Reward = np.zeros(EPISODE_NUM)
  Step_size = np.zeros(EPISODE_NUM)
  for episode in range(EPISODE_NUM):
    s = env.reset()[0]
    #print(f"Type of s: {type(s)}, Value of s: {s}") 
    while True:
      # epsilon greedy
      #a = epsilon_greedy(Q_table[s], epsilon)
      a=get_action(Q_table[s],Step_size[episode],episode)
      #a=epsilon_greedy(Q_table[s], epsilon)
      # next state 
      next_state, reward, done, _,info = env.step(a)
      Reward[episode] += reward
      Step_size[episode] += 1
      
      # update Q(s,a)
      Q_table[s,a] = Q_update(reward, Q_table[s,a], Q_table[next_state,a], gamma, alpha)
      s = next_state
      #if Step_size[episode]>100000:
          # print(episode," can't converge")
          # break
      if (done == 1): break
    #epsilon = epsilon - 2/EPISODE_NUM if epsilon > 0.01 else 0.01
    epsilon*=0.99
    #alpha *= 0.99 if alpha > 0.01 else 0.01
    #alpha *= 0.999
  return Q_table, Reward, Step_size
def Repeated_Sarsa(EPISODE_NUM, gamma, alpha, Repeat_Num, epsilon):
  Q_table = np.zeros([500, 6])
  Reward = np.zeros(EPISODE_NUM)
  Step_size = np.zeros(EPISODE_NUM)
  for i in range(Repeat_Num):
    Q_table_, Reward_, Step_size_ = Sarsa(EPISODE_NUM, gamma, alpha, epsilon)
    Q_table += Q_table_ / Repeat_Num
    Reward += Reward_ / Repeat_Num
    Step_size += Step_size_ / Repeat_Num
  return Q_table, Reward, Step_size

def save_Q_table(Q_table):
    fw = open('taxi_sarsa', 'wb')
    pickle.dump(Q_table, fw)
    fw.close()



#using Q-learning training:
EPISODE_NUM = 1000
Q_table, Reward_Sarsa, Step_size_Sarsa = Sarsa(EPISODE_NUM, gamma = 0.99, alpha = 1, epsilon = 1)
Result_showing(Reward_Sarsa, Step_size_Sarsa)
plt.suptitle('Results using Sarsa', fontsize = 15)
plt.tight_layout()
plt.show()
avg_reward_Sarsa= pd.Series(Reward_Sarsa[round(EPISODE_NUM*4/5):]) / pd.Series(Step_size_Sarsa[round(EPISODE_NUM*4/5):])
#print("\n Sarsa average reward after 4/5 episode >0=",((avg_reward_Sarsa > 0).sum())*100/round(EPISODE_NUM/5),"%\n")

print("\n Sarsa  reward after 4/5 episode >0=",((avg_reward_Sarsa > 0).sum())*100/round(EPISODE_NUM/5),"%\n")




#@title Results using Improved Q-learning
EPISODE_NUM = 1000
Q_table_rp, Reward_Sarsa_rp, Step_size_Sarsa_rp = Repeated_Sarsa(EPISODE_NUM, gamma = 0.99, alpha = 1, Repeat_Num = 20, epsilon = 1)
Result_showing(Reward_Sarsa_rp, Step_size_Sarsa_rp)
plt.suptitle('Results using Sarsa with repeat every episode', fontsize = 15)
plt.tight_layout()
plt.show()
avg_reward_Sarsa_rp= pd.Series(Reward_Sarsa_rp[round(EPISODE_NUM*4/5):]) / pd.Series(Step_size_Sarsa_rp[round(EPISODE_NUM*4/5):])
print("\nrepeat Sarsa average reward after 4/5 episode >0=",(avg_reward_Sarsa_rp > 0).sum()*100/round(EPISODE_NUM/5),"%\n")

save_Q_table(Q_table)