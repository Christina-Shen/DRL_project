#!/usr/bin/env python3
import numpy as np



# define the grid size
size_h = 4
size_w = 4
def grid_world_iterative(decay_gamma,iterations_policy_iter,iterations_policy_eval):
  # define the actions
  actions = np.array(["up", "down", "left", "right"])

  # define the reward for each action (-1 everywhere for all actions,
  # except for the terminal states)
  reward = np.full((size_h, size_w, len(actions)), -1.0)
  reward[0, 0] = np.zeros((4), dtype=np.float32)
  reward[-1, -1] = np.zeros((4), dtype=np.float32)

  # define the pi, now it needs to know the state too
  pi = np.full((size_h, size_w, len(actions)), 0.25)
  # s'|s,a in this problem is deterministic, so I can just define it as a 4x4,
  transfer = np.zeros((size_h, size_w, len(actions), 2), dtype=np.int32)
  for y in range(size_h):
    for x in range(size_w):
      for a in range(len(actions)):
        if actions[a] == "up":
          if y > 0:
            transfer[y, x, a, 0] = y - 1
          else:
            transfer[y, x, a, 0] = y
          transfer[y, x, a, 1] = x
        elif actions[a] == "down":
          if y < size_h - 1:
            transfer[y, x, a, 0] = y + 1
          else:
            transfer[y, x, a, 0] = y
          transfer[y, x, a, 1] = x
        elif actions[a] == "left":
          if x > 0:
            transfer[y, x, a, 1] = x - 1
          else:
            transfer[y, x, a, 1] = x
          transfer[y, x, a, 0] = y
        elif actions[a] == "right":
          if x < size_w - 1:
            transfer[y, x, a, 1] = x + 1
          else:
            transfer[y, x, a, 1] = x
          transfer[y, x, a, 0] = y
  # now fill up the transfer at the end nodes
  transfer[0, 0] = np.zeros((len(actions), 2))
  transfer[-1, -1] = np.full((len(actions), 2), -1)
  # initial value function
  value_0 = np.zeros((size_h, size_w), dtype=np.float32)
  # iterate externally over policy iteration and internally for policy evaluation
  # if this is 1, then we are doing value iteration, and in that case, the intermediate policy is no longer the optimal one for each value function (it didn't converge), only the last policy is! (see value iteration)
  epsilon_policy_eval = 0.0001
  for it_pol_iter in range(iterations_policy_iter):
    for it_pol_eval in range(iterations_policy_eval):
      value_t = np.zeros_like(value_0)
      # do one bellman step in each state
      for y in range(value_0.shape[0]):
        for x in range(value_0.shape[1]):
          for a, action in enumerate(actions):
            # get the coordinates where I go with this action
            newy, newx = transfer[y, x, a]
            # make one lookahead step for this action
            value_t[y, x] += pi[y, x, a] * \
                (reward[y, x, a] + decay_gamma*value_0[newy, newx])
      # if value converged, exit
      norm = 0.0
      for y in range(value_t.shape[0]):
        for x in range(value_t.shape[1]):
          norm += np.abs(value_0[y, x] - value_t[y, x])
          norm /= np.array(value_t.shape, dtype=np.float32).sum()
    # print(norm)
      if norm < epsilon_policy_eval:
        break
      else:
      # if not converged, save current as old to iterate
        value_0 = np.copy(value_t)
   # print("policy eval k: ", it_pol_eval)
    # iterate the policy greedily
    buffered_pi = np.copy(pi)
    for y in range(value_t.shape[0]):
      for x in range(value_t.shape[1]):
        max_v = -float("inf")
        max_v_idx = 0
        for a, action in enumerate(actions):
          # get the coordinates where I go with this action
          newy, newx = transfer[y, x, a]
          # make one lookahead step for this action
          v = reward[y, x, a] + value_t[newy, newx]
          if v > max_v:
            max_v = v
            max_v_idx = a
        # update policy with argmax
        pi[y, x] = np.zeros((len(actions)), dtype=np.float32)
        pi[y, x, max_v_idx] = 1.0



  print("-" * 40)
  print("iterations: ", it_pol_iter + 1)
  print("value:")
  print(value_t)
  print("policy:")
  print(actions[np.argmax(pi, axis=-1)]) 
  return value_t
  
def save_grid(np_grid,file):
  g=np_grid.reshape(16)
  grid=g[1:-1]
  
  array_str = ' '.join(map(str, grid))
  # 写入文件
  with open(file, 'w') as file:
    file.write(array_str)

iterations_policy_iter=100000

iterations_policy_eval=100000
decay_nine_value=grid_world_iterative(0.9,iterations_policy_iter,iterations_policy_eval)
decay_one_value=grid_world_iterative(0.1,iterations_policy_iter,iterations_policy_eval)
file_nine='gamma_0.9.txt'
file_one='gamma_0.1.txt'
save_grid(decay_nine_value,file_nine)
save_grid(decay_one_value,file_one)