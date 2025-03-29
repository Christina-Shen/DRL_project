# DRL_project 
here's the project description
# 1. Random policy iterative

![image](https://github.com/user-attachments/assets/ee781485-4e59-4b2e-ab2b-6485f13c8683)

### Training Method
- Using the **iterative + greedy** algorithm to calculate the reward value for each grid
- **Iterative steps**: Initialize policy → Update value based on current policy → Generate new policy based on updated value
- **Policy configuration**: Each grid records 4 values representing weights of actions (up, down, left, right)
- **Value update steps**:
  - Based on the following formula:
  
  ![Value Update Formula](https://latex.codecogs.com/png.latex?V_{k+1}(s)%20\leftarrow%20\mathbb{E}[R_t%20+%20\gamma%20V_k(S_{t+1})|S_t%20=%20s])
  
  ![Expanded Formula](https://latex.codecogs.com/png.latex?=%20\sum_a%20\pi(a|s)%20\sum_{s'}%20p(s'|s,a)[R_t%20+%20\gamma%20V_k(s')])

- Due to the greedy algorithm, except for the first iteration, subsequent iterations select only one value to update based on the previous state and action
### Results
- For decay_gamma = 0.9: -1.0 -1.9 -2.71 -1.0 -1.9 -2.71 -1.9 -1.9 -2.71 -1.9 -1.0 -2.71 -1.9 -1.0
- For decay_gamma = 0.1: -1.0 -1.1 -1.1108187 -1.0 -1.1 -1.1108187 -1.1 -1.1 -1.1108187 -1.1 -1.0 -1.1108187 -1.1 -1.0
- Values converge to the center: Terminal states are set with center symmetric conditions. With identical initial values and weights, when policy iterates, the primary direction also points to the center, causing values to converge toward the center.
- Differences between decay_gamma = 0.9 and decay_gamma = 0.1: With smaller decay gamma, final converged absolute values are smaller. This is because during value updates, weights for other direction values are smaller. Even though all grids initially have value 0, since rewards in all four directions are -1, each grid value becomes negative during iteration. With larger gamma_decay, weights for other values increase more, resulting in larger absolute converged values.

# 2. using gym taxi-v3 compare Q-Learning with SARSA
### Environment description
- Observation space has 500 states: taxi row position (25) × passenger location (4) × passenger on taxi (2) × destination location (4)
- Observation encoding: s = ((taxi_row*5+taxi_col)*5+pass_loc)*4+dest_idx (see GitHub source code)
- Actions include 6 options: move east/west/north/south (0-3), pickup passenger (4), dropoff passenger (5)
- Rewards: movement: -1, successful dropoff: +20, wrong dropoff location: -10

### Compare Q-Learning excute result with sarsa

- Q-table dimensions: 500 (observation states) × 6 (actions)
- Iterative process updates action and Q-table; each iteration involves repeating until passenger is successfully delivered:
  - Epsilon greedy (epislon) action selection
  - Action updates the environment state
  - Environment state updates Q-table
  - Epsilon decreases
- Epsilon greedy action selection:
  - Generate random number 0-1; if less than epsilon, select random action (higher epsilon = more exploration)
  - Initially, Q-table is not established, so epsilon starts high for more random selection
  - As episodes increase, epsilon decreases, actions follow Q-table values more closely
- Q-table update formula:
  
  ![Q-Learning Formula](https://latex.codecogs.com/png.latex?Q(S,A)%20\leftarrow%20Q(S,A)%20+%20\alpha%20[R%20+%20\gamma%20\max_a%20Q(S',a)%20-%20Q(S,A)])
  
  During each update, using max of Q(s') to indicate the best next action's Q value for the updated state
#### SARSA
- Training method same as Q-learning, except Q value update formula:
  
  ![SARSA Formula](https://latex.codecogs.com/png.latex?Q(S,A)%20\leftarrow%20Q(S,A)%20+%20\alpha%20[R%20+%20Q(S',A')%20-%20Q(S,A)])
  
  Where Q(S',A') is the Q-table value for the next state and the action determined for that state

### Results
![image](https://github.com/user-attachments/assets/a707ee0c-4231-40ec-98fa-421d1607faf7)

![image](https://github.com/user-attachments/assets/c08ddded-d155-4f6e-b3f7-f6948a6da49a)

- Repeated Q-learning results are averaged, showing smoother outcomes
- Comparing Q-learning and SARSA reward/episode trends: Q-learning results are more stable, especially in the second half of episodes
- This stability is likely because Q-learning updates using the maximum expected Q-value for the next state, independent of the actual next action. This means it considers the best possible strategy regardless of the action taken, making it more effective in learning optimal policies.

#  3. 2D Tic-tac-toe (3x3)
![image](https://github.com/user-attachments/assets/c807fa5c-2bce-4611-8474-26d3098e742c)

### Training Method
- Q-learning approach: Each position's Q-table defines state as player symbol + board state for each board position's action
- Iterative + greedy method updates each agent's Q-table:
  - Each iteration repeats until win/tie:
    - p1 selects action based on current board where Q-table value is maximized
    - Update board
    - Add current state and action to p1's recorded states
    - Check for win/tie -> give reward to p1,p2
    - p2 selects action based on current board where Q-table value is maximized
    - Check for win/tie -> give reward to p1,p2
    - Add current state and action to p2's recorded states
  - p1,p2 Q-table update formula:
    - Rewards generated only with win/tie: win = +1, lose = 0, tie = 0.5
    - Update p1,p2 Q-table according to reward using Q-learning method
    - For recorded states: final state is terminal, Q-table->reward; other states (1-learning_rate)Q-table + learning_rate * decay * maximum value in Q-table for that state
### Results
- When both are Q-learning agents (p1 first): p1 win: 58.29%, p1 tie: 12.28%
- When both are Q-learning agents (p2 first): p1 win: 58.69%, p1 tie: 12.89%
- When using trained policy p2 against random agent p3: p2 win: 59.9%, p2 tie: 13.1%

## 4. 3D Tic-tac-toe (4x4x4)
![image](https://github.com/user-attachments/assets/bb7829c6-bc9e-479e-9166-4ed74de05d73)
![image](https://github.com/user-attachments/assets/566992e8-1755-4059-b0ab-4c8f33f2f5ea)

### Training Method
- Using MCNT method: each player (p1,p2) records state value
- 4×4×4 grid, considering winners in each row, column, layer (4 planes), and four diagonal lines
- Using iterative MCNT to update state values:
  - Each iteration with epsilon greedy repeats until win/tie:
    - p1 selects best action based on state value
    - Board updates based on action
    - p1 records current board
    - Check win/tie -> give reward to p1,p2
    - p2 selects best action based on state value
    - Board updates based on action
    - p2 records current board
  - p1,p2 action selection:
    - Based on current symbol turn, generating new state from position, selecting action with highest state value
    - Generate random value; if less than epsilon, use random action (epsilon decreases with episodes)
  - p1,p2 state value updates:
    - Rewards only generated with win/tie: win = +1, lose = 0, tie = 0.5
    - State values updated after each recorded state using MCNT formula:
    
    ![MCNT Formula](https://latex.codecogs.com/png.latex?V(S_t)%20\leftarrow%20V(S_t)%20+%20\frac{1}{N(S_t)}(G_t%20-%20V(S_t)))
    
    - Where G_t represents future feedback, corresponding to reward * gamma^(number of steps from current to end)
### Results
- p1 wins: 51.0%, p1 ties: 0.0%
