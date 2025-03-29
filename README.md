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
- 
# 5.Deep Q-Network Implementation and Optimization Techniques in Gym-supermarion

![image](https://github.com/user-attachments/assets/0b31dfab-91a0-41e0-b678-1b483418a289)

### Double DQN Implementation

- **Double DQN Architecture**:
  ![image](https://github.com/user-attachments/assets/d60be0f6-76fe-4973-a476-5fddb0acce69)

- **Policy/Target Network Update Method** based on the following formulas:

  ![Formula: y_t = r_t + γQ_θ'(s_{t+1}, arg max_{a'} Q_θ(s_{t+1}, a'))]

  ![Formula: θ = θ - α \frac{1}{N} \sum \nabla_θQ(s_i, a_i)(y_i - Q(s_i, a_i))]

  - Q_θ' is the target network, Q_θ is the policy network
  - Action selection (`arg max`) is calculated using the latest state for all actions to find which action has the highest Q value ('a')
  - State(t+1) is input to target network + reward becomes TD_target
  - TD error is defined as TD_target - Q(s_t, a_t), using TD error to update target network and policy network parameters

- Experiments show Double DQN results have significant improvement, but may change depending on reward settings and prioritized memory implementations

 ### Prioritized Memory Optimization

- **Standard Implementation**: When storing actions in memory, training randomly selects batch size data to update Q-network
- **Prioritized Memory**: Uses normalized TD_error results. When randomly selecting batch size data, higher TD_error samples have higher selection probability, prioritizing parameter updates for states with larger TD targets

### Environment Wrapper Methods

- Main functionality divided into the following operations:
  - Using Gym's PyWrappers JoypadSpace module to simplify environment and action space
  - Downscaling environment pixel count by converting RGB information to grayscale and reducing environment visual field
  - Skip multiple frames or steps (frame skip) - representing one training action with multiple game steps
  - Normalizing environment reward results

### Other Optimization Methods During Training

- **Exploration Strategy**: During exploration, setting high probability for up, down, left, and right movements
  - Current implementation finds exploration tends to be confined, with rewards slowly increasing but not significantly even after 1000+ episodes
  - During training, agents often reach high water levels but can't progress, so tested restricting similar consecutive actions by setting 0.9 probability that state.next should not equal consecutive step
  - This prevents agents from staying in the same position without change, forcing random exploration
  - Found that backtracking was common in initial episodes, as consecutive action similarity couldn't be effectively measured

- **Reducing Exploration Rate**: Exploration rate decreases with each episode using epsilon decay
  - Originally all episodes used 0.99975 rate, but testing showed better results with faster decline:
  - Epsilon = 0.01 + (1 - episode) * exp(-1 * (episode + 1) / 100)
  - This formula produced much better effects

- **Environment Reward Calculation Improvement**:
  - Original reward adds points for new sections discovered during actions
  - Can add incentives for actions that increase score
  - Rewards multiplied by 40 to increase model stability
  - This approach can simulate achievement-like reward calculations and significantly improve training effectiveness

### Training Results

- First week: Successful progress through levels
- Second week: Reached challenging areas but encountered failures

# 6. Soft actor critic implement in gym-racing car
![image](https://github.com/user-attachments/assets/32972831-b670-44b7-93a0-1d88d7004643)

This environment is a simple continuous control task for a single agent. The state consists of
96x96x3 pixels representing the agent's surroundings. The agent receives a constant penalty
of -0.1 every timestep, and a reward of +1000/num_tiles for each unique tile it
visits. The motivation behind this reward structure is to provide a sufficiently dense reward
signal for the agent to learn the basic driving skills, while encouraging it to explore and visits many tiles as possible. 

### Main Training Method: SAC (Soft Actor-Critic)

```
Algorithm 1 Soft Actor-Critic
    Initialize parameter vectors ψ, ψ̄, θ, φ.
    for each iteration do
        for each environment step do
            at ~ πφ(at|st)
            st+1 ~ p(st+1|st, at)
            D ← D ∪ {(st, at, r(st, at), st+1)}
        end for
        for each gradient step do
            ψ ← ψ - λV∇ψJV(ψ)
            θi ← θi - λQ∇θiJQ(θi) for i ∈ {1, 2}
            φ ← φ - λπ∇φJπ(φ)
            ψ̄ ← τψ + (1 − τ)ψ̄
        end for
    end for
```

### Network Architecture
- The algorithm utilizes three essential networks:
  - Policy network (strategy network)
  - Critic network
  - Target Q-network
- This implementation uses Q-network and target network as counterparts
- **Target Network Update**: Minimizes the difference between next q value and critic network/q network to ensure q network correctly predicts state/action values
- **Policy Network Update**: Maximizes (q1/q2) - target/Q network estimate. Differs from original actor-critic by considering log pi for minimizing action probability distribution log when selecting specific actions, maintaining action diversity (entropy concept)

### Improvement Methods

#### Roll Out
- Found that this environment requires continuous state space with sufficient memory buffer for sequential data (state, action, reward, next state) to update effectively
- Regarding memory buffer batch size:
  - Higher sizes initially speed up training
  - Later stages show learning reward fluctuations and potential overfitting
  - Initial batch size of 70, reduced to 32 in later stages showed better training results

####  Prioritized Experience
- During data updates, batches with highest TD error are prioritized as network training data
- During training, memory with highest TD error is directly selected rather than random sampling
- After processing, these batches are deleted and other data selected in subsequent updates
- After a fixed size of memory usage, memory is reset to collect new data with large TD errors

####  Reducing log_pi Entropy Impact
- Observed that during later training stages, log_pi (action entropy maximization factor) affects model generalizability
- Weights for this consideration were reduced

###  Results
- Performance: 400-800 score per episode

# 7. NeurIPS 2019 challenge-learn to move
 Learn to Move: Walk Around (“<Student_ID>_hw4_<train|test>.py”)
This is a NeurIPS 2019 challenge. In this challenge, your task is to develop a
controller for a physiologically plausible 3D human model to move (walk or run)
following velocity commands with minimum effort.
![image](https://github.com/user-attachments/assets/71c1cf22-b032-4e3d-83cb-54be6e5e2a08)

### Training Method: Soft Actor-Critic

- The algorithm implemented is Soft Actor-Critic (SAC), with implementation as shown below:

```
Algorithm 1 Soft Actor-Critic
    Initialize parameter vectors ψ, ψ̄, θ, φ.
    for each iteration do
        for each environment step do
            at ~ πφ(at|st)
            st+1 ~ p(st+1|st, at)
            D ← D ∪ {(st, at, r(st, at), st+1)}
        end for
        for each gradient step do
            ψ ← ψ - λV∇ψJV(ψ)           # V critic network
            θi ← θi - λQ∇θiJQ(θi) for i ∈ {1, 2}  # 2 Q critic networks
            φ ← φ - λπ∇φJπ(φ)
            ψ̄ ← τψ + (1 − τ)ψ̄           # Target V critic network
        end for
    end for
```

### Network Architecture and Functions

- **Policy Network**: Responsible for determining which action to take given a state (observation). It outputs the predicted value using two Q-value networks. It estimates the value function for each state.
- **Action Entropy**: Maintains action randomness/entropy. The algorithm calculates the probability of an action being selected in a given state. During training, the action's policy network generates a mean with a log_std network that produces log_std. Based on these two values, a normal distribution is sampled to select an action, and the action's probability under this distribution is calculated as log for entropy.
- **Parameter Update Process**: During parameter updates, 256 data records are extracted from replay memory. Each record contains (State, action, reward, next_state). The three networks are updated as follows:

#### Q-Value Network
- Minimizes the Q network(state,action) prediction error against target value
- Target value calculation: reward + target_network(next_state)
- This minimizes Q network error to ensure accurate state/action value predictions
## Improved Training Methods

### Skip Frame
- During training, when sampling from replay memory, state is input to policy network to generate action
- If environment responds slowly, collect reward/next state and calculate various possible responses
  - Most states have consistent possible next states, requiring occasional random exploration
  - Some states with unstable next state transitions benefit from more exploration and testing

### Prioritized Memory Replay
- When sampling from replay memory, calculate Q value and target value difference (td error)
- After update, records with highest td error are kept in memory
- During subsequent updates, samples with highest error are prioritized
- After several updates, memory is reset to collect new high td error data

### TD Error Priority
- Records with high TD error are prioritized for training, improving model speed
- Knowledge expansion for unfamiliar states is prioritized for better rewards

## Results
- Maximum score: 21-22
- Average score: 13-15
