import random
import numpy as np
import pickle
import math
BOARD_ROWS = 3
BOARD_COLS = 3
class TicTacToe():
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS*BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1

    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.playerSymbol)+str(self.board)
        return self.boardHash

    def winner(self):
        # row
        board=self.board.reshape((3,3))
        for i in range(BOARD_ROWS):
            if sum(board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(board[i, :]) == -3:
                self.isEnd = True
                return -1
        # col
        for i in range(BOARD_COLS):
            if sum(board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(board[:, i]) == -3:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0.5
        # not end
        self.isEnd = False
        return None

    def availablePositions(self):
        board=self.board.reshape(3*3)
        positions = []
        for i in range(9):
                if board[i] == 0:
                    positions.append(i)  # need to be tuple
        return positions
    def updateState(self, position):
        p=position[0]*3+position[1]
        self.board[p] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # only when game ends
    def giveReward(self,win):
        result = win
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        elif result==0.5:
            self.p1.feedReward(0.5)
            self.p2.feedReward(0.5)
    def giveReward2(self,win):
        result = win
        if result == 1:
            self.p1.feedReward(1)
        elif result == -1:
            self.p1.feedReward(0)
        elif result==0.5:
            self.p1.feedReward(0.1)

    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_ROWS* BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1
    def getboard(self):
        board=np.zeros(10)
        board[0]=self.playerSymbol
        board[1:]=self.board
        return board
    def train(self,rounds):
        p1_wins=0
        p1_ties=0
        board_hash=self.getHash()
        for i in range(rounds):
            if(i%10000==0):
                print("round",i,"--------------------")
            while not self.isEnd:
             #   print("------------------")
                # Player 1
                #positions = self.availablePositions()
                board=self.getboard()
                p1_action = self.p1.choose_action(board)
                # take action and upate board state
                self.updateState(p1_action)
             #   print("board p1 :\n",self.board)
                board_hash = self.getHash()
                #self.p1.update_state_value(board_hash)
                self.p1.addState(board_hash,p1_action)
                win = self.winner()
                if win is not None:
              #      print("roond",i," p1 win!")
               #     print("p1 recorded state:",self.p1.recorded_states)
                #    print("p1 state_values:",self.p1.states_value)
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward(win)
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    board=self.getboard()
                    p2_action = self.p2.choose_action(board)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash,p2_action)
                    win = self.winner()
                    if win is not None:
                  #      print("roond",i," p2 win!")
                   #     print("p2 recorded state:",self.p2.recorded_states)
                    #    print("p2 state_values:",self.p2.states_value)
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward(win)
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
            if win==1:
                p1_wins+=1
            elif win==0.5:
                p1_ties+=1
        return p1_wins,p1_ties
    def play_with_RandomAgent(self,rounds):
        p1_wins=0
        p1_ties=0
        p1_loss=0
        board_hash=self.getHash()
        for i in range(rounds):
            if(i%10000==0):
                print("round",i,"--------------------")
            while not self.isEnd:
             #   print("------------------")
                # Player 1
                board=self.getboard()
                p1_action = self.p1.choose_action(board)
                # take action and upate board state
                self.updateState(p1_action)
             #   print("board p1 :\n",self.board)
                board_hash = self.getHash()
                self.p1.addState(board_hash,p1_action)
                win = self.winner()
                if win is not None:
              #      print("roond",i," p1 win!")
               #     print("p1 recorded state:",self.p1.recorded_states)
                #    print("p1 state_values:",self.p1.states_value)
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward2(win)
                    self.p1.reset()
           #         self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    #print("positions:",positions)
                    p2_action = self.p2.choose_action(positions)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    win = self.winner()
                    if win is not None:
                        p1_loss+=1
                        self.p1.reset()
                        self.reset()
                        break
            if win==1:
                p1_wins+=1
            elif win==0.5:
                p1_ties+=1
            elif win==-1:
                p1_loss+=1
        return p1_wins,p1_ties,p1_loss
            
class Agent():
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.recorded_states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value
        self.init=0.6
    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash
    def get_q(self, state_hash):
        if self.states_value.get(state_hash) is None:
            self.states_value[state_hash]=[0]*9
        return self.states_value[state_hash]

    def choose_action(self, board):
        positions=[]
        #print(np.shape(board))
        for i in range(1,10):
            if board[i] == 0:
                positions.append(i-1)  # need to be tuple
        boardhash=str(board)
        q_vals = self.get_q(boardhash)
        if(len(positions)==0):
            print(board)
        max_q = max([q_vals[p] for p in positions])
        move = random.choice([i for i, j in enumerate(q_vals) if ((j == max_q) and (i in positions))])
        action=[0,0]
        action[0]=(int(move/3))
        action[1]=(int(move%3))
        # print("{} takes action {}".format(self.name, action))
        return action

    # append a hash state
    def addState(self, statehash,action):
        action_v=action[0]*3+action[1]
        self.recorded_states.append((statehash,action_v))
    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        first=True
        maximum=0
                
        for st in reversed(self.recorded_states):
            if self.states_value.get(st[0]) is None:
                self.states_value[st[0]]=[0]*9
            if first:
                self.states_value[st[0]][st[1]]=reward
                first=False
            else:
                self.states_value[st[0]][st[1]] = (1 - self.lr)*self.states_value[st[0]][st[1]] + self.lr * self.decay_gamma * maximum
            maximum=max(self.states_value[st[0]])
    def reset(self):
        self.recorded_states = []

    def savePolicy(self,name):
        file='./hw1_3_data/q_learning_policy_'+str(name)
        fw = open(file, 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self):
        fr_q_p1=pickle.load(open("./qlearning_policy/q_learning_policy_p1", 'rb'))
        fr_q_p2=pickle.load(open("./qlearning_policy/q_learning_policy_p2", 'rb'))

        fr_q_p1.update(fr_q_p2)
        self.states_value = fr
        
    def returnPolicy(self):
       # print("policy:\n",self.states_value)
        return self.states_value
class RandomAgent:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
    def choose_action(self, positions):
        #print(positions_v)
        #print("positions_v:",positions_v)
        #print("random chose:",[ j for j in positions_v ])
        move = random.choice([j for j in positions])
        action=[0,0]
        action[0]=(int(move/3))
        action[1]=(int(move%3))
        # print("{} takes action {}".format(self.name, action))
        return action


p1 = Agent("p1")
p2 = Agent("p2")

st = TicTacToe(p1, p2)
print("training...")
epoch_nums=20000
p1_wins,p1_ties=st.train(epoch_nums)
p1.savePolicy("p1")
p2.savePolicy("p2")
print("p1 win:",p1_wins/epoch_nums*100,"%")
print("p1 tie:",p1_ties/epoch_nums*100,"%")



print("-----------change turn-----------")
p1 = Agent("p1")
p2 = Agent("p2")

st = TicTacToe(p2, p1)
print("training...")
epoch_nums=20000
p1_wins,p1_ties=st.train(epoch_nums)
p1.savePolicy("p1")
p2.savePolicy("p2")
print("p1 win:",p1_wins/epoch_nums*100,"%")
print("p1 tie:",p1_ties/epoch_nums*100,"%")
print("play with random agent------------")
# # # play with RandomAgent
# epoch_nums=20000
# p1 = Agent("computer", exp_rate=0)
# p3=RandomAgent("Random")
# st = TicTacToe(p1, p3)   
# p1_wins,p1_ties,p1_loss=st.play_with_RandomAgent(epoch_nums)
# p1.savePolicy()

epoch_nums=1000
p2 = Agent("computer", exp_rate=0)
p2.loadPolicy()
p3=RandomAgent("Random")
st = TicTacToe(p2, p3)   
p2_wins,p2_ties,p2_loss=st.play_with_RandomAgent(epoch_nums)
print("p2 win:",(p2_wins/epoch_nums)*100,"%")
print("p2 tie:",p2_ties/epoch_nums*100,"%")
print("p2 action:",p2.choose_action(np.zeros(10)))



