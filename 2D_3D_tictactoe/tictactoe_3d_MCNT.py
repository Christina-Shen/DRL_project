# 3D MCNT

import numpy as np
import pickle

BOARD_ROWS = 4
BOARD_COLS = 4
BOARD_LAYERS=4


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS,BOARD_LAYERS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1

    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS*BOARD_LAYERS))
        return self.boardHash

    def winner(self):
        #every layer PLANE,every row PLANE,every column PLANE   
        for i in range(BOARD_LAYERS):
            layers=self.board[:,:,i]
            rows_plane=self.board[i,:,:]
            col_plane=self.board[:,i,:]
            # row
            if sum(layers[i, :]) == 4 or sum(rows_plane[i, :])==4  or sum(col_plane[i, :])==4 :
                self.isEnd = True
                return 1
            if sum(layers[i, :]) == -4 or sum(rows_plane[i, :])==-4  or sum(col_plane[i, :])==-4:
                self.isEnd = True
                return -1
           # col
            for i in range(BOARD_COLS):
                if sum(layers[:, i]) == 4 or sum(rows_plane[:, i])==4  or sum(col_plane[:, i])==4:
                    self.isEnd = True
                    return 1
                if sum(layers[:, i]) == -4 or sum(rows_plane[:, i])==-4  or sum(col_plane[:, i])==-4:
                    self.isEnd = True
                    return -1
          # diagonal 
            plane_list=[layers,rows_plane,col_plane]
            for plane in plane_list:
                
                diag_sum1 = sum([plane[i,i] for i in range(BOARD_COLS)])
                diag_sum2 = sum([plane[i,BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
                diag_sum = max(abs(diag_sum1), abs(diag_sum2))
                if diag_sum == 4:
                    self.isEnd = True
                    if diag_sum1 == 4 or diag_sum2 == 4:
                        return 1
                    else:
                        return -1
        #diagonal line:
        diag_lines = [[(0,0,0),(1,1,1),(2,2,2),(3,3,3)], [(3,0,0),(2,1,1),(1,2,2),(0,3,3)], [(0,3,0), (1,2,1), (2,1,2), (3,0,3)],
                     [(0,0,3),(1,1,2),(2,2,1),(3,3,0)]]
        for line in diag_lines:
            diag_sum=sum(self.board[x,y,z] for x, y, z in line)
            if diag_sum==4:
               self.isEnd=True
               return 1
            if diag_sum==-4:
                self.isEnd=True
                return -1
        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def availablePositions(self):
        #print(self.board)
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for k in range(BOARD_LAYERS):
                    if self.board[i,j,k] == 0:
                        positions.append((i,j,k))  # need to be tuple
        return positions
    def updateState(self, position):
        position_t=(position[0],position[1],position[2])
        self.board[position_t] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1
    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.5)
            self.p2.feedReward(0.5)
    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS,BOARD_LAYERS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        p1_wins=0
        p1_ties=0
        for i in range(rounds):
            if i % 10 == 0:
                print("Rounds {}".format(i))
            t=0
            while not self.isEnd:
                t+=1
                # Player 1
                positions = self.availablePositions()
                board=np.zeros(65)
                board[1:]=self.board.reshape(4*4*4)
                board[0]=self.playerSymbol
                p1_action = self.p1.choose_action(board)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end
                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    board=np.zeros(65)
                    board[1:]=self.board.reshape(4*4*4)
                    board[0]=self.playerSymbol
                    p2_action = self.p2.choose_action(board)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
                if t>30000:
                    break
            if win==1:
                p1_wins+=1
            if win==0.5:
                p1_ties+=1
        return p1_wins,p1_ties

    # play with human
    def play2(self):
        while not self.isEnd:
            # Player 1
            positions = self.availablePositions()
            board=np.zeros(65)
            board[1:]=self.board.reshape(4*4*4)
            board[0]=self.playerSymbol
            p1_action = self.p1.choose_action(board)
            # take action and upate board state
            self.updateState(p1_action)
            #self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.availablePositions()
                p2_action = self.p2.choose_action(positions)

                self.updateState(p2_action)
                #self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break



class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS*BOARD_LAYERS))
        return boardHash

    def choose_action(self, current_board):
        board=current_board[1:]
        symbol=current_board[0]
        positions=[]
        for i in range(64):
            if board[i] == 0:
                positions.append(i)  # need to be tuple
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            move = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = board.copy()
                next_board[p] = symbol
                next_boardHash = str(next_board.reshape(64))
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    move = p

            act=[int(move%4),int((move//4)%4),int(move//16)]
 
            
        # print("{} takes action {}".format(self.name, action))
        return [int(move%4),int((move//4)%4),int(move//16)]

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        states_reversed = list(reversed(self.states))
        future_reward = reward
        N=len(states_reversed)
        for state in states_reversed:
            # The return from a state is the reward plus discounted reward from future states
            if self.states_value.get(state) is None:
                self.states_value[state] = 0
            # Update towards the mean return
            self.states_value[state] += 1/N * (future_reward - self.states_value[state])
            future_reward += self.decay_gamma * future_reward
            N-=1
    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('MCNT_policy', 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self):
        fr = open('MCNT_policy', 'rb')
        self.states_value = pickle.load(fr)
        fr.close()
    def returnPolicy(self):
        return self.states_value

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def choose_action(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            layer=int(input("Input your layer"))
            action = [row, col,layer]
            if action in positions:
                return action

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass



p1 = Player("p1")
p2 = Player("p2")
rounds=1000

st = State(p1, p2)
print("training...")
p1_wins,p1_ties=st.play(rounds)
p1.savePolicy()

print("p1 wins:",p1_wins/rounds*100,"%")

print("p1 ties:",p1_ties/rounds*100,"%")
# play with human
p1 = Player("computer", exp_rate=0)
p1.loadPolicy()
print("policy length:",len(list(p1.returnPolicy().items())))
print("p1 action:",p1.choose_action(np.zeros(65)))
#p2 = HumanPlayer("human")

#st = State(p1, p2)
#st.play2()