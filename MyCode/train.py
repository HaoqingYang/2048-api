import sys
sys.path.append("..")
import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from game2048.agents import Agent
from game2048.game import Game
from game2048.expectimax import board_to_move
from keras.models import load_model
from mymodel import model_build
from onehot import change_to_onehot,CAND,MAP_TABLE
from collections import namedtuple

Guide = namedtuple("Guide",("state","action"))
BATCH_SIZE = 1024
epoches = 10

class TrainAgent(Agent):
    def __init__(self, game, capacity):
        super().__init__(game)
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def train(self,begin_score,end_score):
        counter = 0

        '''if os.path.exists("prev_model_scores.json"):
            with open("prev_model_scores.json",'r') as f:
                prev_model_scores = json.load(f)
                prev_model_score = prev_model_scores[-1]
        else:
            prev_model_score = 0
            prev_model_scores = []'''
        
        if os.path.exists("trained_model.h5"):
            print("Load the existed model.")
            self.model = load_model("trained_model.h5")
        else:
            self.model = model_build(self.game.size,self.game.size,CAND)

        while True:
            self.reset_game(begin_score,end_score)
            while not self.game.end:
                self.play()
            print("score:", self.game.score, end='\t')

            if len(self.memory) < BATCH_SIZE:
                continue
            guides = random.sample(self.memory, BATCH_SIZE)
            X = []
            Y = []
            for guide in guides:
                X.append(guide.state)
                ohe_action = [0]*4
                ohe_action[guide.action] = 1
                Y.append(ohe_action)
            loss, acc =self.model.train_on_batch(np.array(X),np.array(Y))
            print("第%d轮 \t loss:%.3f \t acc:%.3f" % (counter,float(loss),float(acc)))
           
            counter += 1
            if counter % epoches == 0:
                evaluate_score = self.evaluate(begin_score,end_score,N_TESTS=10,LIMIT=True,verbose=False)

                #if evaluate_score > prev_model_score:
                self.model.save("trained_model.h5")
                    #prev_model_score = evaluate_score
                    #prev_model_scores.append(prev_model_score)
                    #with open("prev_model_scores.json",'w') as f:
                        #json.dump(prev_model_scores,f)
                if evaluate_score > end_score:
                    break
        print("训练结束。")


    def play(self):
        onehot_board = change_to_onehot(self.game.board)
        self.push(onehot_board, board_to_move(self.game.board))
        self.game.move(self.predict())

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = Guide(*args)
        self.pos = (self.pos+1)%self.capacity

    def reset_game(self,begin_score,end_score):
        if not begin_score in MAP_TABLE:
            raise AssertionError("init_max_score must be a number in %s" % list(MAP_TABLE.key()))
        new_board = np.zeros((self.game.size,self.game.size))
        if begin_score > 2:
            other_scores = [i for i in MAP_TABLE if i <= begin_score]
            other_scores = np.random.choice(other_scores, int(15*random.random()),replace=True)
            locations = np.random.choice(16,1+len(other_scores),replace=False)
            new_board[locations//4, locations%4] = np.append(other_scores,begin_score)

        self.game.board = new_board
        self.game._maybe_new_entry()
        self.game._maybe_new_entry()
        self.game.__end = 0 
        self.game.score_to_win = end_score

    def predict(self):
        onehot_board = change_to_onehot(self.game.board)
        board = np.array([onehot_board])
        direction = int(self.model.predict(board).argmax())
        return direction

    def evaluate(self, begin_score, end_score, N_TESTS, LIMIT=True,verbose=False):
        scores = []
        for i in range(N_TESTS):
            if not LIMIT:
                self.reset_game(begin_score,np.inf)
            else:
                self.reset_game(begin_score,end_score)
            while not self.game.end:
                direction = self.predict()
                self.game.move(direction)
            scores.append(self.game.score)
        if verbose:
            print(scores)
        score = sum(scores) / len(scores)
        return score

mygame = Game(enable_rewrite_board=True)
train_agent = TrainAgent(mygame, 65536)
train_agent.train(0,2048)


