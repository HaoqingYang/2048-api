import sys
sys.path.append("..")

from game2048.agents import Agent
from MyCode.onehot import change_to_onehot
import numpy as np
from keras.models import load_model
import os

current = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current,"trained_model.h5")
my_model = load_model(model_path)

class MyAgent(Agent):
    def __init__(sefl,game,display=None):
        if game.size != 4:
            raise ValueError(
                    "'%s' can only work with game of 'size' 4." % self.__class__.__name__)
        super().__init__(game,display)

    def step(self):
        current_score = self.game.score
        board = np.array([change_to_onehot(self.game.board)])
        direction = int(my_model.predict(board).argmax())
        return direction

