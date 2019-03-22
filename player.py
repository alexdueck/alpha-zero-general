import numpy as np
from MCTS import MCTS
from othello.pytorch.NNet import NNetWrapper as NNet


class Player:
    def __init__(self, game, mcts_args, ckpt_dir, ckpt_file):
        self.game = game
        self.mcts_args = mcts_args
        self.n = NNet(game)
        self.n.load_checkpoint(ckpt_dir, ckpt_file)
        self.mcts = MCTS(self.game, self.n, self.mcts_args)

    def __call__(self, board):
        return np.argmax(self.mcts.getActionProb(board, temp=0))

    def reset_mcts(self):
        self.mcts = MCTS(self.game, self.n, self.mcts_args)
