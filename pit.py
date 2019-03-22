import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame, display
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *
from player import Player

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""


# def get_player(mcts, temp=0):
#     return lambda x: np.argmax(mcts.getActionProb(x, temp=temp))


def setup_player(game, args, ckpt_dir, ckpt_file):
    # n = NNet(g)
    # n.load_checkpoint(chkp_dir, chkp_file)
    # mcts = MCTS(game, n, args)
    # player = get_player(mcts, temp=0)
    player = Player(game, args, ckpt_dir, ckpt_file)
    return player


g = OthelloGame(6)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play

# nnet players
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
p1 = setup_player(game=g,
                  args=args1,
                  ckpt_dir='./pretrained_models/othello/pytorch/',
                  ckpt_file='6x100x25_best.pth.tar')


args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
p2 = setup_player(game=g,
                  args=args2,
                  ckpt_dir='./pretrained_models/othello/pytorch/',
                  ckpt_file='6x100x25_best.pth.tar')

arena = Arena.Arena(p1, p2, g, display=display)
contest_result, game_results = arena.playGames(10, verbose=False)
print('contest result: {}'.format(contest_result))
print('game results:')
print(game_results)
