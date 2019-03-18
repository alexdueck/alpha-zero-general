import numpy as np
from utils import *

import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame, display
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet
import eval


def get_player(mcts, temp=0):
    return lambda x: np.argmax(mcts.getActionProb(x, temp=temp))


def setup_player(game, args, chkp_dir, chkp_file):
    n = NNet(g)
    n.load_checkpoint(chkp_dir, chkp_file)
    mcts = MCTS(game, n, args)
    player = get_player(mcts, temp=0)
    return player


g = OthelloGame(6)

args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
p1 = setup_player(game=g,
                  args=args1,
                  chkp_dir='./pretrained_models/othello/pytorch/',
                  chkp_file='6x100x25_best.pth.tar')

args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
p2 = setup_player(game=g,
                  args=args2,
                  chkp_dir='./pretrained_models/othello/pytorch/',
                  chkp_file='6x100x25_best.pth.tar')

# arena = Arena.Arena(n1p, n2p, g, display=display)
# contest_result = arena.playGames(6, verbose=False)

# my own evaluation function
print('starting contest, p1: white, p2: black')
contest_result1, game_results1 = eval.contest(player_white=p1, player_black=p2, game=g, num_games=5)

print('game results from WHITE\'s perspective (white: p1, black: p2):')
print(game_results1)
print('contest result (white: p1, black: p2): {}'.format(contest_result1))

# my own evaluation function
print()
print('starting contest, p1: black, p2: white')
contest_result2, game_results2 = eval.contest(player_white=p2, player_black=p1, game=g, num_games=5)

print('game results from WHITE\'s perspective (white: p2, black: p1):')
print(game_results2)
print('contest result (white: p2, black: p1): {}'.format(contest_result2))


game_results = np.concatenate((game_results1, -game_results2))
# merge contest results coherently, s.t. #wins & #losses are added from perspective of the same player
contest_result = (contest_result1[0] + contest_result2[2],
                  contest_result1[1] + contest_result2[1],
                  contest_result1[2] + contest_result2[0])

print('all game results from perspective of same player:')
print(game_results)
print('overall contest result: {}'.format(contest_result))
