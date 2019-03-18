import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time


WHITE = 1
BLACK = -1


def play_game(player_white, player_black, game):
    players = {WHITE: player_white, BLACK: player_black}
    cur_player = 1
    board = game.getInitBoard()

    while not game.getGameEnded(board, cur_player):
        action = players[cur_player](game.getCanonicalForm(board, cur_player))
        valids = game.getValidMoves(game.getCanonicalForm(board, cur_player), WHITE)
        if valids[action] == 0:
            print(action)
            assert valids[action] > 0
        board, cur_player = game.getNextState(board, cur_player, action)

    return game.getGameEnded(board, WHITE)  # +1: WHITE wins, -1: BLACK wins, else: DRAW


def contest(player_white, player_black, game, num_games):
    """Returns (#wins, #draws, #losses) from perspective of player_white."""
    n_wins_white, n_draws, n_losses_white = 0, 0, 0
    game_results = []
    for i in range(num_games):
        print('playing game {}/{}'.format(i+1, num_games))
        game_result = play_game(player_white=player_white, player_black=player_black, game=game)
        game_results.append(game_result)
        if game_result == WHITE:
            n_wins_white += 1
        elif game_result == BLACK:
            n_losses_white += 1
        else:
            n_draws += 1

    return (n_wins_white, n_draws, n_losses_white), np.array(game_results)
