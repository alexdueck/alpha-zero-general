import yaml
import glob
import numpy as np
import os
import re
import argparse

from utils import dotdict
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as PyTorchNet
from othello.chainer.NNet import NNetWrapper as ChainerNet
from othello.keras.NNet import NNetWrapper as KerasNet
from MCTS import MCTS
import eval


def get_player(mcts, temp=0):
    return lambda x: np.argmax(mcts.getActionProb(x, temp=temp))


def setup_player(net, game, args, ckpt_filepath):
    n = net(game)
    ckpt_dir, ckpt_file = os.path.split(ckpt_filepath)
    n.load_checkpoint(ckpt_dir, ckpt_file)
    mcts = MCTS(game, n, args)
    player = get_player(mcts, temp=0)
    return player


def extract_ckpt_id(ckpt):
    number = re.findall(r'\d+', ckpt)
    if len(number) != 1:
        raise Exception('found the numbers {} in checkpoint filename {}; 1 number expected'.format(number, ckpt))
    return int(number[0])


def reverse_perspective(res):
    return res[1], res[0], res[2]


def score_from_res(res):
    return (res[0] - res[1]) / sum(res)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', '-c', type=str,
                        help='config file')
    args = parser.parse_args()
    print('running with args: {}'.format(args))

    g = OthelloGame(6)
    nets = {
        'keras': KerasNet,
        'pytorch': PyTorchNet,
        'chainer': ChainerNet
    }
    agent_args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    opponent_args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    with open(args.config, 'r') as f:
            cfg = yaml.load(f)
    print(cfg)
    print('looking for files {}'.format(cfg['chkpt_dir'] + '*.pth.tar'))
    agent_ckpts = sorted(glob.glob(cfg['chkpt_dir'] + '*.pth.tar'))
    print(agent_ckpts)

    for agent_side in cfg['agent_side']:
        for opponent, opponent_ckpt in cfg['opponents'].items():
            print('***************************************************************')
            print('setting up contest with opponent: {}, agent_side: {}'.format(opponent, agent_side))
            opp_net = nets[opponent]
            results_filename = cfg['results_file_basename'] + '_' + agent_side + '_' + opponent + '.csv'
            print(results_filename)
            with open(results_filename, 'w') as f:
                f.write('# agent_side: {}, opponent: {}\n'.format(agent_side, opponent))
                f.write('# results are from the agent\'s perspective\n')
                f.write('# score = = (#wins - #losses) / (#wins + #losses + #draws)\n')
                f.write('# checkpoint; #wins, #losses, #draws, #score\n')
                for agent_ckpt in agent_ckpts:
                    print('benchmarking checkpoint {}'.format(agent_ckpt))
                    if agent_side == 'white':
                        agent_player = setup_player(net=nets['pytorch'], game=g,
                                                    args=agent_args, ckpt_filepath=agent_ckpt)
                        opp_player = setup_player(net=nets[opponent], game=g,
                                                  args=opponent_args, ckpt_filepath=opponent_ckpt)
                        res, _ = eval.contest(player_white=agent_player, player_black=opp_player,
                                              game=g, num_games=cfg['num_games'])
                        it = extract_ckpt_id(agent_ckpt)
                        f.write('{};{};{};{};{}\n'.format(it, res[0], res[1], res[2], score_from_res(res)))
                    elif agent_side == 'black':
                        agent_player = setup_player(net=nets['pytorch'], game=g,
                                                    args=agent_args, ckpt_filepath=agent_ckpt)
                        opp_player = setup_player(net=nets[opponent], game=g,
                                                  args=opponent_args, ckpt_filepath=opponent_ckpt)
                        res, _ = eval.contest(player_white=opp_player, player_black=agent_player,
                                              game=g, num_games=cfg['num_games'])
                        # retrieve agent's perspective (res is given from perspective of player_white)
                        res = reverse_perspective(res)
                        it = extract_ckpt_id(agent_ckpt)
                        f.write('{};{};{};{};{}\n'.format(it, res[0], res[1], res[2], score_from_res(res)))
                    else:
                        raise Exception('invalid agent_side value: {}'.format(agent_side))


if __name__ == '__main__':
    main()
