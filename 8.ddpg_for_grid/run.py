import sys
import multiprocessing
import os.path as osp
import os
# import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
import argparse

# from common.vec_env.vec_frame_stack import VecFrameStack
from common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env
from common.tf_util import get_session
import logger
from importlib import import_module
from env.traffic_env import Traffic_env

# from common.vec_env.vec_normalize import VecNormalize
# from common import atari_wrappers, retro_wrappers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None


grid_x=10  #x horizontal, y vertical; direction 23 horizontal, direction 01 vertical
grid_y=5


def train(args, extra_args):
    # env_type, env_id = get_env_type(args.env)
    # print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    env_type='mlp'
    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    # env = build_env(args)
    env = Traffic_env(grid_x,grid_y)
    
    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    # print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        save_path=save_path,
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env



def test(args, extra_args,save_path):
    # env_type, env_id = get_env_type(args.env)
    # print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    env_type='mlp'
    testing = get_test_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    # env = build_env(args)
    env = Traffic_env(grid_x,grid_y)
    
    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    # print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = testing(
        save_path=save_path,
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env

def retrain(args, extra_args,save_path):
    # env_type, env_id = get_env_type(args.env)
    # print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    env_type='mlp'
    retraining = get_retrain_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    # env = build_env(args)
    env = Traffic_env(grid_x,grid_y)
    
    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    # print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = retraining(
        save_path=save_path,
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env

def get_default_network(env_type):
    if env_type == 'atari':
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        # alg_module = import_module('.'.join(['baselines', alg, submodule]))
        alg_module = import_module('.'.join([ alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn

def get_test_function(alg):
    return get_alg_module(alg).testing

def get_retrain_function(alg):
    return get_alg_module(alg).retraining


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}



def main(args, extra_args, save_path):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    # arg_parser = common_arg_parser()
    # args, unknown_args = arg_parser.parse_known_args()
    # extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, extra_args)
    # env.close()

    # if args.save_path is not None and rank == 0:
    #     save_path = osp.expanduser(args.save_path)
    model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        print()
        # env = build_env(args)
        obs = env.reset()
        def initialize_placeholders(nlstm=128,**kwargs):
            return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))
        state, dones = initialize_placeholders(**extra_args)
        while True:
            actions, _, state, _ = model.step(obs,S=state, M=dones)
            obs, _, done, _ = env.step(actions)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()

        env.close()

def main_test(args, extra_args,save_path):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = test(args, extra_args,save_path)
    

    env.close()

def main_retrain(args, extra_args,save_path):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = retrain(args, extra_args,save_path)
    model.save(save_path)

    env.close()

if __name__ == '__main__':
    save_path='./models/ddpg'
    # args = parser.parse_args()
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = parse_cmdline_kwargs(unknown_args)
    if args.train:
        main(args, extra_args, save_path)
    if args.test:
        main_test(args, extra_args,save_path)
    if args.retrain:  # train based on the train checkpoint stored in save_path
        main_retrain(args, extra_args,save_path)
