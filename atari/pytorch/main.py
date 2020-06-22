import os
import argparse
import gym
import agent as Agents
import numpy as np
from utils import plot_learning_curve, make_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Q Learning")
    # OPTIONAL ARGS
    parser.add_argument('-n_games', type=int, default=300, help="Number of games to play")
    parser.add_argument('-lr', type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument('-eps_min', type=float, default=0.1, help="Minimum value for epsilon-greedy action selection")
    parser.add_argument('-gamma', type=float, default=0.99, help="Discount factor for update equation")
    parser.add_argument('-eps_dec', type=float, default=1e-5, help="Epsilon decrement amount")
    parser.add_argument('-eps', type=float, default=1.0, help="Starting value for epsilon in epsilon-greedy")
    parser.add_argument('-max_mem', type=int, default=20000, help="Maximum size for memory replay buffer")
    parser.add_argument('-repeat', type=int, default=4, help="Number of frames to repeat & stack")
    parser.add_argument('-bs', type=int, default=32, help="Batch size for memory replay sampling")
    parser.add_argument('-replace', type=int, default=1000, help="Interval for replacing target network")
    parser.add_argument('-env', type=str, default='PongNoFrameskip-v4', help="Atari Environment to use, e.g.:\n\
                                                            PongNoFrameskip-v4\nBreakoutNoFrameskip-v4\n\
                                                            SpaceInvadersNoFrameskip-v4\n\EnduroNoFrameskip-v4\n\
                                                            AtlantisNoFrameskip-v4")
    parser.add_argument('-gpu', type=str, default='1', help="GPU: 0 or 1")
    parser.add_argument('-load_checkpoint', type=bool, default=False, help='load model checkpoint')
    parser.add_argument('-path', type=str, default='models/', help="Path for model saving/loading")
    parser.add_argument('-algo', type=str, default='DQNAgent', help="DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent")
    parser.add_argument('-clip_reward', type=bool, default=False, help="Clip rewards to range -1:1")
    parser.add_argument('-no_ops', type=int, default=0, help="Max number of no ops for testing")
    parser.add_argument('-fire_first', type=bool, default=False, help="Set first action of episode to fire")
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    env = make_env(env_name=args.env, repeat=args.repeat, clip_reward=args.clip_reward,
                    no_ops=args.no_ops, fire_first=args.fire_first)
    best_score = -np.inf

    # Find defined class in Agents
    agent_ = getattr(Agents, args.algo)
    print("Agent: {}".format(agent_))
    print("Environment: {}".format(args.env))
    agent = agent_(gamma=args.gamma,
                    epsilon=args.eps,
                    lr=args.lr,
                    num_actions=env.action_space.n,
                    input_dims=env.observation_space.shape,
                    mem_size=args.max_mem,
                    batch_size=args.bs,
                    eps_min=args.eps_min,
                    eps_dec=args.eps_dec,
                    replace=args.replace,
                    algo=args.algo,
                    env_name=args.env)

    if args.load_checkpoint:
        agent.load_models()

    fname = args.algo + '_' + args.env + '_alpha' + str(args.lr) + '_' + str(args.n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(args.n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if not args.load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
    
        avg_score = np.mean(scores[-100:])
        print("Episode: {}, Score: {}, Average Score: {:1f}, Best Score: {:1f}, Epsilon: {:1f}, Steps: {}".format(i, score, avg_score, best_score, agent.epsilon, n_steps))

        if avg_score > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score
        
        eps_history.append(agent.epsilon)
    
    plot_learning_curve(steps_array, scores, eps_history, figure_file)