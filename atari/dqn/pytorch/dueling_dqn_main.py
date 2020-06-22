import numpy as np
from dueling_dqn_agent import DuelingDQNAgent
from utils import make_env, plot_learning_curve

if __name__ == '__main__':

    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 300
    agent = DuelingDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                    num_actions=env.action_space.n,
                    input_dims=(env.observation_space.shape),
                    mem_size=20000, batch_size=32,
                    eps_min=0.1, eps_start=1e-5,
                    replace=1000, algo='DuelingDQNAgent',
                    env_name='PongNoFrameskip-v4',
                    checkpoint_dir='models/')
                    
    if load_checkpoint:
        agent.load_models()
    
    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) \
            + '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
    
        avg_score = np.mean(scores[-100:])
        print("Episode: {}, Score: {}, Average Score: {:1f}, Best Score: {:1f}, Epsilon: {:1f}, Steps: {}".format(i, score, avg_score, best_score, agent.epsilon, n_steps))

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score
        
        eps_history.append(agent.epsilon)
    
    plot_learning_curve(steps_array, scores, eps_history, figure_file)