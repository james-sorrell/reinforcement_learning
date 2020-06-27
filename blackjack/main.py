import gym
import argparse
import matplotlib.pyplot as plt
from mc_policy_iteration import Agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Q Learning")
    # OPTIONAL ARGS
    parser.add_argument('-gamma', type=float, default=0.99, help="Discount factor for monte-carlo")
    parser.add_argument('-eps', type=float, default=0.001, help="Epsilon factor for monte-carlo")
    parser.add_argument('-n_episodes', type=int, default=500000, help="Number of episodes to run")
    args = parser.parse_args()
    
    win_lose_draw = {-1:0, 0:0, 1:0}
    win_rates = []

    env = gym.make('Blackjack-v0')
    agent = Agent(args.eps, args.gamma)

    for i in range(args.n_episodes):

        if i > 0 and i % 1000 == 0:
            pct = win_lose_draw[1] / i
            win_rates.append(pct)
        
        if i % 50000 == 0:
            rates = win_rates[-1] if win_rates else 0.0
            print("Episode {}\t Win Rate: {:.2f}".format(i, rates))

        observation = env.reset()
        done = False

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.memory.append((observation, action, reward))
            observation = observation_

        agent.update_Q()
        win_lose_draw[reward] += 1
    
    plt.plot(win_rates)
    plt.show()
