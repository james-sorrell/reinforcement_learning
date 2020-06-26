import gym
import argparse
from mc_policy_evaluation import Agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Q Learning")
    # OPTIONAL ARGS
    parser.add_argument('-gamma', type=float, default=0.99, help="Discount factor for monte-carlo")
    parser.add_argument('-n_episodes', type=int, default=500000, help="Number of episodes to run")
    args = parser.parse_args()
    
    env = gym.make('Blackjack-v0')
    agent = Agent()

    for i in range(args.n_episodes):
        if i % 50000 == 0:
            print("Episode {}".format(i))
        observation = env.reset()
        done = False
        while not done:
            action = agent.policy(observation)
            observation_, reward, done, info = env.step(action)
            agent.memory.append((observation, reward))
            observation = observation_
        agent.update_V()
    
    print(agent.V[(21,3,True)])
    print(agent.V[(4, 1, False)])
