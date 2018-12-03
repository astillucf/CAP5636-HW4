import gym
import numpy as np

env = gym.make('CartPole-v1')
env.reset()

def runCartPole(parameters):
    observation = env.reset()
    currentReward = 0

    for _ in range(500):
        env.render()
        if np.matmul(parameters, observation) < 0:
            action = 0
        else:
            action = 1

        observation, reward, done, info = env.step(action)

        currentReward += reward
        if done:
            break
    return currentReward

def CartPoleSolver():
    randomScale = .5
    maxReward = 0
    parameters = 1

    for _ in range(1000):
        newParameters = parameters + np.random.rand(4) * randomScale
        reward = runCartPole(newParameters)

        if reward > maxReward:
            maxReward = reward
            parameters = newParameters

        print('Reward:', reward, 'Max reward:', maxReward)

        if reward == 500:
            break

    finalReward = 0

    finalReward += runCartPole(parameters)

    print('Final Score:', finalReward)

if __name__ == '__main__':
    CartPoleSolver()