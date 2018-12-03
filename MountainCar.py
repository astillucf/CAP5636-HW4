import gym
import numpy as np

env = gym.make('MountainCar-v0')
env.reset()

qTable = {}

alpha = 1.0
beta = 0.5
epsilon = 0.2

episodes = 100
steps = 10000


def getAction(observation, index):
    position = int(round(observation[0], 1))
    velocity = int(round(observation[1], 1))
    action = 0

    if np.random.random_sample() < epsilon or ((position, velocity, action) not in qTable):
        return env.action_space.sample()
    elif ((position, velocity, action) not in qTable):
        for a in range(3):
            qTable[(position, velocity, a)] = 0
        return env.action_space.sample()
    else:
        maxQ = qTable[(position, velocity, 0)]
        index = 0
        for a in range(3):
            if maxQ < qTable[(position, velocity, a)]:
                maxQ = qTable[(position, velocity, a)]
                index = a
        return a


def updateQTable(previousObservation, observation, action, reward):
    position = int(round(observation[0], 1))
    velocity = int(round(observation[1], 1))
    previousPosition = int(round(previousObservation[0], 1))
    previousVelocity = int(round(previousObservation[1], 1))
    maxQ = 0

    if (position, velocity, action) in qTable:
        maxQ = qTable[(position, velocity, 0)]
        for a in range(3):
            maxQ = max(maxQ, qTable[(position, velocity, a)])
    else:
        for a in range(3):
            qTable[(position, velocity, a)] = 0
    if (previousPosition, previousVelocity, action) not in qTable:
        for a in range(3):
            qTable[(previousPosition, previousVelocity, a)] = 0
    qTable[(previousPosition, previousVelocity, action)] += alpha * (reward + beta * maxQ - qTable[(previousPosition, previousVelocity, action)])


if __name__ == '__main__':
    for _ in range(episodes):
        print("Episode: ", _)

        observation = env.reset()
        previousObservation = observation
        beta *= 0.99

        for step in range(steps):
            action = getAction(previousObservation, step)
            observation, reward, done, info = env.step(action)
            reward = abs(observation[0]) - 0.5 + abs(observation[1])
            updateQTable(previousObservation, observation, action, reward)
            alpha *= 0.99

            if observation[0] >= 0.5:
                print("Number of Steps: ", step)
                break
    env.close()