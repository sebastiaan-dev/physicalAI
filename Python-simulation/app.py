from dotenv import load_dotenv
import serial
import os
import gym
from components.learner import learn
from components.nn_models.nn_1 import NeuralNetwork

from components.simulation import CartPoleModifiedEnv
from models.NeuralInput import NeuralInput

# load environment variables into program
load_dotenv()
# get the port the arduino is running on
port = os.getenv("PORT")
baudrate = os.getenv("BAUDRATE")
# setup serial connection with arduino
# arduino = serial.Serial(port=port, baudrate=baudrate, timeout=.1)


def read():
    ##arduino.write(bytes(x, 'utf-8'))
    data = arduino.readline().strip()
    return data


# setup simulation environment
env = CartPoleModifiedEnv()

learn()

for i_episode in range(0):
    observation = env.reset()
    for t in range(1000):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)
        input = NeuralInput(ob[0], ob[1], ob[2], ob[3])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
#value = read()
# print(value)
