from dotenv import load_dotenv
import serial
import os
import gym

# load environment variables into program
load_dotenv()
# get the port the arduino is running on
port = os.getenv("PORT")
baudrate = os.getenv("BAUDRATE")
# setup serial connection with arduino
# arduino = serial.Serial(port=port, baudrate=baudrate, timeout=.1)

# setup simulation environment
env = gym.make('CartPole-v0')
env.reset()


def read():
    ##arduino.write(bytes(x, 'utf-8'))
    data = arduino.readline().strip()
    return data


while True:
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()
    #value = read()
    # print(value)
