import gym
import numpy as np
import pickle

from collections import namedtuple


class PGCartPole:
    # hyperparameters

    HIDDEN_LAYER = 200  # number of hidden layer neurons
    INPUT_DIMENSION = 4  # input dimension for the model

    batch_size = 50  # every how many episodes to do a param update?
    learning_rate = 1e-3
    gamma = 0.99  # discount factor for reward

    render = False

    save_model = False
    save_interval = 100

    # resume from previous checkpoint?
    resume = False
    save_path = 'CartPole.pkl'

    transition_ = namedtuple('transition', ('state', 'hidden', 'probability', 'reward'))

    def __init__(self, game_name):
        self.game_name = game_name

        self.model = self.create_model()
        if self.resume:
            self.load(self.save_path)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def discount_rewards(r, gamma):
        """ take 1D float array of rewards and compute discounted reward """
        running_add = 0
        discounted_r = np.zeros_like(r)
        for idx in reversed(range(0, r.size)):
            running_add = running_add * gamma + r[idx]
            discounted_r[idx] = running_add
        return discounted_r

    def action(self, obs):
        action_probability, hidden = self.policy_forward(obs)
        action = 1 if action_probability >= 0.5 else 0
        return action

    def load(self, save_path):
        try:
            self.model = pickle.load(open(save_path, 'rb'))
            print("The following model is loaded:", save_path)
        except FileNotFoundError:
            print("The following model could not be found:", save_path)

    def save(self, save_path):
        with open(save_path, 'wb') as file:
            pickle.dump(self.model, file)

    def create_model(self):
        model = dict()
        model['W1'] = np.random.randn(self.HIDDEN_LAYER, self.INPUT_DIMENSION) / np.sqrt(self.INPUT_DIMENSION)
        model['W2'] = np.random.randn(self.HIDDEN_LAYER) / np.sqrt(self.HIDDEN_LAYER)
        return model

    def policy_forward(self, obs):
        """ Return probability of taking action 1 (right), and the hidden state """
        hidden = np.dot(self.model['W1'], obs)
        hidden[hidden < 0] = 0  # ReLU nonlinearity
        log_probability = np.dot(self.model['W2'], hidden)
        probability = self.sigmoid(log_probability)
        return probability, hidden

    def policy_backward(self, episode_observations, episode_hidden, episode_probability):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(episode_hidden.T, episode_probability)
        dh = np.outer(episode_probability, self.model['W2'])
        dh[episode_hidden <= 0] = 0  # backprop relu
        dW1 = np.dot(dh.T, episode_observations)
        return {'W1': dW1, 'W2': dW2}

    def run(self, nr_games=100):
        env = gym.make(self.game_name)
        running_reward = self.evaluate(nr_games=200)

        # update buffers that add up gradients over a batch and rmsprop memory
        gradient_buffer = {k: np.zeros_like(v, dtype=np.float) for k, v in self.model.items()}

        for episode in range(nr_games):
            score = 0
            memory = list()
            obs = env.reset()
            done = False

            while not done:
                # Render environment
                if self.render:
                    env.render()

                # Calculate forward policy
                action_probability, hidden = self.policy_forward(obs)
                action = 1 if np.random.uniform() < action_probability else 0

                next_obs, reward, done, info = env.step(action)
                score += reward

                # grad that encourages the action that was taken to be taken
                # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
                probability = action - action_probability
                memory.append(self.transition_(obs, hidden, probability, reward))

                obs = next_obs

            # Convert memory to a stack
            observations, hiddens, probabilities, rewards = np.array(list(zip(*memory)), dtype=np.object)
            observations = np.array(list(observations), dtype=np.float)
            hiddens = np.array(list(hiddens), dtype=np.float)

            # Calculate discounted rewards
            discounter_reward = self.discount_rewards(rewards, self.gamma)

            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounter_reward -= np.mean(discounter_reward)
            discounter_reward /= np.std(discounter_reward)

            # modulate the gradient with advantage (PG magic happens right here.)
            probabilities *= discounter_reward
            grad = self.policy_backward(observations, hiddens, probabilities)

            # accumulate grad over batch
            for weight in self.model:
                gradient_buffer[weight] += np.array(grad[weight], dtype=np.float)

            if episode % self.batch_size == 0:
                for layer, weights in self.model.items():
                    self.model[layer] += self.learning_rate * gradient_buffer[layer]
                    gradient_buffer[layer] = np.zeros_like(weights)

            running_reward = running_reward * 0.99 + score * 0.01
            print(f"Episode {episode:6d}, score: {score: 4.0f}, running mean: {running_reward: 6.2f}")

            if self.save_model and episode % self.save_interval == 0:
                self.save(self.save_path)

        env.close()

    def evaluate(self, nr_games=100):
        """ Evaluate the model results.  """
        env = gym.make(self.game_name)

        collected_scores = []

        for episode in range(1, nr_games + 1):

            obs = env.reset()
            done = False
            score = 0

            while not done:
                # Get action from model
                action = self.action(obs)

                # update everything
                obs, reward, done, info = env.step(action)
                score += reward

            print(f"\r\tGame {episode:3d}/{nr_games:3d} score: {score}", end='')

            collected_scores.append(score)

        average = sum(collected_scores) / nr_games
        print(f"\n\nThe model played: {nr_games} games, with an average score of: {average: 5.2f}")
        return average


if __name__ == '__main__':
    agent = PGCartPole(game_name='CartPole-v0')
    agent.run(nr_games=2_000)
    agent.save(agent.save_path)
    agent.evaluate(nr_games=100)