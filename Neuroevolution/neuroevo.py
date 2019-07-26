import torch
import torch.nn as nn
import numpy as np
import gym
import copy
import time
import pickle
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.l1 = nn.Linear(input_size, 64).to(self.device)
        self.l2 = nn.Linear(64, output_size).to(self.device)

    def forward(self, state):
        state = state.float()
        state = state.to(self.device)

        x = torch.tanh(self.l1(state))
        return torch.tanh(self.l2(x)).detach().cpu()

    def mutate(self, lr):
        for layer in [self.l1, self.l2]:
            additive_noise = copy.deepcopy(layer)
            nn.init.normal_(additive_noise.weight, mean=0, std=0.01)
            layer.weight = torch.nn.Parameter(layer.weight.detach() +
                                              lr * additive_noise.weight.detach()).to(self.device)
            additive_noise = None  # Make sure these variables don't hog memory

class Networks:
    def __init__(self, learning_rate, inits=None):
        #self.env_name = 'Walker2d-v2'
        self.env_name = "RoboschoolWalker2d-v1"

        # 15 is close to the limit of what one machine with 8GB of RAM can handle.
        # Edit only if >8GB.
        self.pop_size = 10
        self.env = gym.make(self.env_name)
        self.num_actions = self.env.action_space.sample().size
        self.state_size = self.env.reset().size

        print("Initialising networks")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if inits:
            self.networks = inits
        else:
            self.networks = [Model(self.state_size, self.num_actions).to(self.device) for i in range(self.pop_size)]
        # self.networks = self.networks.to(self.device)

        self.fitnesses = np.zeros(self.pop_size)
        self.elitism_prop = 0.9
        self.mutate_chance = 0.9
        self.lr = learning_rate
        self.crossover_chance = 0
        self.num_generations = 25

        self.total_timesteps = 0

    def update_learning_rate(self, gen_number):
        if gen_number % 10 == 0:
            self.lr *= 0.9
            print("Updated learning rate from", self.lr/0.9, "to", self.lr)

    def find_fitnesses(self):
        self.fitnesses = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            for j in range(2):
                tot_reward = 0
                state = self.env.reset()
                done = False
                while not done:
                    self.total_timesteps += 1
                    action = self.networks[i].forward(torch.from_numpy(state))
                    state, reward, done, _ = self.env.step(action)
                    tot_reward += reward
                self.fitnesses[i] += tot_reward/2

    def crossover(self, parents, replace):
        """
        This method is really messy.
        Nicer way is to put layers in lists and index through, making copies.
        """
        # parents is list with indices
        # replace is index
        layers = np.random.randint(0,2,5)
        self.networks[replace].l1 = copy.deepcopy(self.networks[parents[layers[0]]].l2).to(self.device)
        self.networks[replace].l2 = copy.deepcopy(self.networks[parents[layers[1]]].l1).to(self.device)

    @staticmethod
    def find_time(seconds_elapsed):
        h = seconds_elapsed//3600
        m = (seconds_elapsed%3600)//60
        s = np.around((seconds_elapsed%3600)%60, decimals=0)
        return h, m, s

    def GA(self):

        highest_fitnesses = []
        start_time = time.time()

        for i in range(self.num_generations):
            self.find_fitnesses()
            time_elapsed = np.around((time.time() - start_time), decimals = 2) # Number of minutes elapsed
            h, m, s = self.find_time(time_elapsed)
            highest_fitnesses.append(np.around(np.max(self.fitnesses), decimals=2))

            num_children = int(self.pop_size * self.elitism_prop)
            # Get the indices of worst num_children amount of individuals in the population
            worst_indices = np.argsort(self.fitnesses)[:num_children]

            for j in range(num_children):

                # Decide if crossover
                if np.random.random() < self.crossover_chance:
                    # Get two parents, and perform a crossover
                    parents = []
                    for _ in range(2):
                        # Tournament select each parent. Could change this to just pick best 2 in single tourn?
                        random_parent_indices = np.random.choice(self.pop_size, int(self.pop_size/4), replace = False)
                        parents.append(random_parent_indices[np.argmax(self.fitnesses[random_parent_indices])])
                    self.crossover(parents, worst_indices[j])
                else:
                    # Tournament selection for a single parent
                    random_parent_indices = np.random.choice(self.pop_size, int(self.pop_size/2), replace = False)
                    parent_idx = random_parent_indices[np.argmax(self.fitnesses[random_parent_indices])]
                    # Copy the parent across
                    self.networks[worst_indices[j]] = copy.deepcopy(self.networks[parent_idx]).to(self.device)

                # And possibly mutate after either crossover or copying across.
                if np.random.random() < self.mutate_chance:
                    self.networks[worst_indices[j]].mutate(self.lr)


class NetworkHierarchy:
    def __init__(self):
        self.lrs = np.logspace(-2, 1, 10) # Have 10 learning rates in a logspace from 0.01 to 10
        self.sub_pops = [Networks(self.lrs[i]) for i in range(10)]
        self.best_fitnesses = np.zeros(10)

        self.num_generations = 100

    def generation(self):
        for i in range(len(self.sub_pops)):
            self.sub_pops[i].GA() # Run a GA using that learning rate
            
            self.best_fitnesses = np.array([np.max(self.sub_pops[i].fitnesses) for i in range(10)])
            print("Updated sub-population", i+1, end='\r')

        worst_indices = np.argsort(self.best_fitnesses)[:9]
        for i in range(9):
            # Tournament select
            random_parent_indices = np.random.choice(10, int(10/3), replace=False)
            parent_idx = random_parent_indices[np.argmax(self.best_fitnesses[random_parent_indices])]
            # Replace with learning rate of chosen parent
            self.sub_pops[worst_indices[i]].lr = self.lrs[parent_idx]
            
            # Mutate
            self.sub_pops[worst_indices[i]].lr = np.random.normal(self.sub_pops[worst_indices[i]].lr, 1)

        self.lrs = [self.sub_pops[i].lr for i in range(10)]

    def run_hierarchy(self):

        highest_fitnesses = []
        for i in range(self.num_generations):
            print("Generation:", i, "Best fitnesses of populations are:", self.best_fitnesses, "max is",
                  np.max(self.best_fitnesses))
            highest_fitnesses.append(np.max(self.best_fitnesses))
            self.generation()

            save_name = 'hierarchical_walkerv2_model_at_generation_' + str(i) + '.pt'
            best_population_idx = np.argmax(self.best_fitnesses)
            best_net_idx = np.argmax(self.sub_pops[best_population_idx].fitnesses)

            torch.save( self.sub_pops[best_population_idx].networks[best_net_idx].state_dict(), save_name )

        ax = plt.subplot(111)
        plt.plot([i for i in range(self.num_generations)], highest_fitnesses)
        output = open('mujoco_hierarchy_2x64.pkl', 'wb')
        pickle.dump(ax, output)
        output.close()


if __name__ == "__main__":
    hierarchy = NetworkHierarchy()
    print("Initialised environments and networks - beginning GA")
    hierarchy.run_hierarchy()
