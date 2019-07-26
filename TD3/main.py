import gym
import roboschool
import numpy as np
from td3 import TD3, ReplayMemory

import torch


def train_td3(max_timesteps=1000000):
    """
    Train a td3 policy
    """
    env = gym.make("RoboschoolWalker2d-v1") # Edit here to change the environment used.
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    policy = TD3(state_dim, action_dim, max_action)
    memory = ReplayMemory()

    timestep = 0                # How many time steps the algorithm has run for so far
    episode_num = 0             # How many episodes have elapsed so far
    done = True                 # Flag that dictates if an episode has finished or not
    mb_size = 100               # Mini batch size for td3 updates
    gamma = 0.99                # gamma factor
    tau = 0.005                 # Target network update rate
    p_noise = 0.2               # Adds noise to the target action, prevents exploitation of Q-function errors
    noise_clip = 0.5            # Clips size of the noise
    delay_update_freq = 2       # Frequency of delayed policy updates
    start_step = 1000           # How long to wait whilst filling the replay memory
    e_noise = 0.1               # Standard deviation of the exploration noise

    while timestep < max_timesteps:
        if done:
            if timestep != 0:
                # Show episode statstics and update policy
                print('Current Timestep: {} \t Episode Number: {} \t Episode Length: {} \t Epiode Reward: {}'.format(timestep, episode_num, episode_timesteps, episode_reward))
                policy.train(memory, episode_timesteps, mb_size, gamma, tau, p_noise, noise_clip, delay_update_freq)

            # Reset for a new episode
            state = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        
        # Initialise the memory with some random actions to begin with
        if timestep < start_step:
            action = env.action_space.sample()
        # Use policy after initialising memory with sufficient items
        else:
            action = policy.select_action(np.array(state))
            if e_noise != 0: 
                action = (action + np.random.normal(0, e_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Take action
        next_state, reward, done, _ = env.step(action)
        if episode_timesteps + 1 == max_timesteps:
            done_flag = 0 # done_flag stores if episode is done or not when storing in replay memory
        else:
            done_flag = float(done)
        episode_reward += reward

        # Store the experience in the replay memory
        memory.add((state, next_state, action, reward, done_flag))

        # Set the state to be the next state ready for next timestep
        state = next_state
        episode_timesteps += 1
        timestep += 1

    # Save the policy after training
    policy.save("td3_walkerd2", directory="./pytorch_models")

if __name__ == "__main__":
    train_td3()
