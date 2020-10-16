import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

class ActorCriticModel(nn.Module):
    def __init__(self,num_observations,num_actions,num_hidden):
        super().__init__()

        self.fc1 = nn.Linear(num_observations,num_hidden)
        self.actor = nn.Linear(num_hidden,num_actions)
        self.critic = nn.Linear(num_hidden,1)
    
    def forward(self,observations):
        x = F.relu(self.fc1(observations))

        action_probs = F.softmax(self.actor(x))
        critic_value = self.critic(x)

        return action_probs,critic_value


env = gym.make('CartPole-v0')
model = ActorCriticModel(4,env.action_space.n,128)
gamma = 0.99
running_reward = 0
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
epizode_num = 0
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


while True:
    epizode_num+=1
    done = False
    state = env.reset()

    episode_action_prob = []
    critic_values = []
    episode_rewards = []

    while not done:
        action_probs,value = model(torch.squeeze(torch.tensor(state,dtype=torch.float),dim=-1))
        action_distribution = torch.distributions.categorical.Categorical(action_probs)
        action = action_distribution.sample()

        episode_action_prob.append(action_distribution.log_prob(action))
        critic_values.append(value)

        state,reward,done,_ = env.step(action.item())
        episode_rewards.append(reward)

        if epizode_num%10 == 0:
            env.render()


    discounted_rewards = []
    temp = 0 
    for i in episode_rewards[::-1]:
        temp = i + gamma*temp
        discounted_rewards.insert(0,temp)

    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards))/(np.std(discounted_rewards)+eps)
    running_reward = running_reward*0.95 + np.sum(episode_rewards)*0.05
    
    discounted_rewards = torch.unsqueeze(torch.tensor(discounted_rewards,dtype=torch.float),dim=0)
    episode_action_prob = torch.unsqueeze(torch.stack(episode_action_prob),dim=0)
    critic_values = torch.stack(critic_values).view((1,-1))

    # print(discounted_rewards.shape,episode_action_prob.shape,critic_values.shape)

    advantages = discounted_rewards - critic_values
    critic_loss = F.smooth_l1_loss(critic_values,discounted_rewards)
    actor_loss = -episode_action_prob * advantages
    loss = torch.mean(actor_loss) + torch.mean(critic_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    

    if epizode_num%10 == 0:
        print(f"{epizode_num} runing reward {running_reward}")
    gc.collect()
    # print(critic_loss,actor_loss)

    if running_reward>=195:
        print(f"Solved at epizode {epizode_num}!")
        break


while True:
    done = False
    state = env.reset()
    while not done:
        action_probs,_ = model(torch.squeeze(torch.tensor(state,dtype=torch.float),dim=-1))
        action = np.argmax(torch.squeeze(action_probs.detach(),dim=0).numpy())
        env.render()
        state,_,done,_ = env.step(action)
