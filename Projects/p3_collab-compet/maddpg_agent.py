import numpy as np 
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def transpose_to_list(mylist):
    return list(map(list, zip(*mylist)))

def transpose_to_tensor(input_list):
    make_tensor = lambda x: torch.tensor(x, dtype=torch.float)
    return list(map(make_tensor, zip(*input_list)))



class MADDPG:
    """Creates multiple angents to interacts with and learns from the environment 
    via DDPG(Deep Deterministic Policy Gradient) algorithm."""
    
    def __init__(self, state_size, action_size, 
                 random_seed=1, gamma=0.99, tau=1e-3,               
                 buffer_size=int(1e5), batch_size=64,  
                 lr_actor=1e-4, lr_critic=1e-3, weight_decay=0,
                 num_agents=2):
        """Initialize multiple agents object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            gamma (float): discount factor γ determines the importance of future rewards
            tau (float): for soft update of target parameters
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            lr_actor (float): learning rate of the critic
            lr_critic (float): learninig rate of the actor
            weight_decay (float): L2 weight decay
            num_agents (int): number of the multiple agents interacting with each other
        """
        super(MADDPG, self).__init__()
        
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        #self.iter = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, num_agents).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, num_agents).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # create agent to select action
        #self.agent = Agent(self.actor_local, self.actor_target)

        # Noise process
        self.noise = OUNoise(action_size, random_seed) 

        # Replay memory store experience of all multi-agents
        self.memory_all = ReplayBuffer(device, action_size, buffer_size, batch_size, random_seed)  
                                  

    def act(self, states_all_agents, noise=0.0, preprocess=False):
        """get actions from all agents in the MADDPG object"""
        if preprocess:
            states_all_agents = torch.from_numpy(states_all_agents).float().to(device)  
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states_all_agents).cpu().data.numpy()
        self.actor_local.eval()
        actions += noise * self.noise.sample() 
        return np.clip(actions, -1, 1)

    
    def target_act(self, states_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object"""
        # target actor of each agent: ddpg_agent.target_local
        self.actor_target.eval()
        with torch.no_grad():
            target_actions = self.actor_target(states_all_agents).cpu().data.numpy()
        self.actor_target.eval()
        target_actions += noise * self.noise.sample() 
        return np.clip(target_actions, -1, 1)

                                                            
    def reset(self):
        self.noise.reset()


    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn.
        Note: all input should be in the same data type.
        """
        # Save experience / reward
        self.memory_all.add(states, actions, rewards, next_states, dones)
        
        # Learn, if enough samples are available in memory
        if len(self.memory_all) > self.batch_size:
            experience_samples = self.memory_all.sample()
            for agent_i in range(self.num_agents):
                self.learn(experience_samples, agent_i)
            # Update target network after learning
            self.soft_update(self.critic_local, self.critic_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau) 



    def learn(self, experience_samples, agent_i):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): experience tuple of 
                                (S, S_full, A, R, S', done) of all agents
            agent_i (int): serial number of the agent to learn
        """
        states, states_full, actions, actions_full, rewards, next_states, next_states_full, dones = experience_samples

        # ---------------------------- update actor ---------------------------- #
        # Get predicted next-state actions [A(t+1)]
        next_actions_targets = self.target_act(next_states.t())
        #next_actions_targets_full = torch.cat(transpose_to_tensor(transpose_to_list(next_actions_targets)), dim=1)
        next_actions_targets_full = torch.from_numpy(np.hstack(next_actions_targets)).float().to(device)
        # Q[S(t+1), A(t+1)] targets
        next_Q_targets = self.critic_target(next_states_full, next_actions_targets_full)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards.t()[agent_i].view(-1,1) + (self.gamma * next_Q_targets * (1 - dones.t()[agent_i].view(-1,1)))
        # Compute critic loss = batch mean of (Q_targets - Q(s,a)_targets
        #actions_full = torch.tensor(np.stack([a.reshape(-1) for a in actions]), dtype=torch.float)
        #actions_full = torch.from_numpy(np.stack([a.reshape(-1) for a in actions])).float().to(device)
        Q_expected = self.critic_local(states_full, actions_full)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimise the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # use gradient clipping when training the critic network
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        # Compute actor loss
        actions_pred = self.act(states.t())
        #actions_pred_full = torch.cat(transpose_to_tensor(transpose_to_list(actions_pred)), dim=1)
        actions_pred_full = torch.from_numpy(np.hstack(actions_pred)).float().to(device)
        actor_loss = -self.critic_local(states_full, actions_pred_full).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        
  
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_model(self):
        torch.save(self.actor_local.state_dict(), './Model_Weights/maddpg_actor.pth')
        torch.save(self.critic_local.state_dict(), './Model_Weights/maddpg_critic.pth') 

    def load_model(self):
        self.actor_local.load_state_dict(torch.load('./Model_Weights/maddpg_actor.pth'))
        self.critic_local.load_state_dict(torch.load('./Model_Weights/maddpg_critic.pth'))



class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        #dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "state_full", "action", "action_full", "reward", "next_state", "next_state_full", "done"])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience of all agents to memory."""
        states_full = np.hstack(states)
        actions_full = np.hstack(actions)
        next_states_full = np.hstack(next_states)
        e = self.experience(states, states_full, actions, actions_full, rewards, next_states, next_states_full, dones)
        self.memory.append(e)
    
    def sample(self):
        """Return the sample of experience batch of """
        experiences = random.sample(self.memory, k=self.batch_size)
        device = self.device

        # stack each field in dimension: [samples][paprallel_agents]
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        states_full = torch.from_numpy(np.stack([e.state_full for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).float().to(device)
        actions_full = torch.from_numpy(np.stack([e.action_full for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_states_full = torch.from_numpy(np.stack([e.next_state_full for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, states_full, actions, actions_full, rewards, next_states, next_states_full, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)