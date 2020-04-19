import torch
import numpy as np
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from networks import Actor, Critic

class TD3Agent:
    def __init__(self, env, train_mode=True):
        self.env = env
        self.brain_name = env.brain_names[0]
        self.train_mode = train_mode

        
        self.max_action = 1
        self.policy_freq = 2
        self.policy_freq_it = 0
        self.batch_size = 512
        self.discount = 0.99
        self.replay_buffer = int(1e5)
        
        
        self.device = 'cuda'
        
        self.state_dim = 33
        self.action_dim = 4
        self.max_action = 1
        self.policy_noise = 0.1
        self.agents = 20
        
        self.tau = 5e-3
        
        self.replay_buffer = ReplayBuffer(self.replay_buffer)
        
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.total_score = 0
        
        self.state = self.reset()
    
    def select_action_with_noise(self, state):

        state = torch.FloatTensor(state.reshape(self.agents, -1)).to(self.device)
        
        action = self.actor(state).cpu().data.numpy()
        if self.policy_noise != 0: action = (action + np.random.normal(0, self.policy_noise, size=(state.shape[0], 4))) #ENcODE ? #Is policy noise and action nosie are the same ?

        return action.clip(-self.max_action,self.max_action)
    
    def make_a_step(self, action):
        env_info = self.env.step(action)[self.brain_name] 
        next_state = env_info.vector_observations         # get next state (for each agent)
        reward = env_info.rewards                         # get reward (for each agent)
        done = env_info.local_done                        # see if episode finished
        self.total_score += np.array(reward).mean()
        
        
        return next_state, reward, done
    
    def step(self):

        action = self.select_action_with_noise(self.state)   
        next_state, reward, done = self.make_a_step(action)
        self.done = done
        
        for i in range(self.state.shape[0]):
            self.replay_buffer.add(self.state[i], action[i], reward[i], next_state[i], done[i])
        self.state = next_state
        
        if len(self.replay_buffer)>1e4:
            # Sample mini batch
            s, a, r, s_, d = self.replay_buffer.sample(self.batch_size)

            state = torch.FloatTensor(s).to(self.device)
            action = torch.FloatTensor(a).to(self.device)
            next_state = torch.FloatTensor(s_).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)


            # Select action with the actor target and apply clipped noise
            noise = torch.FloatTensor(a).data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-0.1,0.1) # NOISE CLIP WTF?
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward.reshape(-1,1) + (done.reshape(-1,1) * self.discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.policy_freq_it % self.policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


            self.policy_freq_it += 1
        
        return True
        
    
    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        state = env_info.vector_observations
        self.total_score = 0
        self.policy_freq_it = 0
        return state
    