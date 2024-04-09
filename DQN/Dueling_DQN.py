import collections

import gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import random

class VAnet(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc_A=nn.Linear(hidden_dim,action_dim)
        self.fc_V=nn.Linear(hidden_dim,1)

    #A(s,a)=Q(s,a)-V(s)
    def forward(self,x):
        A=self.fc_A(F.relu(self.fc1(x)))
        V=self.fc_V(F.relu(self.fc1(x)))

        #Q值由V和A计算得到
        Q=V+A-A.mean(1).view(-1,1)
        return Q

class QNet(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        return self.fc2(F.relu(self.fc1(x)))

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=collections.deque(maxlen=capacity)

    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):
        transitions=random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done=zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done

    def size(self):
        return len(self.buffer)

class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,
                 device,dqn_type='VanillaQDN'):
        super().__init__()
        self.state_dim=state_dim
        self.hidden_dim=hidden_dim
        self.action_dim=action_dim

        self.learning_rate=learning_rate
        self.gamma=gamma
        self.epsilon=epsilon
        self.target_update=target_update
        self.device=device
        self.dqn_type=dqn_type

        self.count=0

        if self.dqn_type=='DoublingDQN':
            self.qnet=VAnet(state_dim,hidden_dim,action_dim).to(self.device)
            self.target_qnet=VAnet(state_dim,hidden_dim,action_dim).to(self.device)
        else:
            self.qnet=QNet(state_dim,hidden_dim,action_dim).to(self.device)
            self.target_qnet=QNet(state_dim,hidden_dim,action_dim).to(self.device)

        self.optimizer=torch.optim.Adam(self.qnet.parameters(),learning_rate)

    def take_action(self,state):
        if np.random.random() < self.epsilon:
            action=np.random.randint(self.action_dim)
        else:
            state=torch.tensor([state],dtype=torch.float).to(self.device)
            action=self.qnet(state).argmax().item()
        return action

    def max_q_value(self,state):
        state=torch.tensor([state],dtype=torch.float).to(self.device)
        return self.qnet(state).max().item()

    def update(self,transition_dict):
        states=torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        next_state=torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)

        actions=torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        dones=torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)

        q_values=self.qnet(states).gather(1,actions)

        if self.dqn_type=='DoubleDQN':
            max_action=self.qnet(next_state).max(1)[1].view(-1,1)
            max_next_q_values=self.target_qnet(next_state).gather(1,max_action)
        else:
            max_next_q_values=self.target_qnet(next_state).max(1)[0].view(-1,1)

        q_targets=rewards+self.gamma*max_next_q_values*(1-dones)

        loss=torch.mean(F.mse_loss(q_values,q_targets))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update==0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

        self.count+=1

def moving_avarage(a,window_size):
    cumulative_sum=np.cumsum(np.insert(a,0,0))
    middle=(cumulative_sum[window_size:]-cumulative_sum[:-window_size])/window_size

    r=np.arange(1,window_size-1,2)
    begin=np.cumsum(a[:window_size-1])[::2]/r
    end=(np.cumsum(a[:-window_size:-1])[::2]/r)[::-1]
    return np.concatenate((begin,middle,end))

def train_QDN(agent,env,num_epsiodes,replay_buffer,minimal_size,batch_size):
    return_list=[]
    max_q_value_list=[]
    max_q_value=0

    for i in range(10):
        with tqdm(total=int(num_epsiodes/10),desc='Iteration %d' %i) as pbar:
            for i_epsiode in range(int(num_epsiodes/10)):
                episode_return=0
                state=env.reset()[0]
                done=False

                while not done:
                    action=agent.take_action(state)

                    #平滑处理
                    max_q_value=agent.max_q_value(state)*0.005+max_q_value*0.995
                    max_q_value_list.append(max_q_value)

                    action_continous=dis_to_con(action,env,agent.action_dim)

                    next_state,reward,_,done,_=env.step([action_continous])

                    replay_buffer.add(state,action,reward,next_state,done)

                    state=next_state

                    episode_return+=reward

                    if replay_buffer.size()>minimal_size:
                        b_s,b_a,b_r,b_ns,b_d=replay_buffer.sample(batch_size)
                        transition_dict={
                            'states':b_s,
                            'actions':b_a,
                            'rewards':b_r,
                            'next_states':b_ns,
                            'dones':b_d}
                        agent.update(transition_dict)

                return_list.append(episode_return)
                if (i_epsiode+1) % 10 ==0:
                    pbar.set_postfix({
                        'epsiode':'%d' % (num_epsiodes/10*i+i_epsiode+1),
                        'returns':'%.3f'%np.mean(return_list[-10:])})
                pbar.update(1)

    return return_list,max_q_value_list


lr=1e-2
num_episodes=200
hidden_dim=128
gamma=0.98

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

epsilon=0.01
target_update=50
buffer_size=5000
minimal_size=1000
batch_size=64
device=torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
replay_buffer=ReplayBuffer(buffer_size)

env_name='Pendulum-v1'
env=gym.make(env_name)
state_dim=env.observation_space.shape[0]
action_dim=11

def dis_to_con(discrete_action,env,action_dim):
    action_lowbound=env.action_space.low[0]
    action_upbound=env.action_space.high[0]
    return action_lowbound+(discrete_action/(action_dim-1)) * (action_upbound-action_lowbound)

agent=DQN(state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,device,'DuelingDQN')
return_list,max_q_value_list=train_QDN(agent,env,num_episodes,replay_buffer,minimal_size,batch_size)

epsiodes_list=list(range(len(return_list)))
mv_return=moving_avarage(return_list,5)

plt.plot(epsiodes_list,mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

frames_list=list(range(len(max_q_value_list)))
plt.plot(frames_list,max_q_value_list)
plt.axhline(0,c='orange',ls='--')
plt.axhline(10,c='red',ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('DQN on {}'.format(env_name))
plt.show()

#根据代码运行结果我们可以发现，相比于传统的DQN，Dueling DQN在多个选择下的学习更加稳定，得到回报的最大值也更大。由Dueling DQN
#原理可知，随着动作空间的增大，Dueling DQN相比于DQN的优势更为明显。
