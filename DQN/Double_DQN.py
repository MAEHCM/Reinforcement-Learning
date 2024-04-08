import collections
import random
import gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.f1c=nn.Linear(state_dim,hidden_dim)
        self.f2c=nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x=F.relu(self.f1c(x))
        return self.f2c(x)

class DQN:
    ''' DQN算法,包括Double DQN '''
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type='VanillaDQN'):
        self.action_dim = action_dim
        self.q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = QNet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else: # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

lr=1e-2
num_episodes=200
hidden_dim=128
gamma=0.98
epsilon=0.01

target_update=50
buffer_size=5000
minimal_size=1000
batch_size=64

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name='Pendulum-v1'
env=gym.make(env_name)

state_dim=env.observation_space.shape[0]
action_dim=11#连续动作被分成11个离散的动作

def dis_to_con(discreate_action,env,action_dim):
    #离散动作转回连续的函数
    action_lowbound=env.action_space.low[0]
    action_upbound=env.action_space.high[0]
    return action_lowbound+(discreate_action/(action_dim-1))*(action_upbound-action_lowbound)

def moving_avarge(a,window_size):
    cumulative_sum=np.cumsum(np.insert(a,0,0))
    middle=(cumulative_sum[window_size:]-cumulative_sum[:-window_size])/window_size

    r=np.arange(1,window_size-1,2)
    begin=np.cumsum(a[:window_size-1])[::2]/r
    end=(np.cumsum(a[:-window_size:-1])[::2]/r)[::-1]
    return np.concatenate((begin,middle,end))

def train_DQN(agent,env,
              num_epsiodes,replay_buffer,minimal_size,batch_size):
    return_list=[]
    max_q_value_list=[]
    max_q_value=0

    for i in range(10):
        with tqdm(total=int(num_epsiodes/10),desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_epsiodes/10)):
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
                        transition_dict={'states':b_s,
                                         'actions':b_a,
                                         'rewards':b_r,
                                         'next_states':b_ns,
                                         'dones':b_d}
                        agent.update(transition_dict)

                return_list.append(episode_return)

                if (i_episode+1)%10==0:
                    pbar.set_postfix({
                        'epsiode':'%d' %(num_epsiodes/10*i+i_episode+1),
                        'return' :'%.3f'%np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list,max_q_value_list


###首先定义经验回收池
class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=collections.deque(maxlen=capacity) #队列，先进先出

    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):
        transitions=random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done=zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done

    def size(self):
        return len(self.buffer)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

replay_buffer=ReplayBuffer(buffer_size)

#首先训练DQN
agent=DQN(state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,device)
return_list,max_q_value_list=train_DQN(agent,env,num_episodes,replay_buffer,minimal_size,batch_size)

episodes_list=list(range(len(return_list)))
mv_return=moving_avarge(return_list,5)

plt.plot(episodes_list,mv_return)
plt.xlabel('Epsiodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

#最大Q值

frames_list=list(range(len(max_q_value_list)))

plt.plot(frames_list,max_q_value_list)
plt.axhline(0,c='orange',ls='--')
plt.axhline(10,c='red',ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('DQN on {}'.format(env_name))
plt.show()

#根据代码运行结果我们可以发现，DQN算法在倒立摆环境中取得不错的回报，最后的期望回报在-200左右，但是有不少Q值超过了0
#有一些还超过了10，这现象便是DQN算法中的Q值过高估计。

#下面来看一下Double DQN是否能对此问题就行改善

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer=ReplayBuffer(buffer_size)
agent=DQN(state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,device,'DoubleDQN')
return_list,max_q_value_list=train_DQN(agent,env,num_episodes,replay_buffer,minimal_size,batch_size)

episodes_list=list(range(len(return_list)))
mv_return=moving_avarge(return_list,5)
plt.plot(episodes_list,mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Double DQN {}'.format(env_name))
plt.show()

frames_list=list(range(len(max_q_value_list)))
plt.plot(frames_list,max_q_value_list)
plt.axhline(0,c='orange',ls='--')
plt.axhline(10,c='red',ls='--')
plt.xlabel('Frames')
plt.ylabel('Q Value')
plt.title('Double DQN on {}'.format(env_name))
plt.show()

#我们可以发现与普通的DQN相比，Double DQN比较少出现Q值大于0的情况，说明Q值过高的估计问题得到了很大的缓解
