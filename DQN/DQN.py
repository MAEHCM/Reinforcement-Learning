import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt

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


###定义一层Q网络
class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.f1c=torch.nn.Linear(state_dim,hidden_dim)
        self.f2c=torch.nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x=F.relu(self.f1c(x))
        return self.f2c(x)

def moving_avarge(a,window_size):
    cumulative_sum=np.cumsum(np.insert(a,0,0))
    middle=(cumulative_sum[window_size:]-cumulative_sum[:-window_size])/window_size

    r=np.arange(1,window_size-1,2)
    begin=np.cumsum(a[:window_size-1])[::2]/r
    end=(np.cumsum(a[:-window_size:-1])[::2]/r)[::-1]
    return np.concatenate((begin,middle,end))


###定义DQN算法
class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device):
        self.state_dim=state_dim
        self.hidden_dim=hidden_dim
        self.action_dim=action_dim
        self.learning_rate=learning_rate
        self.gamma=gamma
        self.epsilon=epsilon
        self.target_update=target_update
        self.device=device

        self.q_net=Qnet(self.state_dim,self.hidden_dim,self.action_dim).to(device)
        self.target_qnet=Qnet(self.state_dim,self.hidden_dim,self.action_dim).to(device)

        self.optimizer=torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)

        self.count=0

    def take_action(self,state):
        #epsilon 贪婪策略采取动作
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.action_dim)
        else:
            state=torch.tensor(state,dtype=torch.float).to(self.device)
            #torch.Size([4])
            action=self.q_net(state).argmax().item()
            #tensor(0)
        return action

    def update(self,transition_dict):
        states=torch.tensor(transition_dict['state'],dtype=torch.float).to(self.device)
        #torch.Size([64, 4])
        actions=torch.tensor(transition_dict['action']).view(-1,1).to(self.device)
        #torch.Size([64, 1])
        rewards=torch.tensor(transition_dict['reward'],dtype=torch.float).view(-1,1).to(self.device)
        #torch.Size([64, 1])
        next_states=torch.tensor(transition_dict['next_state'],dtype=torch.float).to(self.device)
        #torch.Size([64, 4])
        dones=torch.tensor(transition_dict['done'],dtype=torch.float).view(-1,1).to(self.device)
        #torch.Size([64, 1])


        q_values=self.q_net(states).gather(1,actions)#action [64,1]
        #torch.Size([64, 2]) -> torch.Size([64, 1])

        #下一个状态的最大Q值 torch.Size([64, 2])->torch.Size([64])->torch.Size([64, 1])
        max_next_q_values=self.target_qnet(next_states).max(1)[0].view(-1,1)
        q_targets=rewards+self.gamma*max_next_q_values*(1-dones)

        loss=torch.mean(F.mse_loss(q_values,q_targets))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update==0:
            self.target_qnet.load_state_dict(self.q_net.state_dict())#更新目标网络参数

        self.count+=1

lr=2e-3
num_episodes=500
hidden_dim=128
gamma=0.98
epsilon=0.01
target_update=10
buffer_size=10000
minimal_size=500
batch_size=64

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name='CartPole-v1'
env=gym.make(env_name)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

replay_buffer=ReplayBuffer(buffer_size)
state_dim=env.observation_space.shape[0]
#4
action_dim=env.action_space.n
#2

agent=DQN(state_dim,hidden_dim,action_dim,
          lr,gamma,epsilon,target_update,device
          )

return_list=[]
for i in range(10):
    with tqdm(total=int(num_episodes/10),desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes/10)):#50

            episode_return=0
            state=env.reset()[0]
            done=False
            while not done:
                action=agent.take_action(state)
                next_state,reward,done,_,_=env.step(action)
                replay_buffer.add(state,action,reward,next_state,done)
                state=next_state
                episode_return+=reward

                if replay_buffer.size()>minimal_size:
                    b_state,b_action,b_reward,b_next_state,b_done=replay_buffer.sample(batch_size)

                    transition_dict={
                        'state':b_state,
                        'action':b_action,
                        'reward':b_reward,
                        'next_state':b_next_state,
                        'done':b_done
                    }

                    agent.update(transition_dict)

            return_list.append(episode_return)
            if (i_episode+1) % 10==0:
                pbar.set_postfix({
                    'episode':'%d' % (num_episodes/10*i+i_episode+1),
                    'return':'%.3f'% np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list=list(range(len(return_list)))
plt.plot(episodes_list,return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return=moving_avarge(return_list,9)
plt.plot(episodes_list,mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()


```
import numpy as np

def moving_avarge(a,window_size):
    #insert:[0 1 2 3 3 4 5 4 5 6 5 6 7]
    cumulative_sum=np.cumsum(np.insert(a,0,0))
    #cumsum:[0 1 3 6 9 13 18 22 27 33 38 44 51]
    middle=(cumulative_sum[window_size:]-cumulative_sum[:-window_size])/window_size
    #[ 6  9 13 18 22 27 33 38 44 51]-[ 0  1  3  6  9 13 18 22 27 33]
    #[ 6  8 10 12 13 14 15 16 17 18]/3
    #[2. 2.66666667 3.33333333 4. 4.33333333 4.66666667 5. 5.33333333 5.66666667 6. ]

    r=np.arange(1,window_size-1,2)
    #[1]

    #print(np.cumsum(a[:window_size-1])[::2])
    #[ 1  3  6  9 13 18 22 27 33 38 44 51]
    #[ 1  6 13 22 33 44]
    begin=np.cumsum(a[:window_size-1])[::2]/r

    #[ 4  9 15 20 26 33 34 36 39 42 46 51]
    #[ 4 15 26 34 39 46]
    end=(np.cumsum(a[:-window_size:-1])[::2]/r)[::-1]
    return np.concatenate((begin,middle,end))

a=np.array([[[1,2,3],
            [3,4,5]],
            [[4,5,6],
            [5,6,7]]])

window_size=3

b=moving_avarge(a,window_size)
print(b)
```



