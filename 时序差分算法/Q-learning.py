#在线策略算法：行为策略和目标策略是同一个策略
#离线策略算法：行为策略和目标策略不是同一个策略

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CliffWalkingEnv:
    def __init__(self,ncol,nrow):
        super(CliffWalkingEnv, self).__init__()

        self.nrow=nrow
        self.ncol=ncol

        self.x=0
        self.y=self.nrow-1


    def step(self,action):
        change=[[0,-1],[0,1],[-1,0],[1,0]]

        self.x=min(self.ncol-1,max(0,self.x+change[action][0]))
        self.y=min(self.nrow-1,max(0,self.y+change[action][1]))

        next_state=self.y*self.ncol+self.x
        done=False
        reward=-1

        if self.y==self.nrow-1 and self.x>0:
            done=True
            if self.x!=self.ncol-1:
                reward=-100

        return next_state,reward,done

    def reset(self):
        self.x=0
        self.y=self.nrow-1
        return self.y*self.ncol+self.x

class QLearning:
    def __init__(self,ncol,nrow,epsilon,alpha,gamma,n_action=4):
        super(QLearning, self).__init__()
        self.nrow=nrow
        self.ncol=ncol
        self.epsilon=epsilon
        self.alpha=alpha
        self.gamma=gamma
        self.n_action=n_action

        self.Q_table=np.zeros([self.nrow*self.ncol,n_action])

    def take_action(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.n_action)
        else:
            action=np.argmax(self.Q_table[state])
        return action

    def best_action(self,state):
        Q_max=np.max(self.Q_table[state])
        a=[0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state,i]==Q_max:
                a[i]=1
        return a

    def update(self,s0,a0,r,s1):
        td_error=r+self.gamma*self.Q_table[s1].max()-self.Q_table[s0,a0]
        self.Q_table[s0,a0]+=self.alpha*td_error


np.random.seed(0)
epsilon=0.1
alpha=0.1
gamma=0.9

env=CliffWalkingEnv(ncol=12,nrow=4)
agent=QLearning(ncol=12,nrow=4,epsilon=epsilon,alpha=alpha,gamma=gamma)
num_episodes=500

return_list=[]
for i in range(10):
    with tqdm(total=int(num_episodes/10),desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return=0
            state=env.reset()
            done=False

            while not done:
                action=agent.take_action(state)
                next_state,reward,done=env.step(action)
                episode_return+=reward
                agent.update(state,action,reward,next_state)
                state=next_state

            return_list.append(episode_return)

            if(i_episode+1)%10==0:
                pbar.set_postfix({
                    'episode':'%d' % (num_episodes/10*i+i_episode+1),
                    'return':'%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list=list(range(len(return_list)))
plt.plot(episodes_list,return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Q-learning on {}'.format('Cliff Walking'))
plt.show()

def print_agent(agent,env,action_meaning,disaster,end):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i*env.ncol+j) in disaster:
                print('****',end=' ')
            elif (i*env.ncol+j) in end:
                print('EEEE',end=' ')
            else:
                a=agent.best_action(i*env.ncol+j)
                pi_str=''
                for k in range(len(action_meaning)):
                    pi_str+=action_meaning[k] if a[k]>0 else 'o'
                print(pi_str,end=' ')
        print()

action_meaning=['^','v','<','>']
print('Q-learning 算法最终收敛得到的策略为：')
print_agent(agent,env,action_meaning,list(range(37,47)),[47])

```
Iteration 0: 100%|██████████| 50/50 [00:00<00:00, 1698.48it/s, episode=50, return=-105.700]
Iteration 1: 100%|██████████| 50/50 [00:00<00:00, 2816.63it/s, episode=100, return=-70.900]
Iteration 2: 100%|██████████| 50/50 [00:00<00:00, 3842.49it/s, episode=150, return=-56.500]
Iteration 3: 100%|██████████| 50/50 [00:00<00:00, 4971.91it/s, episode=200, return=-46.500]
Iteration 4: 100%|██████████| 50/50 [00:00<00:00, 5871.25it/s, episode=250, return=-40.800]
Iteration 5: 100%|██████████| 50/50 [00:00<00:00, 7141.19it/s, episode=300, return=-20.400]
Iteration 6: 100%|██████████| 50/50 [00:00<00:00, 8331.62it/s, episode=350, return=-45.700]
Iteration 7: 100%|██████████| 50/50 [00:00<00:00, 9865.24it/s, episode=400, return=-32.800]
Iteration 8: 100%|██████████| 50/50 [00:00<00:00, 12492.71it/s, episode=450, return=-22.700]
Iteration 9: 100%|██████████| 50/50 [00:00<00:00, 12496.44it/s, episode=500, return=-61.700]
Q-learning 算法最终收敛得到的策略为：
^ooo ovoo ovoo ^ooo ^ooo ovoo ooo> ^ooo ^ooo ooo> ooo> ovoo 
ooo> ooo> ooo> ooo> ooo> ooo> ^ooo ooo> ooo> ooo> ooo> ovoo 
ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ovoo 
^ooo **** **** **** **** **** **** **** **** **** **** EEEE 

进程已结束,退出代码0
```
