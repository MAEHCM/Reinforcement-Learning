import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import time

class ClifWalkingEnv:
    def __init__(self,ncol,nrow):
        self.ncol=ncol
        self.nrow=nrow
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


class DynaQ:
    def __init__(self,ncol,nrow,epsilon,alpha,gamma,n_planning,n_action=4):
        self.Q_table=np.zeros([ncol*nrow,n_action])

        self.n_action=n_action

        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=epsilon

        ################################################################
        self.n_planning=n_planning# 执行Q-planning的次数，对应1次Q-learning
        self.model=dict()#环境模型
        ################################################################

    def take_action(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.n_action)
        else:
            action=np.argmax(self.Q_table[state])
        return action

    def q_learning(self,s0,a0,r,s1):
        td_error=r+self.gamma*self.Q_table[s1].max()-self.Q_table[s0,a0]
        self.Q_table[s0,a0]+=self.alpha*td_error


    def update(self,s0,a0,r,s1):
        #现有的
        self.q_learning(s0,a0,r,s1)
        self.model[(s0,a0)]=r,s1
        #过去的
        for _ in range(self.n_planning):#Q-planning 循环
            #随机选择曾经遇到过的状态动作对
            (s,a),(r,s_)=random.choice(list(self.model.items()))
            self.q_learning(s,a,r,s_)

def DynaQ_CliffWalking(n_planning):
    ncol=12
    nrow=4
    env=ClifWalkingEnv(ncol,nrow)

    epsilon=0.01
    alpha=0.1
    gamma=0.9

    agent=DynaQ(ncol,nrow,epsilon,alpha,gamma,n_planning)
    num_episodes=300

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
                        'episode':'%d'%(num_episodes/10*i+i_episode+1),
                        'return' :'%.3f'%np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list


np.random.seed(0)
random.seed(0)
n_planning_list=[0,2,20]

for n_planning in n_planning_list:
    print('Q-planning步数为:%d'%n_planning)
    time.sleep(0.5)
    return_list=DynaQ_CliffWalking(n_planning)

    episodes_list=list(range(len(return_list)))

    plt.plot(episodes_list,return_list,label=str(n_planning)+' planning steps')

plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Dyna-Q on {}'.format('Cliff Walking'))
plt.show()

```
Q-planning步数为:0
Iteration 0: 100%|██████████| 30/30 [00:00<00:00, 1270.71it/s, episode=30, return=-138.400]
Iteration 1: 100%|██████████| 30/30 [00:00<00:00, 1993.87it/s, episode=60, return=-64.100]
Iteration 2: 100%|██████████| 30/30 [00:00<00:00, 2999.29it/s, episode=90, return=-46.000]
Iteration 3: 100%|██████████| 30/30 [00:00<00:00, 3854.94it/s, episode=120, return=-38.000]
Iteration 4: 100%|██████████| 30/30 [00:00<00:00, 4119.47it/s, episode=150, return=-28.600]
Iteration 5: 100%|██████████| 30/30 [00:00<00:00, 5244.85it/s, episode=180, return=-25.300]
Iteration 6: 100%|██████████| 30/30 [00:00<00:00, 5998.72it/s, episode=210, return=-23.600]
Iteration 7: 100%|██████████| 30/30 [00:00<00:00, 7498.31it/s, episode=240, return=-20.100]
Iteration 8: 100%|██████████| 30/30 [00:00<00:00, 5998.72it/s, episode=270, return=-17.100]
Iteration 9: 100%|██████████| 30/30 [00:00<00:00, 9997.55it/s, episode=300, return=-16.500]
Q-planning步数为:2
Iteration 0: 100%|██████████| 30/30 [00:00<00:00, 937.79it/s, episode=30, return=-53.800]
Iteration 1: 100%|██████████| 30/30 [00:00<00:00, 1506.32it/s, episode=60, return=-37.100]
Iteration 2: 100%|██████████| 30/30 [00:00<00:00, 2160.71it/s, episode=90, return=-23.600]
Iteration 3: 100%|██████████| 30/30 [00:00<00:00, 2726.64it/s, episode=120, return=-18.500]
Iteration 4: 100%|██████████| 30/30 [00:00<00:00, 2854.56it/s, episode=150, return=-16.400]
Iteration 5: 100%|██████████| 30/30 [00:00<00:00, 3332.69it/s, episode=180, return=-16.400]
Iteration 6: 100%|██████████| 30/30 [00:00<00:00, 3749.04it/s, episode=210, return=-13.400]
Iteration 7: 100%|██████████| 30/30 [00:00<00:00, 3749.15it/s, episode=240, return=-13.200]
Iteration 8: 100%|██████████| 30/30 [00:00<00:00, 3995.08it/s, episode=270, return=-13.200]
Iteration 9: 100%|██████████| 30/30 [00:00<00:00, 4417.38it/s, episode=300, return=-13.500]
Q-planning步数为:20
Iteration 0: 100%|██████████| 30/30 [00:00<00:00, 358.86it/s, episode=30, return=-18.500]
Iteration 1: 100%|██████████| 30/30 [00:00<00:00, 614.29it/s, episode=60, return=-13.600]
Iteration 2: 100%|██████████| 30/30 [00:00<00:00, 679.43it/s, episode=90, return=-13.000]
Iteration 3: 100%|██████████| 30/30 [00:00<00:00, 629.62it/s, episode=120, return=-13.500]
Iteration 4: 100%|██████████| 30/30 [00:00<00:00, 649.74it/s, episode=150, return=-13.500]
Iteration 5: 100%|██████████| 30/30 [00:00<00:00, 637.03it/s, episode=180, return=-13.000]
Iteration 6: 100%|██████████| 30/30 [00:00<00:00, 660.61it/s, episode=210, return=-22.000]
Iteration 7: 100%|██████████| 30/30 [00:00<00:00, 633.40it/s, episode=240, return=-23.200]
Iteration 8: 100%|██████████| 30/30 [00:00<00:00, 627.25it/s, episode=270, return=-13.000]
Iteration 9: 100%|██████████| 30/30 [00:00<00:00, 646.72it/s, episode=300, return=-13.400]
```
