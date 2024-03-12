#有一个4*12的网格世界，每一个网格表示一个状态。智能体的起点是左下角的状态，终点是右下角的状态。
#智能体在每一个状态中都可以采取四种动作：上，下，左，右。
#如果智能体采取动作后碰到边界墙壁则状态不发生变化，否则就会进入下一个状态。
#环境中有一段悬崖，智能体掉入悬崖或者达到目标状态都会结束动作并回到起点。
#也就是说掉入悬崖或者达到目标状态是终止状态。
#智能体每走一步的奖励是-1，掉入悬崖的奖励是-100

#建立悬崖漫步算法环境

import copy

class CliffWalkingEnv:
    """
    悬崖漫步环境
    """
    def __init__(self,ncol=12,nrow=4):
        self.nrow=nrow#行
        self.ncol=ncol#列
        self.P=self.createP()
        #P[state][action]=[(p,next_state,reward,done)]#包含下一个状态，当前位置的奖励，done用来判定是否走到悬崖或者终点

    def createP(self):
        P=[[[] for j in range(4)] for i in range(self.nrow*self.ncol)] #48个状态，4个动作上下左右
        change=[[0,-1],[0,1],[-1,0],[1,0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    #位置在悬崖或者目标状态，因为无法继续交互，所以状态保持不变，任何奖励都是0
                    if i==self.nrow-1 and j>0:
                        P[i*self.ncol+j][a]=[(1,i*self.ncol+j,0,True)]
                        continue
                    #其他位置
                    next_x=min(self.ncol-1,max(0,j+change[a][0]))
                    next_y=min(self.nrow-1,max(0,i+change[a][1]))
                    new_state=next_y*self.ncol+next_x
                    reward=-1
                    done=False

                    #如果下一个位置是悬崖或者是终点
                    if next_y==self.nrow-1 and next_x>0:
                        done=True
                        if next_x!=self.ncol-1:
                            #下一个位置是悬崖
                            reward=-100
                    P[i*self.ncol+j][a]=[(1,new_state,reward,done)]
        return P

class PolicyIteration:
    """
    策略迭代
    """
    def __init__(self,env,theta,gamma):
        self.env=env
        self.theta=theta#策略评估收敛阈值
        self.gamma=gamma#折扣因子

        #初始化为均匀随机策略
        self.pi=[[0.25,0.25,0.25,0.25] for i in range(self.env.ncol*self.env.nrow)]
        #初始化价值为0
        self.v=[0]*self.env.ncol*self.env.nrow

    def polcy_evaluation(self):#策略评估
        cnt=1
        while 1:
            max_diff=0
            new_v=[0]*self.env.ncol*self.env.nrow
            for s in range(self.env.ncol*self.env.nrow):

                #计算动作价值函数Q(s,a)
                qsa_list=[]
                for a in range(4):
                    qsa=0
                    for res in self.env.P[s][a]:
                        p,next_state,r,done=res
                        #环境下的下一个状态，奖励，可行性
                        qsa+=p*(r+self.gamma*self.v[next_state]*(1-done))
                #计算新的状态价值函数V(s)
                    qsa_list.append(self.pi[s][a]*qsa)
                new_v[s]=sum(qsa_list)
                max_diff=max(max_diff,abs(new_v[s]-self.v[s]))
            self.v=new_v
            if max_diff<self.theta: break
            cnt+=1

        print("策略评估进行%d轮后完成"%cnt)


    def policy_improvement(self):#策略提升
        for s in range(self.env.ncol*self.env.nrow):
            qsa_list=[]
            for a in range(4):
                qsa=0
                for res in self.env.P[s][a]:
                    p,new_state,r,done=res
                    qsa+=p*(r+self.gamma*self.v[new_state]*(1-done))
                qsa_list.append(qsa)
            #Qsa list中会加入新的四个动作下的价值
            maxq=max(qsa_list)
            #计算有几个动作得到了最大的Q值
            cntq=qsa_list.count(maxq)
            #动作概率分解,最大概率均分，其余取0
            self.pi[s]=[1/cntq if q==maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):#策略迭代
        while 1:
            self.polcy_evaluation()
            old_pi=copy.deepcopy(self.pi)
            new_pi=self.policy_improvement()
            #策略不再发生改变的时候就停止
            if old_pi==new_pi:
                break

def print_agent(agent,action_meaning,disaster,end):
    print("状态价值：")

    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            #保持输出六个字符
            print('%6s' % ('%.3f' % agent.v[i*agent.env.ncol+j]),end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            #一些特殊的状态，比如悬崖
            if (i*agent.env.ncol+j) in disaster:
                print('****',end=' ')
            elif (i*agent.env.ncol+j) in end:
                print('EEEE',end=' ')
            else:
                action=agent.pi[i*agent.env.ncol+j]
                pi_str=''
                for k in range(len(action_meaning)):
                    pi_str+=action_meaning[k] if action[k]>0 else 'o'
                print(pi_str,end=' ')
        print()

#定义环境
env=CliffWalkingEnv()
action_meaning=['^','v','<','>']
theta=0.001
gamma=0.9
agent=PolicyIteration(env,theta,gamma)
agent.policy_iteration()
#12*4=48,从37，47是边界
print_agent(agent,action_meaning,list(range(37,47)),[47])

'''
策略评估进行60轮后完成
策略提升完成
策略评估进行72轮后完成
策略提升完成
策略评估进行44轮后完成
策略提升完成
策略评估进行12轮后完成
策略提升完成
策略评估进行1轮后完成
策略提升完成
状态价值：
-7.712 -7.458 -7.176 -6.862 -6.513 -6.126 -5.695 -5.217 -4.686 -4.095 -3.439 -2.710 
-7.458 -7.176 -6.862 -6.513 -6.126 -5.695 -5.217 -4.686 -4.095 -3.439 -2.710 -1.900 
-7.176 -6.862 -6.513 -6.126 -5.695 -5.217 -4.686 -4.095 -3.439 -2.710 -1.900 -1.000 
-7.458  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 
策略：
ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovoo 
ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovoo 
ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ovoo 
^ooo **** **** **** **** **** **** **** **** **** **** EEEE 
'''