"""PPO uses the same Arena as official matches, including the physical oni."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from competition.arena import Arena
from competition.rules import Rules, HIGH_STEPS


class FleeEnv(gym.Env):
    def __init__(self,walk_policy,reward=None):
        self.arena=Arena()
        self.walk_policy=walk_policy
        self.reward={'survival':1.,'distance':.1,'fall_penalty':20.,'tag_penalty':20.,**(reward or {})}
        self.observation_space=spaces.Box(-np.inf,np.inf,(19,),np.float32)
        self.action_space=spaces.Box(-1.,1.,(3,),np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.arena.reset(int(self.np_random.integers(1000,2**31-1)),
                         ('east','north','west','south')[int(self.np_random.integers(4))])
        return self.arena.high_obs(),{}

    def step(self,action):
        command=np.clip(action,-1,1)*np.array([.5,.3,1.])
        before=self.arena.data.time
        for _ in range(HIGH_STEPS):
            self.arena.step(self.walk_policy(self.arena.walk_obs('flee',command)))
            if self.arena.reason:break
        obs=self.arena.high_obs()
        reward=self.reward['survival']*(self.arena.data.time-before)/.2
        reward+=self.reward['distance']*obs[2]*.01
        if self.arena.reason=='captured':reward-=self.reward['tag_penalty']
        if self.arena.reason in ('fell','out_of_bounds'):reward-=self.reward['fall_penalty']
        return obs,reward,bool(self.arena.reason and self.arena.reason!='survived'),self.arena.reason=='survived',self.arena.result()

    def close(self):self.arena.close()
