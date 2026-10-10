"""Shared physical training/scoring arena with a four-legged pursuing robot."""
import math
import mujoco
import numpy as np
from competition.rules import (Rules, CONTROL_DT, ACTION_SCALE, JOINT_DAMPING,
                               TORQUE_LIMITS, HIGH_OBS_DIM)
from competition.scene import scene_xml
from roboquest.envs.go2_walk_env import STANDING_POS


class Arena:
    def __init__(self, rules=None):
        self.rules = rules or Rules()
        self.model = mujoco.MjModel.from_xml_string(scene_xml())
        self.data = mujoco.MjData(self.model)
        self.parts = {}
        for index, role in enumerate(('flee', 'oni')):
            root = self.model.joint(role+'_root')
            actuators = np.arange(index*12, (index+1)*12)
            joints = self.model.actuator_trnid[actuators, 0]
            self.parts[role] = dict(q=int(root.qposadr[0]), v=int(root.dofadr[0]),
                body=self.model.body(role+'_base_link').id, act=actuators,
                jpos=self.model.jnt_qposadr[joints], jvel=self.model.jnt_dofadr[joints])
            self.model.dof_damping[self.parts[role]['jvel']] = JOINT_DAMPING
            limits = np.tile(TORQUE_LIMITS, 4)
            self.model.actuator_forcelimited[actuators] = True
            self.model.actuator_forcerange[actuators] = np.column_stack([-limits, limits])
        self.last = {r:np.zeros(12) for r in self.parts}
        self._geom_role = {}
        for gid in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[gid])
            while body:
                for role, part in self.parts.items():
                    if body == part['body']:
                        self._geom_role[gid] = role
                body = int(self.model.body_parentid[body])
        self.renderer = None
        self.reset()

    def reset(self, seed=100, spawn='east'):
        if spawn not in ('east', 'north', 'west', 'south'):
            raise ValueError('Unknown spawn')
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        angle = ('east', 'north', 'west', 'south').index(spawn)*math.pi/2
        for role, part in self.parts.items():
            q = part['q']
            xy = (0., 0.) if role == 'flee' else tuple(self.rules.spawn_distance*np.array([math.cos(angle), math.sin(angle)]))
            yaw = 0. if role == 'flee' else angle+math.pi
            self.data.qpos[q:q+7] = [*xy,.27,math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
            # Noise only in articulated joints, identical for each role/seed.
            self.data.qpos[part['jpos']] = STANDING_POS + rng.uniform(-.01,.01,12)
            self.data.ctrl[part['act']] = STANDING_POS
            self.last[role] = np.zeros(12)
        self.ticks = 0
        self.fall_ticks = 0
        self.reason = None
        self.seed, self.spawn = seed, spawn
        mujoco.mj_forward(self.model, self.data)

    def position(self, role):
        q = self.parts[role]['q']
        return self.data.qpos[q:q+3].copy()

    def rotation(self, role):
        q = self.parts[role]['q']
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, self.data.qpos[q+3:q+7])
        return matrix.reshape(3,3)

    def walk_obs(self, role, command):
        p = self.parts[role]
        return np.concatenate([command, self.data.qvel[p['v']+3:p['v']+6],
            self.rotation(role).T @ np.array([0.,0.,-1.]),
            self.data.qpos[p['jpos']]-STANDING_POS,
            self.data.qvel[p['jvel']], self.last[role],
            math.sin(2*math.pi*1.5*self.data.time),
            math.cos(2*math.pi*1.5*self.data.time)]).astype(np.float32)

    def high_obs(self):
        xy = self.position('flee')
        oni = self.position('oni')
        rotation = self.rotation('flee')
        rel = rotation.T @ (oni-xy)
        p, o = self.parts['flee'], self.parts['oni']
        vel = rotation.T @ self.data.qvel[p['v']:p['v']+3]
        relvel = rotation.T @ (self.data.qvel[o['v']:o['v']+3]-self.data.qvel[p['v']:p['v']+3])
        h = self.rules.arena_half
        obs = np.array([rel[0],rel[1],np.linalg.norm((oni-xy)[:2]),
            xy[0],xy[1],rotation[0,0],rotation[1,0],*vel,
            h-xy[0],h-xy[1],h+xy[0],h+xy[1],
            max(0.,1-self.ticks*CONTROL_DT/self.rules.duration),
            *relvel[:2],oni[2],self.rotation('oni')[2,2]], dtype=np.float32)
        assert obs.shape == (HIGH_OBS_DIM,)
        return obs

    def direct_obs(self):
        return np.concatenate([self.walk_obs('flee',np.zeros(3)),self.high_obs()])

    def oni_action(self):
        """Fixed feedback + inverse-kinematic trot; never writes robot poses."""
        target = self.position('flee')-self.position('oni')
        local = self.rotation('oni').T @ target
        error = math.atan2(local[1], local[0])
        yaw_rate = float(np.clip(2*error,-.8,.8))
        speed = self.rules.oni_speed*max(0.,math.cos(error))
        return gait_action(self.ticks*CONTROL_DT, speed, yaw_rate)

    def step(self, flee_action):
        if self.reason:
            raise RuntimeError('Match already ended; reset before stepping')
        action = np.asarray(flee_action, dtype=float)
        if action.shape != (12,) or not np.all(np.isfinite(action)):
            raise ValueError('Joint action must have 12 finite values')
        oni_action = self.oni_action()
        for role, a in [('flee',action),('oni',oni_action)]:
            part = self.parts[role]
            a = np.clip(a,-1,1)
            limits = self.model.actuator_ctrlrange[part['act']]
            self.data.ctrl[part['act']] = np.clip(STANDING_POS+ACTION_SCALE*a,limits[:,0],limits[:,1])
            self.last[role] = a.copy()
        # Capture is checked at every physics step, not just the 5Hz policy step.
        captured = False
        for _ in range(round(CONTROL_DT/self.model.opt.timestep)):
            mujoco.mj_step(self.model,self.data)
            for c in self.data.contact[:self.data.ncon]:
                roles = {self._geom_role.get(int(c.geom1)),self._geom_role.get(int(c.geom2))}
                if roles == {'flee','oni'} and c.dist <= 0:
                    captured = True
                    break
            if captured:
                break
        self.ticks += 1
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            self.reason = 'physics_error'
        elif captured:
            self.reason = 'captured'
        else:
            pos = self.position('flee')
            self.fall_ticks = self.fall_ticks+1 if pos[2]<self.rules.fall_height else 0
            if np.any(np.abs(pos[:2])>self.rules.arena_half): self.reason = 'out_of_bounds'
            elif self.fall_ticks*CONTROL_DT >= self.rules.fall_seconds: self.reason = 'fell'
            elif self.data.time >= self.rules.duration-1e-9: self.reason = 'survived'
        return self.result()

    def result(self):
        return dict(seed=self.seed,spawn=self.spawn,oni_speed=self.rules.oni_speed,
                    survived_seconds=min(float(self.data.time),self.rules.duration),
                    reason=self.reason,rule_version=self.rules.version,rule_sha256=self.rules.digest)

    def render(self,width=480,height=480):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model,height=height,width=width)
        self.renderer.update_scene(self.data,camera='top_view')
        return self.renderer.render()

    def close(self):
        if self.renderer: self.renderer.close()


def gait_action(time, vx, omega=0., vy=0.):
    """Geometric diagonal gait with differential foot velocities for steering."""
    frequency = 1.5
    phase = (time*frequency+np.array([0.,.5,.5,0.]))%1
    swing = phase < .5
    u = np.where(swing,phase*2,(phase-.5)*2)
    factor = np.where(swing,-np.cos(np.pi*u),1-2*u)*1.6/(frequency*4)
    hip_x = np.array([.1934,.1934,-.1934,-.1934])
    hip_y = np.array([-.15,.15,-.15,.15])
    x = factor*(vx-omega*hip_y)
    y = factor*(vy+omega*hip_x)
    z = -.31+np.where(swing,.05*np.sin(np.pi*u),0.)
    if abs(vx)+abs(omega)+abs(vy)<.01:
        x[:]=0.; y[:]=0.; z[:]=-.31
    zz = -np.sqrt(z*z+y*y)
    knee = -np.arccos(np.clip((x*x+zz*zz-2*.213**2)/(2*.213**2),-1,1))
    thigh = np.arctan2(-x,-zz)-knee/2
    hip = np.arctan2(y,-z)
    target = np.column_stack([hip,thigh,knee]).reshape(-1)
    return np.clip((target-STANDING_POS)/ACTION_SCALE,-1,1).astype(np.float32)
