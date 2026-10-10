"""Approved RQ2026 rules. Training and scoring import the same constants."""
from dataclasses import asdict, dataclass
import hashlib
import json

VERSION = 'rq2026-1v1-v1'
CONTROL_DT = .02
HIGH_STEPS = 10
HIGH_OBS_DIM = 19
DIRECT_OBS_DIM = 64
WALK_OBS_DIM = 45
ACTION_SCALE = .6
JOINT_STIFFNESS = 60.
JOINT_DAMPING = 2.
TORQUE_LIMITS = (23.7, 23.7, 45.43)
SEEDS = (100, 101, 102, 103, 104)
SPAWNS = ('east', 'north', 'west', 'south')


@dataclass(frozen=True)
class Rules:
    version: str = VERSION
    duration: float = 60.
    arena_half: float = 2.5
    oni_speed: float = .3
    fall_height: float = .15
    fall_seconds: float = .5
    spawn_distance: float = 2.

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError('Unsupported rules version')
        if self.duration != 60 or self.arena_half != 2.5:
            raise ValueError('Official matches require 60 seconds and a 5m arena')
        if self.oni_speed not in (.3, .4, .5):
            raise ValueError('Official oni speed must be .3, .4 or .5 m/s')
        if (self.fall_height,self.fall_seconds,self.spawn_distance) != (.15,.5,2.):
            raise ValueError('Official fall and spawn conditions are fixed')

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


HIGH_OBSERVATIONS = (
    'oni_body_dx', 'oni_body_dy', 'oni_distance',
    'self_world_x', 'self_world_y', 'heading_cos', 'heading_sin',
    'self_body_vx', 'self_body_vy', 'self_body_vz',
    'wall_east', 'wall_north', 'wall_west', 'wall_south', 'time_left',
    'oni_relative_body_vx', 'oni_relative_body_vy', 'oni_height', 'oni_upright',
)
