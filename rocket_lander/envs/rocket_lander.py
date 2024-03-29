__credits__ = ["Esteban calderón"]

import pygame
import pymunk
import pymunk.pygame_util
import math
from random import randint, randrange, random
import numpy as np
from typing import Optional
import gym
from gym import spaces

try:
    from colorama import Fore

except ModuleNotFoundError:
    print("Module named Colorama Not Found!")

"""

The objective of this environment is to land a rocket on a ship.

STATE VARIABLES
The state consists of the following variables:
    - x position
    - y position
    - angle
    - first leg ground contact indicator
    - second leg ground contact indicator
    - throttle
    - engine gimbal
If VEL_STATE is set to true, the velocities are included:
    - x velocity
    - y velocity
    - angular velocity
    
CONTROL INPUTS
Discrete control inputs are:
    - gimbal left
    - gimbal right
    - throttle up
    - throttle down
    - use first control thruster
    - use second control thruster
    - no action
    
Continuous control inputs are:
    - gimbal (left/right)
    - throttle (up/down)
    - control thruster (left/right)

"""

#TODO: visualization of the cold gas control thrusters using particles

pygame.init()

# Display Setup

VIEWPORT_WIDTH          = 1000#1920
VIEWPORT_HEIGHT         = 1020
CLOCK                   = pygame.time.Clock()
FONT                    = pygame.font.SysFont("ariel", 24)

FPS                     = 30 # Default FPS used if no fps is provided
SCALE                   = 1  # Temporal Scaling, lower is faster - adjust forces appropriately


# Environment Variables
CONTINUOUS              = False
MAX_STEP_NUMBER         = 1200
LANDING_TICKS           = 60

# Pymunk Space Setup
X_GRAVITY, Y_GRAVITY    = (0, 910 * SCALE) # Original gravity 956
DIFFICULTY_LEVEL        = 1 # Normal difficulty, the difficulty level affects the range of the starting pos, the STARTING_ANG_VEL_RANGE, and the CRASHING_SPEED
STARTING_POS            = (VIEWPORT_WIDTH//2, -200) # This is the mean starting pos
STARTING_POS_DEVIATION  = ((VIEWPORT_WIDTH * SCALE) / 3.3) * DIFFICULTY_LEVEL
STARTING_ANG_VEL_RANGE  = 500 * DIFFICULTY_LEVEL * SCALE

# Sky
SKY_COLOR               = (212, 234, 255)

MIN_THROTTLE            = 0.3
GIMBAL_THRESHOLD        = 0.15
MAIN_ENGINE_POWER       = 24000 * SCALE
SIDE_ENGINE_POWER       = 3000 * SCALE

# ROCKET
ROCKET_WIDTH            = 40 * SCALE
ROCKET_HEIGHT           = ROCKET_WIDTH * 5
ROCKET_SIZE             = (ROCKET_WIDTH, ROCKET_HEIGHT)
ROCKET_MASS             = 30 * SCALE
ROCKET_ELASTICITY       = 0.1
ROCKET_FRICTION         = 0.5
ROCKET_COLOR            = (161, 159, 159, 250)

# ENGINE
ENGINE_SIZE             = (ROCKET_WIDTH * 0.4, ROCKET_WIDTH * 0.5)
ENGINE_HEIGHT           = (ROCKET_HEIGHT/2) * 0.86
ENGINE_MASS             = ROCKET_MASS * 0.1
ENGINE_ELASTICITY       = 0.1
ENGINE_FRICTION         = 0.5
ENGINE_COLOR            = (111, 109, 109, 250)

# FIRE_WIDTH             = ROCKET_WIDTH * 3
# FIRE_HEIGHT            = FIRE_WIDTH * 3.4

# COLD_GAS_WIDTH         = FIRE_WIDTH/1.4
# COLD_GAS_HEIGHT        = COLD_GAS_WIDTH * 3

# CONTROL THRUSTERS
THRUSTER_HEIGHT         = (ROCKET_HEIGHT/2) * -0.86

# LEGS
LEG_HEIGHT              = ROCKET_SIZE[1] * 0.35
LEG_SPRING_HEIGHT       = ROCKET_SIZE[1] * 0.1
LEG_SIZE                = (ROCKET_SIZE[0] * 0.3, ROCKET_SIZE[1] * 0.4)
LEG_MASS                = 8 * SCALE
LEG_COLOR               = (220, 20, 30, 20)
LEG_ELASTICITY          = 0.3
LEG_FRICTION            = 0.6

# WATER
WATER_HEIGHT            = 80 * SCALE
WATER_COLOR             = (0, 157, 196, 180)

# LANDING PAD
LANDING_PAD_HEIGHT      = ROCKET_WIDTH * 0.6
LANDING_PAD_WIDTH       = LANDING_PAD_HEIGHT * 14
LANDING_PAD_SIZE        = (LANDING_PAD_WIDTH, LANDING_PAD_HEIGHT)
LANDING_PAD_POS         = (VIEWPORT_WIDTH / 2, VIEWPORT_HEIGHT - (WATER_HEIGHT) - (LANDING_PAD_SIZE[1]/2))

LANDING_PAD_ELASTICITY  = 0.3
LANDING_PAD_FRICTION    = 0.7
LANDING_PAD_COLOR       = (50, 64, 63, 150)

CRASHING_SPEED          = 0.005 / (DIFFICULTY_LEVEL) # The maximum y_vel that the lander can have at touch down, to avoid crashing.

# SMOKE FOR VISUALS
SMOKE_LIFETIME          = 0 # Lifetime
PARTICLE_TTL_SUBTRACT   = (1 / FPS)  # Amount to subtract ttl per frame
MAX_ENGINE_PARTICLES    = 85
PARTICLE_STARTING_TTL   = 1.0
SMOKE_RATE              = 0.98 # The rate at which the smoke gets generated. Range = [0 - PARTICLE_STARTING_TTL]
PARTICLE_GROWTH_RATE    = 20 / FPS
PARTICLE_MAX_RADIUS     = 50 * SCALE
PARTICLE_Y_VEL_RANGE    = [100, 300]
MAX_THRUSTER_PARTICLES  = 40
THRUSTER_PARTICLE_COLOR = (0, 0, 255, 255)
THRUSTER_PARTICLE_VEL   = 3.5
TRHUSTER_PARTICLE_RAD   = 4
THRUSTER_PARTICLE_MASS  = 0.9

# OTHER
DRAW_FLAGS              = False
DEBUG_FLAG              = False

class Rocket(gym.Env):
    f'''

    ### Description
    This environment is a classic rocket trajectory optimization problem.
    The objective of this environment is to land a rocket on a landing pad or ship.

    There are two environment versions: discrete or continuous.
    The landing pad is always at coordinates (0,0). The coordinates are the
    first two numbers in the state vector.
    Landing outside of the landing pad is NOT possible. Fuel is infinite but fuel consumption is penalized, so an agent
    can learn to fly and then land on its first attempt.

    ### Action Space
    There are four discrete actions available: do nothing, fire left
    orientation engine, fire main engine, fire right orientation engine.
    Discrete control inputs are:
        - gimbal left
        - gimbal right
        - throttle up
        - throttle down
        - use first control thruster
        - use second control thruster
        - no action
    
    Continuous control inputs are:
        - gimbal (left/right)
        - throttle (up/down)
        - control thruster (left/right)

    ### Observation Space
    The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
    velocities in `x` & `y`, its angle, its angular velocity, and two booleans
    that represent whether each leg is in contact with the ground or not.
    The state consists of the following variables:
        - x position        [-inf -- inf]
        - y position        [-inf -- inf]
        - x linear velocity [-inf -- inf]
        - y linear velocity [-inf -- inf]
        - angle             [-inf -- inf]
        - angular velocity  [-inf -- inf]
        # - throttle          [0 -- 1]
        # - engine gimbal     [{-GIMBAL_THRESHOLD} -- {GIMBAL_THRESHOLD}]
        - left leg ground contact indicator  [bool]
        - right leg ground contact indicator [bool]

    ### Rewards
    After every step a reward is granted. The total reward of an episode is the
    sum of the rewards for all the steps within that episode.

    ### Version History
    - v0: Initial version
    - v1: ...

    '''

    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": FPS
    }

    def __init__(
        self, 
        space:       Optional[pymunk.Space] = None,
        render_mode: Optional[str]          = None,
        gravity:     tuple                  = (0, 20),
        clock:       Optional[bool]         = False,
        fps:         Optional[int]          = None

    ) -> None:
        '''
        Creates a Rocket environment

        :py:data:`Hello World!`

        :Parameters:
                space : pymunk.Space
                    The basic unit of the pymunk simulation
                render_mode : str
                    The render mode for the simulation
                gravity : tuple
                    The gravitational force applied to all dynamic bodies in the simulation
                clock : bool
                    If True then pygame Clock is going to be initialized inside the environment
        '''
        super(Rocket, self).__init__()

        if space is None:
            self.space = pymunk.Space()

        else:
            self.space = space

        self.space.gravity = gravity

        if (render_mode != None) and (render_mode not in self.metadata['render_modes']):
            raise Exception(f"The Render Mode provided is not available. \n Available Render Modes = {self.metadata['render_modes']}")

        if render_mode == "human":
            self.screen = pygame.display.get_surface()

            if self.screen is None:
                self.screen   = pygame.display.set_mode((VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
        
        if clock:
            self.clock    = pygame.time.Clock()
            self.fps      = fps if fps else self.metadata["render_fps"]

        else:
            self.clock    = None
            self.fps      = None

        self.dt           = 1 / self.fps
        self.render_mode  = render_mode

        # low = np.array(
        #     [
        #         # these are bounds for position
        #         # realistically the environment should have ended
        #         # long before we reach more than 50% outside
        #         -1.5,
        #         -1.5,
        #         # velocity bounds is 5x rated speed
        #         -1.0,
        #         -1.0,
        #         -math.pi,
        #         -5.0,
        #         -0.0,
        #         -0.0,
        #     ],
        #     dtype = np.float32
        # )

        # high = np.array(
        #     [
        #         # these are bounds for position
        #         # realistically the environment should have ended
        #         # long before we reach more than 50% outside
        #         1.5,
        #         1.5,
        #         # velocity bounds is 1x rated speed
        #         1.0,
        #         1.0,
        #         math.pi,
        #         5.0,
        #         1.0,
        #         1.0,
        #     ],
        #     dtype = np.float32
        # )

        # useful range is -1 .. +1, but spikes can be higher
        #self.observation_space = spaces.Box(low, high, shape = (8,), dtype = np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(8,), dtype=np.float32
        )
        
        if CONTINUOUS:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 0% to 100% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        else:
            # TODO: Make main engine throttle 0 or 1 (binary action) only. Easier for the ai to learn
            # Gimbal left, Gimbal right, Increase Throttle, Decrease Throttle, left control thruster, right control thruster, and No action
            self.action_space = spaces.Discrete(7)
        
        self.reset()
    
    def _setup(self,):

        self.water                   = self._create_water()
        
        self.lander, self.mainEngine = self._create_lander()

        self.legs                    = []
        self.leg_contacts            = []
        self._create_legs(self.lander.body)

        self.landing_pad             = self._create_landing_pad()
    
    def _create_lander(self):
        # Rocket
        size             = ROCKET_SIZE
        pos              = self.starting_pos

        inertia          = pymunk.moment_for_box(mass = ROCKET_MASS, size = ROCKET_SIZE)

        body             = pymunk.Body(mass = ROCKET_MASS, moment = inertia, body_type = pymunk.Body.DYNAMIC)
        body.position    = pos

        shape            = pymunk.Poly.create_box(body, size)
        shape.mass       = ROCKET_MASS
        shape.elasticity = ROCKET_ELASTICITY
        shape.friction   = ROCKET_FRICTION
        shape.color      = ROCKET_COLOR

        # Engine
        body_engine, shape_engine      = self._create_engine((body.position[0], body.position[1] + (ROCKET_SIZE[1] / 2) + ((ENGINE_SIZE[1] / 2) * 1.05)))

        # A PivotJoint connecting the engine and the rocket
        pivotjoint = pymunk.PivotJoint(body_engine, body, (0, (-ENGINE_SIZE[1] / 2) * 1.05), (0, ROCKET_SIZE[1] / 2))
        pivotjoint.max_force = 500000 * SCALE * 2

        self.space.add(body, shape, body_engine, shape_engine, pivotjoint)

        return shape, shape_engine
    
    def _create_engine(self, pos):
        size             = ENGINE_SIZE

        inertia          = pymunk.moment_for_box(mass = ENGINE_MASS, size = size)

        body             = pymunk.Body(mass = ENGINE_MASS, moment = inertia, body_type = pymunk.Body.DYNAMIC)
        body.position    = (pos[0], pos[1])

        shape            = pymunk.Poly.create_box(body, size)
        shape.mass       = ENGINE_MASS
        shape.elasticity = ENGINE_ELASTICITY
        shape.friction   = ENGINE_FRICTION
        shape.color      = ENGINE_COLOR

        return body, shape
    
    def _create_legs(self, rocket: pymunk.Body):

        leg_rects = [
            [LEG_SIZE, (rocket.position[0] - ROCKET_SIZE[0]//2 - LEG_SIZE[0]//2, rocket.position[1] + LEG_HEIGHT + LEG_SIZE[1]/2), -1],
            [LEG_SIZE, (rocket.position[0] + ROCKET_SIZE[0]//2 + LEG_SIZE[0]//2, rocket.position[1] + LEG_HEIGHT + LEG_SIZE[1]/2), 1]
        ]

        for size, pos, leg_side in leg_rects:
            inertia          = pymunk.moment_for_box(mass = LEG_MASS, size = size)

            body             = pymunk.Body(mass = LEG_MASS, moment = inertia, body_type = pymunk.Body.DYNAMIC)
            body.position    = pos

            shape            = pymunk.Poly.create_box(body, size)
            shape.mass       = LEG_MASS
            shape.elasticity = LEG_ELASTICITY
            shape.friction   = LEG_FRICTION
            shape.color      = LEG_COLOR

            # A PivotJoint connecting the leg and the rocket
            #pinjoint = pymunk.PinJoint(body, rocket, (-leg_side * LEG_SIZE[0]/2, -LEG_SIZE[1] / 2), (leg_side * ROCKET_SIZE[0]/2, LEG_HEIGHT))
            pivotjoint = pymunk.PivotJoint(body, rocket, (-leg_side * LEG_SIZE[0]/2, -LEG_SIZE[1] / 2), (leg_side * ROCKET_SIZE[0]/2, LEG_HEIGHT))
            pivotjoint.max_force = 80000 * SCALE * 2

            # A SlideJoint connecting the leg and the rocket
            # slidejoint = pymunk.SlideJoint(body, rocket, (leg_side * LEG_SIZE[0]/2, LEG_SIZE[1] / 2.6), (leg_side * ROCKET_SIZE[0]/2, LEG_SPRING_HEIGHT), 100 * SCALE, 115 * SCALE)
            # slidejoint.max_force = 50000 * SCALE * 2

            # A SpringJoint connecting the leg and the rocket
            springjoint = pymunk.DampedSpring(body, rocket, (leg_side * LEG_SIZE[0]/2, LEG_SIZE[1] / 2.6), (leg_side * ROCKET_SIZE[0]/2, LEG_SPRING_HEIGHT), -16000 * SCALE, -2.5, 30)

            self.legs.append(shape)

            self.space.add(body, shape, pivotjoint, springjoint)
        
        # A SpringJoint connecting both legs
        springjoint = pymunk.DampedSpring(self.legs[0].body, self.legs[1].body, (LEG_SIZE[0]/2, LEG_SIZE[1] / 3), (-LEG_SIZE[0]/2, LEG_SIZE[1] / 3), -4000 * SCALE, -2.5, 40)
        
        # A SlideJoint connecting both legs
        slidejoint  = pymunk.SlideJoint(self.legs[0].body, self.legs[1].body, (LEG_SIZE[0]/2, LEG_SIZE[1] / 3), (-LEG_SIZE[0]/2, LEG_SIZE[1] / 3), 0 * SCALE, 135 * SCALE)
        slidejoint.max_force = 50000 * SCALE * 2
  
        self.space.add(springjoint, slidejoint)
    
    def _create_water(self):
        water_body          = pymunk.Body(body_type=pymunk.Body.STATIC)
        water_body.position = (VIEWPORT_WIDTH / 2, VIEWPORT_HEIGHT - (WATER_HEIGHT / 2))

        water               = pymunk.Poly.create_box(water_body, (VIEWPORT_WIDTH, WATER_HEIGHT))
        water.friction      = 3
        water.elasticity    = 0.2
        water.color         = WATER_COLOR
        self.space.add(water_body, water)
        return water

    def _create_landing_pad(self):
        size             = LANDING_PAD_SIZE
        pos              = LANDING_PAD_POS

        body             = pymunk.Body(body_type = pymunk.Body.STATIC)
        body.position    = pos
        shape            = pymunk.Poly.create_box(body, size)
        shape.elasticity = LANDING_PAD_ELASTICITY
        shape.friction   = LANDING_PAD_FRICTION
        shape.color      = LANDING_PAD_COLOR
        self.space.add(body, shape)

        return shape
    
    def _create_particle(self, x, y, ttl, radius, particle_type = 0, vel = (0, 0)):
        
        if particle_type == 0:
            p = [(x, y), ttl, radius, (0, 0, 0, 0)]
            self.engine_particles.append(p)

        elif particle_type == 1:
            p = [(x, y), ttl, radius, THRUSTER_PARTICLE_COLOR, vel]
            self.thruster_particles.append(p)

        self.particles.append(p)

    def _clean_particles(self, clean_all: bool):
        if clean_all:
            self.particles          = []
            self.engine_particles   = []
            self.thruster_particles = []
            return

        while self.particles and self.particles[0][1] <= 0:
            if self.particles.pop(0) in self.engine_particles:
                self.engine_particles.pop(0)
            
            else:
                self.thruster_particles.pop(0)
    
    def _check_leg_contacts(self, check_landing_pad):
        contacts = []
        shape = self.landing_pad if check_landing_pad else self.water

        if (len(self.legs) > 0) and ((self.landing_pad is not None and check_landing_pad is True) or (self.water is not None and not(check_landing_pad))):
            for leg in self.legs:
                contacts.append((leg.shapes_collide(shape).normal)[1] >= 1)

        else:
            return [False, False]
        
        return contacts
    
    def _check_lander_collision(self):
        contacts = [False, False] # index 0 is contact with landing pad, and index 1 is contact with water

        if (len(self.legs) > 0):
            for leg in self.legs:
                for shape in [self.landing_pad, self.water]:
                    if shape == self.landing_pad:
                        contacts[0] = True if ((self.lander.shapes_collide(shape).normal)[1] >= 1) else contacts[0]

                    elif shape == self.water:
                        contacts[1] = True if ((self.lander.shapes_collide(shape).normal)[1] >= 1) else contacts[1]
        
        return contacts
    
    def _destroy(self):
        self.space         = None

        self.lander        = None
        self.mainEngine    = None
        self.water         = None
        self.landing_pad   = None
        self.legs          = []
        self.leg_contacts  = []

        self.screen        = None

        self.dt            = 0

        self.throttle      = 0
        self.gimbal        = 0.0
        self.power         = 0
        self.force_dir     = 0

        self.engine_pos    = ()
        self.thruster_pos  = []
        self.starting_pos  = STARTING_POS

        self.engine_particles   = []
        self.thruster_particles = []
        self.particles          = []

        self.stepNumber    = 0
        self.landingTicks  = 0

        self.draw_options  = None

        self._clean_particles(True)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._destroy()

        if seed:
            np.random.seed(seed)
        
        self.space = pymunk.Space()

        self.space.gravity = (X_GRAVITY, Y_GRAVITY)

        if self.render_mode == "human":
            self.screen = pygame.display.get_surface()
            
            if self.screen is None:
                self.screen   = pygame.display.set_mode((VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
        
        if self.clock:
            self.clock = pygame.time.Clock()
        else:
            self.clock = None

        self.dt          = 1 / self.metadata['render_fps']
        self.render_mode = self.render_mode

        self.isopen       = True
        self.done         = False
        self.truncated    = False
        self.prev_shaping = None

        if not DEBUG_FLAG:
            # Randomizing Rocket Starting Pos
            self.starting_pos = ((np.random.randint(-STARTING_POS_DEVIATION, STARTING_POS_DEVIATION, size=1) + STARTING_POS[0]), STARTING_POS[1])
        
        self._setup()

        if self.render_mode == 'human':
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        
        if not DEBUG_FLAG:
            # Apply random angular vel to the rocket
            self.lander.body.apply_impulse_at_local_point((np.random.randint(-STARTING_ANG_VEL_RANGE, STARTING_ANG_VEL_RANGE, 1), 0), (0, -ROCKET_SIZE[1]/2))

        # Checking for leg contact with landing pad
        self.leg_contacts = self._check_leg_contacts(True)

        return self.step(6)[0]
    
    def step(self, action):
        assert action != None, "Action is None"

        self.stepNumber += 1

        self.force_dir = 0

        if action == 0:    # Gimbal left
            self.gimbal += 0.01
        elif action == 1:  # Gimbal right
            self.gimbal -= 0.01
        elif action == 2:  # Increase Throttle
            self.throttle += 0.01
            self.throttle = 1.00
        elif action == 3:  # Decrease Throttle
            self.throttle -= 0.01
            self.throttle = 0.00
        elif action == 4:  # left control thruster
            self.force_dir = -1
        elif action == 5:  # right control thruster
            self.force_dir = 1
        else:              # No action
            ...

        # Declaring important variables
        reward    = 0
        done      = False
        truncated = False
        info      = {}

        self.gimbal = np.clip(self.gimbal, -GIMBAL_THRESHOLD, GIMBAL_THRESHOLD)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.power = 0 if self.throttle == 0.0 else (MIN_THROTTLE + self.throttle * (1 - MIN_THROTTLE)) * (SCALE * 2)

        # main engine force
        # force_pos = (self.lander.body.position[0] + (np.sin(-self.lander.body.angle) * ROCKET_HEIGHT/2), self.lander.body.position[1] + (np.cos(-self.lander.body.angle) * ROCKET_HEIGHT/2))
        # force = (-np.sin(-self.lander.body.angle + self.gimbal) * MAIN_ENGINE_POWER * self.power,
        #          np.cos(-self.lander.body.angle + self.gimbal) * MAIN_ENGINE_POWER * self.power)
        # force_pos = list(ENGINE_HEIGHT * np.array(
        #               (-np.sin(self.lander.body.angle), np.cos(self.lander.body.angle))))
        force_pos = (0, 0)
        force     = (0, -MAIN_ENGINE_POWER * self.power)

        #self.lander.body.apply_force_at_local_point(force = force, point = force_pos)
        self.mainEngine.body.apply_force_at_local_point(force = force, point = force_pos)

        self.debug = (self.lander.body.position[0] + force_pos[0], self.lander.body.position[1] + force_pos[1])

        # Main Engine Gimbal
        self.mainEngine.body.angle = self.lander.body.angle + self.gimbal
        
        # control thruster force
        force_pos_c = list(THRUSTER_HEIGHT * np.array(
                      (-np.sin(self.lander.body.angle), np.cos(self.lander.body.angle))))
        # force_c = (-self.force_dir * np.cos(self.lander.body.angle) * SIDE_ENGINE_POWER,
        #            self.force_dir * np.sin(self.lander.body.angle) * SIDE_ENGINE_POWER)
        force_c = (SIDE_ENGINE_POWER * self.force_dir, 0)
        #self.debug = (self.lander.body.position[0] + force_pos_c[0], self.lander.body.position[1] + force_pos_c[1])
        
        self.lander.body.apply_force_at_local_point(force = force_c, point=force_pos_c)

        self.engine_pos    = (self.mainEngine.body.position[0] + (-np.sin(self.mainEngine.body.angle)), self.mainEngine.body.position[1] + (np.cos(self.mainEngine.body.angle)))
        self.thruster_pos  = list((THRUSTER_HEIGHT * 0.85) * np.array((-np.sin(self.lander.body.angle), np.cos(self.lander.body.angle)))) 
        self.thruster_pos[0] += self.lander.body.position[0]
        self.thruster_pos[1] += self.lander.body.position[1]

        # Checking for leg contact with landing pad
        self.leg_contacts = self._check_leg_contacts(True)

        # Observation
        pos    = self.lander.body.position
        vel    = self.lander.body.velocity
        angVel = self.lander.body.angular_velocity

        state = [
            (pos.x - VIEWPORT_WIDTH / 2) / (VIEWPORT_WIDTH / 2),
            (-pos.y + (LANDING_PAD_POS[1] - LEG_SIZE[1]/2 - ROCKET_SIZE[1]/2)) / (VIEWPORT_HEIGHT),
            vel.x / 1000,
            vel.y / 1000,
            self.lander.body.angle,
            20.0 * angVel / FPS,
            1.0 if self.leg_contacts[0] else 0.0,
            1.0 if self.leg_contacts[1] else 0.0,
        ]

        # Reward
        outside = True if abs(state[0]) > 1.2 else False
        landed = state[6] and state[7] and vel.x < 0.3 and vel.y < 0.2
        crashed = False
        
        shaping = (
            -35 * abs(state[0]) # X pos is really important to nail down
            -60 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            -50 * abs(state[3]) # Y vel is really important to nail down
            -60 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            -70 * abs(state[4])
            -30 * abs(state[5])
            +25 * state[6]
            +25 * state[7]
        )  # Fifteen points for each leg contact
           # If you lose contact after landing, you get negative reward
        
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= self.power * 0.10 # less fuel spent is better, about -30 for heuristic landing
        reward -= abs(self.force_dir) * 0.06
        
        if (state[3] >= CRASHING_SPEED) and (any(self.leg_contacts)):
            crashed = True
        
        # Checking for leg contact with water
        if any(self._check_leg_contacts(False)):
            outside = True

        # Check for rocket body colliding with ground or water
        if any(self._check_lander_collision()):
            crashed = True
        
        if not all(self.leg_contacts):
            reward -= 0.12 / FPS

        if self.stepNumber >= MAX_STEP_NUMBER:
            truncated = True
        
        if landed and not crashed:
            if self.landingTicks >= LANDING_TICKS:
                done   = True
                reward += 250
                #print(f"HOORAY!!!, Landed Sucessfully")
            else:
                if self.landingTicks == 0: # Just landed
                    ...
                self.landingTicks += 1
                reward += 2.6 / FPS
        
        if outside or crashed:
            done   = True
            reward = -80.0000000000

        self.space.step(self.dt)

        self.render(mode = self.render_mode)

        return np.array(state, dtype = np.float32), reward, done, info # observation, reward, done, truncated, info
    
    def render(self, mode, **kwargs):

        if mode is not None:
            self.surf = pygame.Surface((VIEWPORT_WIDTH, VIEWPORT_HEIGHT))
            self.particle_surf = self.surf.copy()

            pygame.draw.rect(self.surf, SKY_COLOR, self.surf.get_rect())
            pygame.draw.rect(self.particle_surf, (255, 255, 255), self.particle_surf.get_rect())

            for obj in self.particles:

                if obj in self.engine_particles: # Main Engine Particles
                    NewTTL = (((obj[1] - 0) * 2) / 1) - 1

                    obj[0] = (obj[0][0], obj[0][1] + (randint(PARTICLE_Y_VEL_RANGE[0], PARTICLE_Y_VEL_RANGE[1]) / FPS) * NewTTL) # Move the smoke upwards
                    obj[1] -= random() * PARTICLE_TTL_SUBTRACT # Take time from its lifetime
                    obj[2] += PARTICLE_GROWTH_RATE if obj[2] < PARTICLE_MAX_RADIUS else 0 # radius grows as the particle gets older
                    ttl = 1 - obj[1]
                    
                    obj[3] = (
                        int(ttl * 255),
                        int(ttl * 255),
                        int(ttl * 255),
                        int(255)
                    )
                
                elif obj in self.thruster_particles: # Control Thruster Particles
                    Vel = (obj[4][0] * THRUSTER_PARTICLE_VEL, obj[4][1])

                    obj[0] = (obj[0][0] + (Vel[0] / FPS), obj[0][1] + (Vel[1]) / FPS) # Move the smoke left or right
                    obj[1] -= random() * (PARTICLE_TTL_SUBTRACT * 1.5) # Take time from its lifetime
                    obj[2] += (PARTICLE_GROWTH_RATE * 0.5) if obj[2] < (PARTICLE_MAX_RADIUS / 3.5) else 0 # radius grows as the particle gets older
                    
                    NewTTL = 1 - obj[1]
                    obj[3] = (
                        int(np.clip(20 + (NewTTL * 255), 0, 255)),
                        int(np.clip(20 + (NewTTL * 255), 0, 255)),
                        int(obj[3][2]),
                        int(obj[3][3])
                    )

                else:
                    Exception(f"The particles for rendering have invalid attributes, particle_type should be a discrete range between{Fore.BLUE} 0 {Fore.RESET}and {Fore.BLUE}1{Fore.RESET}")

            # Drawing the Particles
            for obj in self.particles:
                try:
                    pygame.draw.circle(
                        self.particle_surf,
                        color=obj[3],
                        center = (obj[0][0], obj[0][1]),
                        radius = obj[2], 
                    )

                except ValueError: # particle RGB value is invalid
                    ...

            # Draw Two Flags on either side of the landing pad

            if DRAW_FLAGS: 
                for i, x in enumerate([LANDING_PAD_POS[0] - LANDING_PAD_WIDTH/2, LANDING_PAD_POS[0] + LANDING_PAD_WIDTH/2]):
                    flagy1 = LANDING_PAD_HEIGHT
                    flagy2 = flagy1 * 2
                    pygame.draw.line(
                        self.surf,
                        color=(255, 255, 255),
                        start_pos=(x, flagy1),
                        end_pos=(x, flagy2),
                        width=1,
                    )

                    if i == 0:
                        pygame.draw.polygon(
                            self.surf,
                            color=(204, 204, 0),
                            points=[
                                (x, flagy2),
                                (x, flagy2 - 10),
                                (x - 25, flagy2 - 5),
                            ],
                        )
                        # gfxdraw.aapolygon(
                        #     self.surf,
                        #     [(x, flagy2), (x, flagy2 - 10), (x - 25, flagy2 - 5)],
                        #     (204, 204, 0),
                        # )
                            
                    else:
                        pygame.draw.polygon(
                            self.surf,
                            color=(204, 204, 0),
                            points=[
                                (x, flagy2),
                                (x, flagy2 - 10),
                                (x + 25, flagy2 - 5),
                            ],
                        )
                        # gfxdraw.aapolygon(
                        #     self.surf,
                        #     [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                        #     (204, 204, 0),
                        # )

            # Create Main Engine Smoke Particles
            if len(self.engine_particles) <= MAX_ENGINE_PARTICLES:
                if len(self.engine_particles) > 0:
                    if self.engine_particles[-1][1] < SMOKE_RATE * self.throttle and self.throttle > 0.05:
                        self._create_particle(randrange(-10, 10) + self.engine_pos[0], randrange(-10, 10) + self.engine_pos[1], PARTICLE_STARTING_TTL, 8 * SCALE)
                else:
                    if self.throttle > 0:
                        self._create_particle(randrange(-10, 10) + self.engine_pos[0], randrange(-10, 10) + self.engine_pos[1], PARTICLE_STARTING_TTL, 8 * SCALE)

            # Create Control Thrusters Particles
            if len(self.thruster_particles) <= MAX_THRUSTER_PARTICLES:
                vel = list(THRUSTER_HEIGHT * np.array((self.force_dir * -np.sin(self.lander.body.angle - 1.5), (self.force_dir * np.cos(self.lander.body.angle  - 1.5))))) 
                vel[1] += (self.lander.body.velocity[1] * THRUSTER_PARTICLE_MASS)
                
                if len(self.thruster_particles) > 0:
                    if (self.thruster_particles[-1][1] < SMOKE_RATE * abs(self.force_dir)) and (abs(self.force_dir) > 0):
                        self._create_particle(self.thruster_pos[0], self.thruster_pos[1], PARTICLE_STARTING_TTL, 8 * SCALE, particle_type = 1, vel = vel)
                
                else:
                    if abs(self.force_dir) > 0:
                        self._create_particle(self.thruster_pos[0], self.thruster_pos[1], PARTICLE_STARTING_TTL, 8 * SCALE, particle_type = 1, vel = vel)

            self._clean_particles(False)

            self.surf = pygame.transform.flip(self.surf, False, False)

            # DEBUG

            #surf = pygame.Surface(ROCKET_SIZE)
            #surf_trans = pygame.transform.rotate(surf, math.degrees(self.lander.body.angle))
            #rect = surf_trans.get_rect(center = self.lander.body.position)

            font_surf = FONT.render(str(self.stepNumber), False, "Black")
            font_rect = font_surf.get_rect(topleft = (20, 20))

        if mode == "human":
            assert self.screen is not None, "Screen is NONE"

            self.screen.blit(self.surf, (0, 0))

            self.screen.blit(self.particle_surf, (0, 0), special_flags= pygame.BLEND_RGBA_MULT)

            self.space.debug_draw(self.draw_options)

            self.screen.blit(font_surf, font_rect)

            pygame.event.pump()

            if self.clock is not None:
                self.clock.tick(self.fps)

            pygame.display.flip()

        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

def run():

    import sys

    #create_boundaries(space, width, height)
    env = Rocket(render_mode = 'human', gravity = (X_GRAVITY, Y_GRAVITY))
    action = 0

    mouse_joint = None
    mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

    done      = False

    while not done:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                sys.exit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if mouse_joint is not None:
                    env.space.remove(mouse_joint)
                    mouse_joint = None

                p = pymunk.Vec2d(*event.pos)
                hit = env.space.point_query_nearest(p, 5, pymunk.ShapeFilter())
                if hit is not None and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
                    shape = hit.shape
                    # Use the closest point on the surface if the click is outside
                    # of the shape.
                    if hit.distance > 0:
                        nearest = hit.point
                    else:
                        nearest = p
                    mouse_joint = pymunk.PivotJoint(
                        mouse_body, shape.body, (0, 0), shape.body.world_to_local(nearest)
                    )
                    mouse_joint.max_force = 80000
                    mouse_joint.error_bias = (1 - 0.15) ** 60
                    env.space.add(mouse_joint)

            elif event.type == pygame.MOUSEBUTTONUP:
                if mouse_joint is not None:
                    env.space.remove(mouse_joint)
                    mouse_joint = None

        keys = pygame.key.get_pressed()

        # Input
        if keys[pygame.K_LEFT]: # Increase Gimbal
            action =  0
        
        elif keys[pygame.K_RIGHT]: # Decrease Gimbal
            action =  1

        elif keys[pygame.K_w]: # Increase Throttle
            action =  2
        
        elif keys[pygame.K_s]: # Decrease Throttle
            action =  3
        
        elif keys[pygame.K_a]:
            action =  4
            #env.reset(seed = 19)
        
        elif keys[pygame.K_d]:
            action =  5
            #env.space.gravity = (0, 0)

        elif keys[pygame.K_LCTRL] and keys[pygame.K_s]: # Screenshot
            pygame.image.save(env.screen, "rocket_landing_sim_screenshot.png")
        
        elif keys[pygame.K_r]:
            env.reset()

        else:
            action = 6

        # Step
        observation, reward, done, info = env.step(action)

        try:
            print(f"Observation: {Fore.BLUE}{observation}{Fore.RESET}, Reward: {Fore.GREEN if reward > 0 else Fore.RED}{reward}{Fore.RESET}")
        except NameError:
            print(f"Observation: {observation}, Reward: {reward}")
        
        if (done): print("FINISHED SIMULATION")

        # Mouse Interaction
        mouse_pos = pygame.mouse.get_pos()
        mouse_body.position = mouse_pos

        CLOCK.tick(FPS)

        pygame.display.set_caption(f"fps: {CLOCK.get_fps()}")
    env.close()

if __name__ == '__main__':
    run()
