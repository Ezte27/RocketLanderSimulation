from gym.envs.registration import register

register(
    id='RocketLander-v0',
    entry_point='rocket_lander.envs:Rocket',
    max_episode_steps=1000,
    reward_threshold=250,
)