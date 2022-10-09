from setuptools import setup

setup(name='rocket_lander',
      version='0.1',
      url='https://github.com/Ezte27/RocketLander',
      author='Esteban Calderon',
      author_email='estedcg27@gmail.com',
      description='Open AI Gym Environment for a Rocket Lander Simulation in Python.',
      license='MIT',
      packages=['rocket_lander','rocket_lander.envs'],
      install_requires=['gym','numpy', 'pygame']
      )