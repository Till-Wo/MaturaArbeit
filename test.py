import gym, time
import pybulletgym


env = gym.make("AtlasPyBulletEnv-v0")
env.render()
env.reset()
for i in range(5000):
    env.step(env.action_space.sample())
time.sleep(2)
env.close()


"""
MountainCarContinuous-v0
BipedalWalker-v3
CartPole-v1

RoboSchool Envs			
InvertedPendulumPyBulletEnv-v0
InvertedDoublePendulumPyBulletEnv-v0	
InvertedPendulumSwingupPyBulletEnv-v0	
ReacherPyBulletEnv-v0	
Walker2DPyBulletEnv-v0
HalfCheetahPyBulletEnv-v0
AntPyBulletEnv-v0
HopperPyBulletEnv-v0
HumanoidPyBulletEnv-v0
HumanoidFlagrunPyBulletEnv-v0	
HumanoidFlagrunHarderPyBulletEnv-v0	
AtlasPyBulletEnv-v0
PusherPyBulletEnv-v0
ThrowerPyBulletEnv-v0
StrikerPyBulletEnv-v0


"""