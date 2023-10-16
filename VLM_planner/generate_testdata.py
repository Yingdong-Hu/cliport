import os
import numpy as np
import random
import imageio

from cliport import tasks
from cliport.environments.environment import Environment


task_name = 'sort-primary-color-blocks'
n_eval = 10
save_video = False
root_dir = '/home/huyingdong/cliport-master'
assets_root = os.path.join(root_dir, 'cliport/environments/assets/')

save_dir = '/home/huyingdong/cliport-master/VLM_planner/testdata'
save_dir = os.path.join(save_dir, task_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

record_cfg = {
    'save_video': save_video,
    'save_video_path': '/home/huyingdong/cliport-master/images',
    'add_text': True,
    'fps': 20,
    'video_height': 640,
    'video_width': 720
}

env = Environment(
    assets_root,
    disp=False,
    shared_memory=False,
    hz=480,
    record_cfg=record_cfg
)

task = tasks.names[task_name]()
task.mode = 'test'
seed = 9999

# Initialize scripted oracle agent
agent = task.step_oracle(env)

success_times = 0

for i in range(n_eval):
    print(f'\nEvaluation Instance: {i + 1}/{n_eval}')

    # Set seeds.
    seed += 2
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    env.set_task(task)
    obs = env.reset()
    front_obs = obs['color'][0]   # front camera, 480 x 640 x 3
    info = env.info

    high_level_lang_goal = info['high_level_lang_goal']
    # capitalize the first letter
    high_level_lang_goal = high_level_lang_goal[0].upper() + high_level_lang_goal[1:]
    high_level_lang_goal = 'Task: ' + high_level_lang_goal
    print(high_level_lang_goal)

    episode_dir = os.path.join(save_dir, f'episode_{i}')
    os.makedirs(episode_dir)
    # save front_obs to episode_dir
    imageio.imwrite(os.path.join(episode_dir, 'front_obs.png'), front_obs)
    # save high_level_lang_goal to episode_dir
    with open(os.path.join(episode_dir, 'instruction.txt'), 'w') as f:
        f.write(high_level_lang_goal)



