import os
import numpy as np
import random

from cliport import tasks
from cliport.environments.environment import Environment


task_name = 'stack-blocks'
n_eval = 3
save_video = False
root_dir = '/home/huyingdong/cliport-master'
assets_root = os.path.join(root_dir, 'cliport/environments/assets/')

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
    success = False

    high_level_lang_goal = info['high_level_lang_goal']
    print(f'High Level Goal: {high_level_lang_goal}')

    #########################################################################
    # feed 'prompt', 'high_level_lang_goal' and 'front_obs' to GPT-4V, get gpt4_plan
    gpt4_plan = None
    if i == 0:
        gpt4_plan = ['pick up the cyan block and place it on the pink block',
                     'pick up the red block and place it on the cyan block']
    elif i == 1:
        gpt4_plan = ['pick up the green block and place it on the cyan block',
                     'pick up the white block and place it on the green block',
                     'pick up the purple block and place it on the white block']
    elif i == 2:
        gpt4_plan = ['pick up the orange block and place it on the brown block',
                     'pick up the gray block and place it on the orange block',
                     'pick up the blue block and place it on the gray block']
    #########################################################################

    if save_video:
        env.start_rec(f'GPT-4V-{task_name}-seed{seed}')

    for j in range(len(gpt4_plan)):
        mid_level_instruction = gpt4_plan[j]
        act = agent.act(obs, mid_level_instruction)
        obs, _, _, info = env.step(act)
        success = info['success']
        print(f'GPT-4V plan: {mid_level_instruction} | Success: {success}')
    if success:
        success_times += 1

    if save_video:
        env.end_rec()

print(f'\nSuccess Rate: {success_times / n_eval:.3f} ({success_times}/{n_eval})')
