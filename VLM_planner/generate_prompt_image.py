import os
import numpy as np
import random
import imageio
import numpy as np

from cliport import tasks
from cliport.environments.environment import Environment

# stack-blocks
# put-blocks-on-corner-side
# put-blocks-matching-colors
for task_name in ['put-blocks-matching-colors']:

    n_eval = 1
    save_video = False
    root_dir = '/home/huyingdong/cliport-master'
    assets_root = os.path.join(root_dir, 'cliport/environments/assets/')

    save_dir = '/home/huyingdong/cliport-master/VLM_planner/testdata/prompt'

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

    if task_name == 'put-blocks-on-corner-side':
        seed = 10000
    else:
        seed = 9999

    # Initialize scripted oracle agent
    agent = task.step_oracle(env)

    success_times = 0

    for i in range(n_eval):

        # Set seeds.
        seed += 2
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)

        env.set_task(task)
        obs = env.reset()
        front_obs = obs['color'][0]   # front camera, 480 x 640 x 3
        # top_down_obs, _, _ = task.get_true_image(env)
        # top_down_obs = np.transpose(top_down_obs, (1, 0, 2))
        top_down_obs, _, _ = env.render_camera(task.oracle_cams[0])
        info = env.info

        high_level_lang_goal = info['high_level_lang_goal']
        # capitalize the first letter
        high_level_lang_goal = high_level_lang_goal[0].upper() + high_level_lang_goal[1:]
        high_level_lang_goal = 'Task: ' + high_level_lang_goal
        print(high_level_lang_goal)

        # save front_obs to episode_dir
        imageio.imwrite(os.path.join(save_dir, '{}-front_obs.png'.format(task_name)), front_obs)
        # save top_down_obs to episode_dir
        imageio.imwrite(os.path.join(save_dir, '{}-top_down_obs.png'.format(task_name)), top_down_obs)
