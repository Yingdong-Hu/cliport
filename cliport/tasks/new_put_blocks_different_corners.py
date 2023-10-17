"""Put Blocks in Bowl Task."""

import collections
import re
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import random
import pybullet as p


class PutBlocksDifferentCorners(Task):

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put all the blocks in different corners"
        self.task_completed_desc = "done putting all the blocks in different corners."

    def reset(self, env):
        super().reset(env)
        n_bowls = np.random.randint(1, 3)   # 1, 2
        n_blocks = np.random.randint(2, 5)  # 2, 3, 4

        color_names = self.get_colors()
        bowl_color_names = random.sample(color_names, n_bowls)
        bowl_colors = [utils.COLORS[cn] for cn in bowl_color_names]
        block_color_names = random.sample(color_names, n_blocks)
        block_colors = [utils.COLORS[cn] for cn in block_color_names]

        corner = [
            'top left corner',
            'top right corner',
            'bottom left corner',
            'bottom right corner'
        ]
        all_corner_pos = [utils.CORNER_OR_SIDE[c] for c in corner]
        all_corner_pos = [(c[0], c[1], 0.02) for c in all_corner_pos]

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size, all_corner_pos)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_colors[i] + [1])
            bowl_poses.append(bowl_pose)

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        self.color2block_id = {}
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size, all_corner_pos)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_colors[i] + [1])
            blocks.append((block_id, (0, None)))
            self.color2block_id[block_color_names[i]] = block_id

        all_corner_pose = [(c, (0, 0, 0, 1)) for c in all_corner_pos]
        # Goal: put all the blocks in different corners
        targ_matrix = np.ones((len(blocks), len(all_corner_pos)))
        self.goals.append((blocks, targ_matrix, all_corner_pose, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template)

        self.targ_matrix = targ_matrix
        self.blocks = blocks
        self.all_corner_pose = all_corner_pose

        # Only one mistake allowed.
        self.max_steps = n_blocks + 1

        self.high_level_lang_goal = 'put all the blocks in different corners'

    def get_colors(self):
        all_colors = utils.ALL_COLORS
        return all_colors

    def success(self):
        matches = self.targ_matrix.copy()
        for i in range(len(self.blocks)):
            object_id, (symmetry, _) = self.blocks[i]
            pose = p.getBasePositionAndOrientation(object_id)
            targets_i = np.argwhere(matches[i, :]).reshape(-1)
            for j in targets_i:
                target_pose = self.all_corner_pose[j]
                if self.is_match(pose, target_pose, symmetry):
                    matches[i, :] = 0
                    matches[:, j] = 0
        is_all_zero = np.all(matches == 0)
        return is_all_zero

    def step_oracle(self, env):
        """Oracle agent."""
        OracleAgent = collections.namedtuple('OracleAgent', ['act'])

        def act(obs, language_goal):
            """Calculate action."""

            # Oracle uses perfect RGB-D orthographic images and segmentation masks.
            cmap, hmap, obj_mask = self.get_true_image(env)

            pattern = r"pick up the (.+?) block and place it on the (.+?)$"
            match = re.search(pattern, language_goal)
            assert match is not None, "Oracle could not parse goal"
            pick_color = match.group(1)
            corner_or_side = match.group(2)
            assert corner_or_side in ['top left corner', 'top right corner', 'bottom right corner', 'bottom left corner']
            pick_block_id = self.color2block_id[pick_color]

            # pick pose
            pick_mask = np.uint8(obj_mask == pick_block_id)
            if np.sum(pick_mask) == 0:
                raise ValueError("Pick block not found in segmentation mask")
            pick_pix = utils.sample_gaussian_distribution(pick_mask)
            pick_pos = utils.pix_to_xyz(pick_pix, hmap, self.bounds, self.pix_size)
            pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

            # place pose
            corner_or_side_pos = utils.CORNER_OR_SIDE[corner_or_side]
            corner_or_side_pos = (corner_or_side_pos[0], corner_or_side_pos[1], 0)
            corner_or_side_pix = utils.xyz_to_pix(corner_or_side_pos, self.bounds, self.pix_size)
            height = hmap[corner_or_side_pix[0], corner_or_side_pix[1]]
            targ_height = height + 0.03
            targ_pos = (corner_or_side_pos[0], corner_or_side_pos[1], targ_height)
            targ_pose = (np.asarray(targ_pos), np.asarray((0, 0, 0, 1)))

            obj_pose = p.getBasePositionAndOrientation(pick_block_id)  # pylint: disable=undefined-loop-variable
            if not self.sixdof:
                obj_euler = utils.quatXYZW_to_eulerXYZ(obj_pose[1])
                obj_quat = utils.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
                obj_pose = (obj_pose[0], obj_quat)
            world_to_pick = utils.invert(pick_pose)
            obj_to_pick = utils.multiply(world_to_pick, obj_pose)
            pick_to_obj = utils.invert(obj_to_pick)
            place_pose = utils.multiply(targ_pose, pick_to_obj)

            place_pose = (np.asarray(place_pose[0]), np.asarray(place_pose[1]))

            return {'pose0': pick_pose, 'pose1': place_pose}

        return OracleAgent(act)

