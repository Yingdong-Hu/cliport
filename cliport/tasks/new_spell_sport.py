"""Stack Blocks Task."""

import collections
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import random
import pybullet as p
import re
import os


class SpellSport(Task):

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "pick up the {pick} and place it on the {place}"
        self.task_completed_desc = "done spelling out a sport using the present letters."

    def reset(self, env):
        super().reset(env)

        sport_list = [
            'run',
            'golf',
            'swim',
            'dive',
            'hike',
            'surf',
            'shot',
            'jump',
            'ride',
            'trap'
        ]

        sport = random.choice(sport_list)
        sport_letter_names = [l.upper() for l in sport]
        n_letters = len(sport_letter_names)

        # Cannot place initial object on the bottom side of the table
        bottom_left_corner_pos = utils.CORNER_OR_SIDE['bottom left corner']
        bottom_right_corner_pos = utils.CORNER_OR_SIDE['bottom right corner']
        bottom_left_corner_pos = (bottom_left_corner_pos[0], bottom_left_corner_pos[1], 0.0)
        bottom_right_corner_pos = (bottom_right_corner_pos[0], bottom_right_corner_pos[1], 0.0)
        obstacle_y = np.linspace(bottom_left_corner_pos[1], bottom_right_corner_pos[1], 10)
        obstacle_pos = [(bottom_left_corner_pos[0], y, 0.0) for y in obstacle_y]

        template = 'letter-objects/object-template.urdf'
        letter_size = (0.08, 0.08, 0.02)
        scale = [0.003, 0.003, 0.001]
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))

        # Add sport letters.
        sport_letters = []
        self.letter2id = {}
        for i in range(n_letters):
            pose = self.get_random_pose(env, letter_size, obstacle_pos)
            pose = (pose[0], rot)
            fname = f'{sport_letter_names[i]}.obj'
            fname = os.path.join(self.assets_root, 'letter-objects', fname)
            replace = {'FNAME': (fname,), 'SCALE': scale, 'COLOR': utils.COLORS['red']}
            urdf = self.fill_template(template, replace)
            letter_id = env.add_object(urdf, pose)
            self.letter2id[sport_letter_names[i]] = letter_id
            if os.path.exists(urdf):
                os.remove(urdf)
            sport_letters.append((letter_id, (0, None)))
        self.sport_letters = sport_letters

        # Goal: correctly spell out a sport using the present letters
        bottom_left_corner_pose = (bottom_left_corner_pos, (0, 0, 0, 1))

        place_y = [i * 0.1 for i in range(n_letters)]
        relative_pos = [(0, y, 0.01) for y in place_y]
        targets = [(utils.apply(bottom_left_corner_pose, relative_pos[i]), rot) for i in range(n_letters)]
        self.targets = targets

        for i in range(n_letters):
            self.goals.append(([sport_letters[i]], np.ones((1, 1)), [targets[i]],
                               False, True, 'pose', None, 1 / n_letters))
            if i == 0:
                self.lang_goals.append(self.lang_template.format(pick=sport_letter_names[i],
                                                                 place='bottom left corner'))
            else:
                self.lang_goals.append(self.lang_template.format(pick=sport_letter_names[i],
                                                                 place='right of ' + sport_letter_names[i - 1]))

        self.max_steps = n_letters + 1
        self.high_level_lang_goal = 'correctly spell out a sport using the present letters'

    def success(self):
        letter_poses = [p.getBasePositionAndOrientation(lid) for lid, _ in self.sport_letters]
        matches = [self.is_match(letter_poses[i], self.targets[i], symmetry=0) for i in range(len(self.sport_letters))]
        return all(matches)

    def step_oracle(self, env):
        """Oracle agent."""
        OracleAgent = collections.namedtuple('OracleAgent', ['act'])

        def act(obs, language_goal):
            """Calculate action."""

            # Oracle uses perfect RGB-D orthographic images and segmentation masks.
            cmap, hmap, obj_mask = self.get_true_image(env)

            pattern = r"pick up the (.+?) and place it on the (.+?)$"
            match = re.match(pattern, language_goal)
            assert match is not None, f"Language goal '{language_goal}' does not match pattern '{pattern}'"
            pick_letter = match.group(1)
            alphabet = utils.ALPHABET
            assert pick_letter in alphabet, f"Letter '{pick_letter}' is not in the alphabet"
            place_loc = match.group(2)

            pick_letter_id = self.letter2id[pick_letter]
            # pick pose
            pick_mask = np.uint8(obj_mask == pick_letter_id)
            if np.sum(pick_mask) == 0:
                raise ValueError(f"Letter '{pick_letter}' is not found in segmentation mask")
            pick_pix = utils.sample_gaussian_distribution(pick_mask)
            pick_pos = utils.pix_to_xyz(pick_pix, hmap, self.bounds, self.pix_size)
            pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

            # place pose
            rot = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))
            if 'bottom left corner' in place_loc:
                bottom_left_corner_pos = utils.CORNER_OR_SIDE['bottom left corner']
                bottom_left_corner_pos = (bottom_left_corner_pos[0], bottom_left_corner_pos[1], 0.01)
                targ_pose = (bottom_left_corner_pos, rot)
            elif 'bottom side' in place_loc:
                bottom_side_pos = utils.CORNER_OR_SIDE['bottom side']
                bottom_side_pos = (bottom_side_pos[0], bottom_side_pos[1], 0.01)
                targ_pose = (bottom_side_pos, rot)
            elif 'right of' in place_loc:
                right_of_letter = place_loc.split(' ')[-1]
                right_of_letter_pose = p.getBasePositionAndOrientation(self.letter2id[right_of_letter])
                targ_position = (right_of_letter_pose[0][0], right_of_letter_pose[0][1] + 0.1, 0.01)
                targ_pose = (targ_position, rot)
            else:
                raise ValueError(f"Unknown place location '{place_loc}'")

            obj_pose = p.getBasePositionAndOrientation(pick_letter_id)
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