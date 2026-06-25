# Extracted from code/robotwin-repo/envs/_base_task.py and code/robotwin-repo/envs/robot/robot.py
# Shows how grasp candidates are scored differently per arm / embodiment.

class Robot:
    def __init__(self, ...):
        # Each embodiment declares a preferred grasp direction and wrist rotation limits.
        self.left_perfect_direction = left_embodiment_args.get(
            "grasp_perfect_direction", ["front_right", "front_left"]
        )[0]
        self.right_perfect_direction = right_embodiment_args.get(
            "grasp_perfect_direction", ["front_right", "front_left"]
        )[1]
        self.left_rotate_lim = left_embodiment_args.get("rotate_lim", [0, 0])
        self.right_rotate_lim = right_embodiment_args.get("rotate_lim", [0, 0])

    def create_target_pose_list(self, origin_pose, center_pose, arm_tag=None):
        # Build ROTATE_NUM angular perturbations biased toward the arm's reachable space.
        rotate_lim = self.left_rotate_lim if arm_tag == "left" else self.right_rotate_lim
        rotate_step = (rotate_lim[1] - rotate_lim[0]) / CONFIGS.ROTATE_NUM
        res_lst = []
        for i in range(CONFIGS.ROTATE_NUM):
            now_pose = transforms.rotate_along_axis(
                origin_pose, center_pose, [0, 1, 0],
                rotate_step * i + rotate_lim[0],
                axis_type="target", towards=[0, -1, 0],
            )
            res_lst.append(now_pose)
        return res_lst


class Base_Task:
    def choose_grasp_pose(self, actor, arm_tag, pre_dis=0.1, contact_point_id=None):
        pref_direction = self.robot.get_grasp_perfect_direction(arm_tag)
        for i, _ in (contact_point_id or actor.iter_contact_points()):
            pre_pose = self.get_grasp_pose(actor, arm_tag, contact_point_id=i, pre_dis=pre_dis)
            pose = get_grasp_pose(pre_pose, pre_dis - target_dis)

            # Score 1: top-down preference
            now_dis_top_down = cal_quat_dis(
                pose[-4:],
                GRASP_DIRECTION_DIC[(
                    "top_down_little_left" if arm_tag == "right" else "top_down_little_right"
                )],
            )
            # Score 2: embodiment-preferred side direction
            now_dis_side = cal_quat_dis(pose[-4:], GRASP_DIRECTION_DIC[pref_direction])

            # Hybrid scoring: 0.7 top-down + 0.3 embodiment direction
            now_dis = 0.7 * now_dis_top_down + 0.3 * now_dis_side
            # ... keep best pose

        # Fall back to top-down if clearly top-down, else side, else hybrid best.
        if dis_top_down < 0.15:
            return res_pre_top_down_pose, res_top_down_pose
        if dis_side < 0.15:
            return res_pre_side_pose, res_side_pose
        return res_pre_pose, res_pose
