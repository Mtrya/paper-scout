# Extracted from code/robotwin-repo/envs/_base_task.py
# Shows how the five domain-randomization axes are wired into the simulator.

class Base_Task(gym.Env):
    def _init_task_env_(self, table_xy_bias=[0, 0], table_height_bias=0, **kwags):
        # ...
        random_setting = kwags.get("domain_randomization")
        self.random_background = random_setting.get("random_background", False)
        self.cluttered_table = random_setting.get("cluttered_table", False)
        self.clean_background_rate = random_setting.get("clean_background_rate", 1)
        self.random_head_camera_dis = random_setting.get("random_head_camera_dis", 0)
        self.random_table_height = random_setting.get("random_table_height", 0)
        self.random_light = random_setting.get("random_light", False)
        self.crazy_random_light_rate = random_setting.get("crazy_random_light_rate", 0)
        self.crazy_random_light = (
            0 if not self.random_light else np.random.rand() < self.crazy_random_light_rate
        )
        # ...
        # Table height is biased down by up to random_table_height meters.
        self.table_z_bias = (
            np.random.uniform(low=-self.random_table_height, high=0) + table_height_bias
        )
        # ...
        self.create_table_and_wall(table_xy_bias=table_xy_bias, table_height=0.74)
        # ...
        if self.cluttered_table:
            self.get_cluttered_table()

    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        self.table_xy_bias = table_xy_bias
        wall_texture, table_texture = None, None
        table_height += self.table_z_bias

        if self.random_background:
            # seen textures at train time, unseen at eval time
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory_path = f"./assets/background_texture/{texture_type}"
            file_count = len([...])  # number of generated Stable-Diffusion textures
            wall_texture = f"{texture_type}/{np.random.randint(0, file_count)}"
            table_texture = f"{texture_type}/{np.random.randint(0, file_count)}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
        # ... create wall/table with sampled texture_id

    def setup_scene(self, **kwargs):
        # ...
        # Light colors are randomized per channel when random_light is on.
        for direction_light in direction_lights:
            if self.random_light:
                direction_light[1] = [np.random.rand(), np.random.rand(), np.random.rand()]
            self.direction_light_lst.append(
                self.scene.add_directional_light(...)
            )
        for point_light in point_lights:
            if self.random_light:
                point_light[1] = [np.random.rand(), np.random.rand(), np.random.rand()]
            self.point_light_lst.append(
                self.scene.add_point_light(...)
            )

    def _update_render(self):
        # "Crazy" light mode: every render step re-samples light colors.
        if self.crazy_random_light:
            for renderColor in self.point_light_lst:
                renderColor.set_color([np.random.rand(), np.random.rand(), np.random.rand()])
            for renderColor in self.direction_light_lst:
                renderColor.set_color([np.random.rand(), np.random.rand(), np.random.rand()])
            now_ambient_light = self.scene.ambient_light
            now_ambient_light = np.clip(
                np.array(now_ambient_light) + np.random.rand(3) * 0.2 - 0.1, 0, 1
            )
            self.scene.set_ambient_light(now_ambient_light)
        # ...
