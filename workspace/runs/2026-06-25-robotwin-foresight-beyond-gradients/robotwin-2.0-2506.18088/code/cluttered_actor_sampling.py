# Extracted from code/robotwin-repo/envs/utils/rand_create_cluttered_actor.py
# and code/robotwin-repo/envs/utils/actor_utils.py
# Shows how distractor objects are sampled, filtered, and placed.

# Object metadata comes from the object library + a same-object map so that
# task-relevant objects are not re-sampled as distractors.
def get_available_cluttered_objects(entity_on_scene: list):
    global cluttered_objects_info, cluttered_objects_list, same_obj
    model_in_use = []
    for entity_name in entity_on_scene:
        if same_obj.get(entity_name) is not None:
            model_in_use += same_obj[entity_name]
        model_in_use.append(entity_name)
    available_models = set(cluttered_objects_list) - set(model_in_use)
    return list(available_models), cluttered_objects_info


def get_cluttered_table(self, cluttered_numbers=10,
                        xlim=[-0.59, 0.59], ylim=[-0.34, 0.34], zlim=[0.741]):
    # Build exclusion list from objects already in the scene.
    task_objects_list = []
    for entity in self.scene.get_all_actors():
        actor_name = entity.get_name()
        if actor_name in ["", "table", "wall", "ground"]:
            continue
        task_objects_list.append(actor_name)
    self.obj_names, self.cluttered_item_info = get_available_cluttered_objects(task_objects_list)

    success_count = 0
    max_try = 50
    while success_count < cluttered_numbers and trys < max_try:
        # Random object category + instance id
        obj_name = self.obj_names[np.random.randint(len(self.obj_names))]
        obj_idx = np.random.randint(len(self.cluttered_item_info[obj_name]["ids"]))
        obj_idx = self.cluttered_item_info[obj_name]["ids"][obj_idx]
        # Physical metadata used for collision-aware placement
        obj_radius = self.cluttered_item_info[obj_name]["params"][obj_idx]["radius"]
        obj_offset = self.cluttered_item_info[obj_name]["params"][obj_idx]["z_offset"]
        obj_maxz = self.cluttered_item_info[obj_name]["params"][obj_idx]["z_max"]

        success, self.cluttered_obj = rand_create_cluttered_actor(
            self.scene,
            xlim=xlim,
            ylim=ylim,
            zlim=np.array(zlim) + self.table_z_bias,
            modelname=obj_name,
            modelid=obj_idx,
            modeltype=self.cluttered_item_info[obj_name]["type"],
            rotate_rand=True,
            rotate_lim=[0, 0, math.pi],
            size_dict=self.size_dict,
            obj_radius=obj_radius,
            z_offset=obj_offset,
            z_max=obj_maxz,
            prohibited_area=self.prohibited_area,
        )
        if success:
            self.cluttered_objs.append(self.cluttered_obj)
            self.size_dict.append(pose + [obj_radius])
            success_count += 1
            self.record_cluttered_objects.append(
                {"object_type": obj_name, "object_index": obj_idx}
            )

# Actor objects expose annotated grasp/contact/functional points as transforms.
class Actor:
    def get_point(self, type, idx, ret):
        actor_matrix = self.actor.get_pose().to_transformation_matrix()
        local_matrix = np.array(self.config[type][idx])
        local_matrix[:3, 3] *= np.array(self.config["scale"])
        world_matrix = actor_matrix @ local_matrix
        # ... return matrix / list / sapien.Pose
