# Extracted from code/robotwin-repo/description/utils/generate_task_description.py,
# code/robotwin-repo/description/utils/generate_object_description.py,
# code/robotwin-repo/description/utils/generate_episode_instructions.py
#
# Object descriptions: a VLM (image + object name) produces 15 descriptions,
# split into seen / unseen pools.
class ObjDescFormat(BaseModel):
    raw_description: str
    wholePart: subPart
    subParts: List[subPart]
    description: List[str]


def generate_obj_description(object_name, glb_file_name):
    imgstr = get_image_from_glb(object_file_path)
    result = make_prompt_generate(imgstr, object_name)   # calls vision LLM
    save_json(save_dir, glb_file_name, result)


# Task templates: an LLM abstracts the full task description into a JSON schema
# and a set of template strings with placeholders such as {A}, {B}, {a}.
class InstructionFormat(BaseModel):
    stepsOfTask: List[str]
    instructions: List[Instruction]


def generate_task_description(task_name, instruction_num):
    result = make_prompt_generate(
        task_info["full_description"],
        task_info["preference"],
        task_info["schema"],
        instruction_num,
    )
    task_info["seen"].extend(result[2:])
    task_info["unseen"].extend(result[0:2])


# Episode instructions: placeholders are replaced by (random) object descriptions
# and arm names. This is the actual language conditioning stored with a trajectory.
def replace_placeholders(instruction: str, episode_params: Dict[str, str]):
    for key, value in episode_params.items():
        # Value may be a path to a JSON description file -> sample a phrase.
        json_path = os.path.join(parent_directory, "../objects_description", value + ".json")
        if os.path.exists(json_path):
            description = random.choice(json_data.get("seen", []))
            value = f"the {description}"
        elif len(key) == 1 and "a" <= key <= "z":
            value = f"the {value} arm"
        instruction = instruction.replace("{" + key + "}", value)
    return instruction
