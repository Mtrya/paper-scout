# Extracted from code/robotwin-repo/code_gen/task_generation_mm.py
# The "simulation-in-the-loop" code-generation pipeline.

def main(task_info_dic):
    task_info = now_task_info = task_info_dic
    messages = [{
        "role": "system",
        "content": "You need to generate relevant code for some robot tasks ..."
    }]
    generate_num = 5           # max code attempts
    success_threshold = 0.5    # stop if >= 50% success over 10 sim trials
    las_error_message = None
    observation_feedback = None

    best_code = None
    best_success_rate = 0

    for id in range(generate_num):
        # 1. Generate / repair code using DeepSeek-V3 by default.
        res_code, success_rate, las_error_message, error_count, run_records = generate_code(
            now_task_info,
            las_error_message,
            observation_feedback,
            messages,
            generate_num_id=id,
        )

        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_code = res_code

        if success_rate >= success_threshold:
            break

        # 2. Pick the highest-priority failed trial for VLM observation.
        error_list = [
            "The code can not run",
            "The target position of the object is incorrect.",
            "The left arm failed to grasp the object",
            "The right arm failed to grasp the object",
            "Plan execution failed",
            "Unknown error occurred during execution",
        ]
        observe_index = 0
        highest_priority = len(error_list)
        for i, record in enumerate(run_records):
            if record == "success!":
                continue
            for p, error_pattern in enumerate(error_list):
                if error_pattern in record:
                    if p < highest_priority:
                        highest_priority = p
                        observe_index = i
                    break

        # 3. VLM observer (Moonshot vision model) diagnoses the failure from
        #    saved head-camera images and the problematic code.
        generate_specific_dir = os.path.join(
            camera_dir, task_name.lower(), f"generate_num_{id}"
        )
        observation_feedback = observe_task_execution(
            episode_id=observe_index,
            task_name=f"{task_name}",
            task_info={
                "description": task_info["task_description"],
                "goal": "Successfully execute the robot task"
            },
            problematic_code=res_code,
            save_dir=os.path.dirname(generate_specific_dir),
            generate_dir_name=f"generate_num_{id}",
        )

        # 4. Append execution error + VLM feedback to the next LLM prompt.
        now_task_info["task_description"] = (
            f"{task_description}\nFailed to generate code, error message: "
            f"{las_error_message}, error count: {str(error_count)}\n" + change_info
        )
        now_task_info["current_code"] = res_code
