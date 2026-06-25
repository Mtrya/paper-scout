# Extracted from code/robotwin-repo/script/eval_policy.py
# Standardized benchmark evaluation: 100 rollouts per task, Easy vs Hard,
# language sampled from seen/unseen pools.

def eval_policy(task_name, TASK_ENV, args, model, st_seed, test_num=100, ...):
    now_seed = st_seed
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    args["eval_mode"] = True

    while succ_seed < test_num:
        # 1. First run the *expert* program to obtain a feasible scene seed.
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info = TASK_ENV.play_once()
        TASK_ENV.close_env()

        if TASK_ENV.plan_success and TASK_ENV.check_success():
            succ_seed += 1
        else:
            now_seed += 1
            continue

        # 2. Reset to the same seed and sample a language instruction for the policy.
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        results = generate_episode_descriptions(args["task_name"], [episode_info["info"]], test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        # 3. Policy roll-out with step limit.
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                TASK_ENV.suc += 1
                break

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        TASK_ENV.test_num += 1
        print(f"Success rate: {TASK_ENV.suc}/{TASK_ENV.test_num} => "
              f"{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%")
        now_seed += 1
