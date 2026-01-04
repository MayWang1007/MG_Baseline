def run_episode(env, algo):
    """
    Runs an episode in the environment using the given algorithm.

    Args:
        env: The environment to run the episode in.
        algo: The algorithm used for action selection.

    Returns:
        ResultsHolder: Object containing the results of the episode.
    """
    algo.reset_states()
    results_holder = ResultsHolder()
    obs, _ = env.reset(seed=env.grid_config.seed)
    while True:
        obs, rew, dones, tr, infos = env.step(algo.act(obs))
        results_holder.after_step(infos)

        if all(dones) or all(tr):
            break
    return results_holder.get_final()


class ResultsHolder:
    """
    Holds and manages the results obtained during an episode.

    """

    def __init__(self):
        """
        Initializes an instance of ResultsHolder.
        """
        self.results = dict()

    def after_step(self, infos):
        """
        Updates the results with the metrics from the given information.

        Args:
            infos (List[dict]): List of dictionaries containing information about the episode.

        """
        if 'metrics' in infos[0]:
            self.results.update(**infos[0]['metrics'])

    def get_final(self):
        """
        Returns the final results obtained during the episode.

        Returns:
            dict: The final results.

        """
        return self.results

    def __repr__(self):
        """
        Returns a string representation of the ResultsHolder.

        Returns:
            str: The string representation of the ResultsHolder.

        """
        return str(self.get_final())

def run_mgpf_episode(env, algo, task_instances=None):
    """
    Runs a multi-goal episode in the environment.
    Manually manages goal sequences for each agent.
    """
    algo.reset_states()
    results_holder = ResultsHolder()
    obs, _ = env.reset(seed=env.grid_config.seed)
    
    num_agents = len(task_instances)
    current_goal_idx = [0] * num_agents  # 当前目标索引
    goals_completed = [0] * num_agents   # 已完成目标数
    total_goals = [len(t) for t in task_instances]
    
    pogema_env = env
    while hasattr(pogema_env, 'env'):
        pogema_env = pogema_env.env
    step_count = 0
    max_steps = env.grid_config.max_episode_steps * max(total_goals) * 2
    
    # 追踪上一步的状态，避免重复检测
    prev_positions = [tuple(obs[i]['xy']) for i in range(num_agents)]
    
    while step_count < max_steps:
        step_count += 1
        
        # 执行一步
        obs, rew, dones, tr, infos = env.step(algo.act(obs))
        results_holder.after_step(infos)
        
        # 检查每个智能体
        for agent_id in range(num_agents):
            # 如果已完成所有目标，跳过
            if goals_completed[agent_id] >= total_goals[agent_id]:
                continue
            
            current_pos = tuple(obs[agent_id]['xy'])
            target_pos = tuple(obs[agent_id]['target_xy'])
            
            # 检查是否到达当前目标
            if current_pos == target_pos and current_pos != prev_positions[agent_id]:
                goals_completed[agent_id] += 1
                current_goal_idx[agent_id] += 1
                
                print(f"[Step {step_count}] ✓ Agent {agent_id} 完成目标 "
                      f"{goals_completed[agent_id]}/{total_goals[agent_id]}")
                
                # 检查是否还有下一个目标
                if current_goal_idx[agent_id] < len(task_instances[agent_id]):
                    next_goal = task_instances[agent_id][current_goal_idx[agent_id]]
                    
                    # 更新目标 - 直接修改agent对象
                    try:
                        pogema_env.grid.positions_xy[agent_id] = list(next_goal)
                        print(f"     → 新目标: {next_goal}")
                    except Exception as e:
                        print(f"     ⚠ 目标更新失败: {e}")
                        try:
                            if hasattr(pogema_env, 'agents'):
                                pogema_env.agents[agent_id].finishes = [list(next_goal)]
                                print(f"     → 新目标(方式2): {next_goal}")
                        except Exception as e2:
                            print(f"     ⚠ 目标更新失败(方式2): {e2}")
            
            prev_positions[agent_id] = current_pos
        
        # 检查是否所有智能体都完成
        if all(goals_completed[i] >= total_goals[i] for i in range(num_agents)):
            print(f"\n{'='*60}")
            print(f"✓ 成功！所有智能体完成所有任务")
            print(f"总步数: {step_count}")
            print(f"{'='*60}")
            break
        
        # 定期进度报告
        if step_count % 100 == 0:
            print(f"\n[进度 - Step {step_count}]")
            for aid in range(num_agents):
                print(f"  Agent {aid}: {goals_completed[aid]}/{total_goals[aid]} 目标")
    
    # 最终统计
    print(f"\n{'='*80}")
    print(f"任务总结")
    print(f"{'='*80}")
    for aid in range(num_agents):
        status = "✓" if goals_completed[aid] >= total_goals[aid] else "✗"
        print(f"  {status} Agent {aid}: {goals_completed[aid]}/{total_goals[aid]} "
              f"({goals_completed[aid]/total_goals[aid]*100:.1f}%)")
    print(f"{'='*80}\n")
    
    return results_holder.get_final()