import argparse
import random
import numpy as np
import torch
from typing import List, Dict, Any
import yaml
import random
from pogema import AnimationConfig, AnimationMonitor, pogema_v0, GridConfig
from mg_map import get_map_info, visualize_map_info, visualize_all_agents_paths
import gymnasium as gym
from mg_env import  run_mgmapf_python

def convert_xy_to_rowcol(pos_xy: tuple) -> tuple:
    x, y = pos_xy
    return (y, x)  

def load_map_from_yaml(yaml_file, map_name):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    map_str = data.get(map_name)
    if map_str is None:
        raise KeyError(f"Map '{map_name}' not found in YAML file.")
    return map_str

def generate_task_instance_list(
    num_instances: int,  
    fixed_goal_num: int,  
    valid_starts: list[tuple],  
    obstacles: list[tuple],  
    boundaries: list[tuple]  
) -> tuple[list[list[tuple]], list[tuple]]:
    task_instances = []
    start_positions = []
    valid_positions = valid_starts.copy()

    if not isinstance(valid_starts, list) or len(valid_starts) == 0:
        print("Error: valid_starts is none")
        return [], []

    # 均匀分布选择起始位置
    valid_starts_len = len(valid_starts)
    step = max(1, valid_starts_len // num_instances)

    candidate_start_positions = []
    for i in range(num_instances):
        index = min(i * step, valid_starts_len - 1)
        candidate_start_positions.append(valid_starts[index])

    # 验证起始位置有效性
    for start_pos in candidate_start_positions:
        if start_pos not in obstacles and start_pos not in boundaries:
            start_positions.append(start_pos)
        else:
            for backup_pos in valid_starts:
                if backup_pos not in obstacles and backup_pos not in boundaries and backup_pos not in start_positions:
                    start_positions.append(backup_pos)
                    break

    # 确保起始位置数量正确
    while len(start_positions) < num_instances and valid_starts:
        for pos in valid_starts:
            if pos not in obstacles and pos not in boundaries and pos not in start_positions:
                start_positions.append(pos)
                if len(start_positions) == num_instances:
                    break

    start_positions = start_positions[:num_instances]

    # 生成目标候选
    valid_goal_candidates = [
        pos for pos in valid_positions
        if pos not in obstacles and pos not in boundaries
    ]

    # 为每个智能体生成任务（转换为 row, col 格式）
    start_positions_rc = []
    task_instances_rc = []
    
    for idx, agent_start_xy in enumerate(start_positions):
        # 转换起始位置
        start_rc = convert_xy_to_rowcol(agent_start_xy)
        start_positions_rc.append(start_rc)
        
        # 生成目标点
        agent_task_goals = []
        if valid_goal_candidates:
            available_goals = [goal for goal in valid_goal_candidates if goal != agent_start_xy]
            goal_count = min(fixed_goal_num, len(available_goals))
            if goal_count > 0:
                selected_goals_xy = random.sample(available_goals, goal_count)
                # 转换为 row, col
                agent_task_goals = [convert_xy_to_rowcol(g) for g in selected_goals_xy]
        
        task_instances_rc.append(agent_task_goals)

    print(f"\nGenerated {len(task_instances_rc)}  task instances")
    return task_instances_rc, start_positions_rc


def main(args):
    valid_positions, valid_starts, obstacles, boundaries = get_map_info(
        './env/test-maps.yaml', args.map_name
    )
    
    task_instances, start_positions = generate_task_instance_list(
        args.num_agents, args.num_goals, valid_starts, obstacles, boundaries
    )
    
    results = run_mgmapf_python(
        map_name=args.map_name,
        task_instances=task_instances,
        start_positions=start_positions,
        max_steps=args.max_episode_steps,
        obs_radius=5,
        save_animation=args.save_animation, 
        avoid_blocking=True,
        animation_path=f'renders/{args.map_name}_{args.num_agents}_{args.num_goals}.svg',
    )
    
    if results.get('success'):
        print(f"All succeeced!")
    print(f"{'='*70}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Agent Path Planning Test')
    parser.add_argument('--num_agents', type=int, default=10, help='Number of agents')
    parser.add_argument('--num_goals', type=int, default=10, help='Number of goals for each agents')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--map_name', type=str, default='maze-32-32-4', help='Map name')
    parser.add_argument('--max_episode_steps', type=int, default=2000, help='Max steps')
    parser.add_argument('--save_animation', action='store_true', default=True,
                       help='Save SVG animation (default: True)')
    parser.add_argument('--no_animation', dest='save_animation', action='store_false',
                       help='Disable SVG animation')
    parser.add_argument('--show_map_names', action='store_true', help='Show available maps')
    
    
    args = parser.parse_args()
    
    if args.show_map_names:
        with open('./env/test-maps.yaml', 'r', encoding='utf-8') as f:
            maps = yaml.safe_load(f)
        for idx, map_name in enumerate(sorted(maps.keys()), 1):
            print(f"  {idx}. {map_name}")
        exit()

    main(args)
