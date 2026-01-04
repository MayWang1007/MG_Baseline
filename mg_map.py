
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from pathlib import Path
import yaml

def get_map_info(yaml_file_path: str, map_name: str):
    #  symbol categories
    obstacle_symbols = {'#'}  # obstacles
    valid_passable_symbols = {'!', '$', '@', '.'}  # valid positions

    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        map_data = yaml.safe_load(f)

    map_str = map_data[map_name]
    map_lines = [line.strip('\n') for line in map_str.split('\n') if line.strip('\n')]

    valid_positions = []
    valid_start_positions = []
    grid_obstacles = []

    for y, line in enumerate(map_lines):  
        for x, char in enumerate(line):    
            current_coords = (x, y)
            if char in obstacle_symbols:
                grid_obstacles.append(current_coords)
            elif char in valid_passable_symbols:
                valid_positions.append(current_coords)
                valid_start_positions.append(current_coords)
            else:
                continue

    boundaries = []
    if map_lines:
        max_y = len(map_lines) - 1
        max_x = len(map_lines[0]) - 1
        for x in range(max_x + 1):
            boundaries.append((x, 0))
        for x in range(max_x + 1):
            boundaries.append((x, max_y))
        for y in range(1, max_y):
            boundaries.append((0, y))
        for y in range(1, max_y):
            boundaries.append((max_x, y))

    return valid_positions, valid_start_positions, grid_obstacles, boundaries


def visualize_map_info(valid_start_positions, obstacle_positions, boundary_positions, 
                       task_instances=None, start_positions=None,
                       save_path=None, show_plot=True):
    all_coords = []
    all_coords.extend(valid_start_positions)
    all_coords.extend(obstacle_positions)
    all_coords.extend(boundary_positions)
    
    if start_positions and isinstance(start_positions, list):
        all_coords.extend([pos for pos in start_positions if isinstance(pos, (tuple, list)) and len(pos) == 2])
    if task_instances and isinstance(task_instances, list):
        for task_goals in task_instances:
            if isinstance(task_goals, list):
                all_coords.extend([pos for pos in task_goals if isinstance(pos, (tuple, list)) and len(pos) == 2])


    xs = [coord[0] for coord in all_coords if isinstance(coord, (tuple, list)) and len(coord) == 2]
    ys = [coord[1] for coord in all_coords if isinstance(coord, (tuple, list)) and len(coord) == 2]
    map_height = max(xs) + 2
    map_width = max(ys) + 2

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')


    if boundary_positions:
        b_x = [coord[0] for coord in boundary_positions]
        b_y = [coord[1] for coord in boundary_positions]
        ax.scatter(b_y, b_x, c='red', s=50, label='Boundary', alpha=0.8, marker='s')

    if obstacle_positions:
        o_x = [coord[0] for coord in obstacle_positions]
        o_y = [coord[1] for coord in obstacle_positions]
        ax.scatter(o_y, o_x, c='black', s=80, label='Obstacle', alpha=1.0, marker='s')

    if valid_start_positions:
        v_x = [coord[0] for coord in valid_start_positions]
        v_y = [coord[1] for coord in valid_start_positions]
        ax.scatter(v_y, v_x, c='forestgreen', s=30, label='Valid Start Position', alpha=0.6, marker='o')


    if start_positions and isinstance(start_positions, list) and task_instances and isinstance(task_instances, list):
        valid_starts = [pos for pos in start_positions if isinstance(pos, (tuple, list)) and len(pos) == 2]
        valid_tasks = []
        for task_goals in task_instances:
            if isinstance(task_goals, list):
                valid_task_goals = [pos for pos in task_goals if isinstance(pos, (tuple, list)) and len(pos) == 2]
                valid_tasks.append(valid_task_goals)
        
        colors = ['dodgerblue', 'darkorange', 'purple', 'limegreen', 'magenta', 
                  'cyan', 'darkred', 'gold', 'brown', 'teal', 'indigo', 'coral']
        
        for agent_idx in range(len(valid_starts)):
            if agent_idx >= len(valid_tasks):
                break
            
            start_x, start_y = valid_starts[agent_idx]
            task_goals = valid_tasks[agent_idx]
            
            color = colors[agent_idx % len(colors)]

            ax.scatter(start_y, start_x, c=color, s=100, 
                       label=f'Agent {agent_idx} Start' if agent_idx == 0 else "", 
                       alpha=1.0, marker='o', zorder=5) 
            for goal_idx, (goal_x, goal_y) in enumerate(task_goals):
                ax.scatter(goal_y, goal_x, c=color, s=80, 
                           label=f'Agent {agent_idx} Goal' if (agent_idx == 0 and goal_idx == 0) else "",
                           alpha=0.9, marker='^', zorder=6) 
                if agent_idx < 15: 
                    ax.text(start_y + 0.2, start_x, f'A{agent_idx}', 
                            fontsize=8, ha='left', va='center', weight='bold', color=color)
                    ax.text(goal_y + 0.2, goal_x, f'A{agent_idx}-G{goal_idx}', 
                            fontsize=7, ha='left', va='center', weight='bold', color=color)
    ax.set_xlim(-1, map_width)
    ax.set_ylim(-1, map_height)
    ax.set_title('Map Information Visualization (Agent Start & Corresponding Goals)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Y Coordinate', fontsize=12)
    ax.set_ylabel('X Coordinate', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=10)
    ax.grid(alpha=0.2, linestyle='--')

    if save_path is not None and isinstance(save_path, str):
        plt.savefig(save_path, dpi=150, bbox_inches='tight')


def convert_xy_to_rowcol(pos_xy: tuple) -> tuple:
    x, y = pos_xy
    return (y, x)  

def convert_rowcol_to_xy(pos_rc: tuple) -> tuple:
    row, col = pos_rc
    return (col, row)  

def visualize_all_agents_paths(
    step_count: int,
    obstacles: list[tuple],
    agent_trajectories: dict,
    task_instances: list[list[tuple]],
    start_positions: list[tuple],
    output_dir: str = 'visualizations'
):
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')

    obstacles_xy = [convert_rowcol_to_xy(obs) for obs in obstacles]

    all_coords = []
    all_coords.extend(obstacles_xy)
    
    start_positions_xy = [convert_rowcol_to_xy(pos) for pos in start_positions]
    all_coords.extend(start_positions_xy)
    
    for task_goals in task_instances:
        task_goals_xy = [convert_rowcol_to_xy(g) for g in task_goals]
        all_coords.extend(task_goals_xy)
    
    for trajectory in agent_trajectories.values():
        trajectory_xy = [convert_rowcol_to_xy(pos) for pos in trajectory]
        all_coords.extend(trajectory_xy)

    if not all_coords:
        print("Warning: No coordinates to visualize")
        return None

    xs = [coord[0] for coord in all_coords]
    ys = [coord[1] for coord in all_coords]
    map_width = max(xs) + 2
    map_height = max(ys) + 2

    if obstacles_xy:
        o_x = [coord[0] for coord in obstacles_xy]
        o_y = [coord[1] for coord in obstacles_xy]
        ax.scatter(o_x, o_y, c='black', s=80, label='Obstacle', alpha=1.0, marker='s', zorder=1)

    colors = ['dodgerblue', 'darkorange', 'purple', 'limegreen', 'magenta', 
              'cyan', 'darkred', 'gold', 'brown', 'teal', 'indigo', 'coral']
    
    for agent_idx in range(len(start_positions)):
        if agent_idx >= len(task_instances):
            break
        
        start_pos_rc = start_positions[agent_idx]
        task_goals_rc = task_instances[agent_idx]
        trajectory_rc = agent_trajectories.get(agent_idx, [])
        
        start_x, start_y = convert_rowcol_to_xy(start_pos_rc)
        task_goals_xy = [convert_rowcol_to_xy(g) for g in task_goals_rc]
        trajectory_xy = [convert_rowcol_to_xy(pos) for pos in trajectory_rc]
        
        color = colors[agent_idx % len(colors)]

        if trajectory_xy and len(trajectory_xy) > 1:
            traj_x = [pos[0] for pos in trajectory_xy]
            traj_y = [pos[1] for pos in trajectory_xy]
            ax.plot(traj_x, traj_y, '-', color=color, linewidth=2, alpha=0.6, 
                   label=f'Agent {agent_idx} Path', zorder=3)

        ax.scatter(start_x, start_y, c=color, s=200, alpha=1.0, marker='o', 
                  edgecolors='black', linewidths=2, zorder=5)
        
        if trajectory_xy:
            curr_x, curr_y = trajectory_xy[-1]
            ax.scatter(curr_x, curr_y, c=color, s=150, alpha=0.9, marker='s', 
                      edgecolors='white', linewidths=2, zorder=6)
        
        for goal_idx, (goal_x, goal_y) in enumerate(task_goals_xy):
            ax.scatter(goal_x, goal_y, c=color, s=120, alpha=0.9, marker='^', 
                      edgecolors='black', linewidths=1, zorder=4)
        ax.text(start_x + 0.5, start_y, f'A{agent_idx}', 
               fontsize=10, ha='left', va='center', weight='bold', color=color)
        for goal_idx, (goal_x, goal_y) in enumerate(task_goals_xy):
            ax.text(goal_x + 0.5, goal_y, f'G{agent_idx}-{goal_idx}', 
                   fontsize=8, ha='left', va='center', weight='bold', color=color)

    ax.set_xlim(-1, map_width)
    ax.set_ylim(-1, map_height)
    ax.set_title(f'All Agents Paths at Step {step_count}', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Coordinate (Column)', fontsize=12)
    ax.set_ylabel('Y Coordinate (Row)', fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    ax.legend(unique.values(), unique.keys(), loc='upper right', fontsize=10)
    ax.grid(alpha=0.2, linestyle='--')

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{output_dir}/all_agents_step{step_count}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {filename}")
    return filename