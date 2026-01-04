import numpy as np
import yaml
import heapq
from typing import List, Tuple, Dict, Optional
from follower.inference import FollowerInferenceConfig, FollowerInference
from mg_metrics import PerformanceMetrics

class SimpleAStarPlanner:
    def __init__(self, obstacles: List[Tuple], map_size: Tuple[int, int]):
        self.obstacles = set(obstacles)
        self.height, self.width = map_size
    
    def heuristic(self, pos: Tuple, goal: Tuple) -> float:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def get_neighbors(self, pos: Tuple) -> List[Tuple]:
        row, col = pos
        neighbors = []
        for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.height and 
                0 <= new_col < self.width and
                (new_row, new_col) not in self.obstacles):
                neighbors.append((new_row, new_col))
        return neighbors
    
    def find_path(self, start: Tuple, goal: Tuple) -> Optional[List[Tuple]]:
        if start == goal:
            return [start]
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return list(reversed(path))
            
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None


class SimpleMapfEnv:
    def __init__(self, map_str: str, start_positions: List[Tuple], obs_radius: int = 5):
        self.map_lines = [line.strip() for line in map_str.strip().split('\n') if line.strip()]
        self.height = len(self.map_lines)
        self.width = len(self.map_lines[0]) if self.height > 0 else 0
        
        self.obstacles = []
        for row in range(self.height):
            for col in range(len(self.map_lines[row])):
                if self.map_lines[row][col] == '#':
                    self.obstacles.append((row, col))
        
        self.num_agents = len(start_positions)
        self.start_positions = start_positions
        self.current_positions = list(start_positions)
        self.obs_radius = obs_radius
        self.planner = SimpleAStarPlanner(self.obstacles, (self.height, self.width))
    
    def get_observation(self, agent_id: int, target: Tuple, path: List[Tuple]) -> Dict:
        pos = self.current_positions[agent_id]
        row, col = pos
        r = self.obs_radius
        
        obs_size = 2 * r + 1
        obstacles_map = np.zeros((obs_size, obs_size), dtype=np.float32)
        agents_map = np.zeros((obs_size, obs_size), dtype=np.float32)
        
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                world_row, world_col = row + dr, col + dc
                local_row, local_col = r + dr, r + dc
                
                if world_row < 0 or world_row >= self.height or \
                   world_col < 0 or world_col >= self.width:
                    obstacles_map[local_row, local_col] = -1.0
                elif (world_row, world_col) in self.obstacles:
                    obstacles_map[local_row, local_col] = -1.0
        
        if path:
            for px, py in path:
                dx, dy = px - row, py - col
                if abs(dx) <= r and abs(dy) <= r:
                    local_row, local_col = r + dx, r + dy
                    obstacles_map[local_row, local_col] = 1.0
        
        for other_id in range(self.num_agents):
            if other_id == agent_id:
                agents_map[r, r] = 1.0
            else:
                other_pos = self.current_positions[other_id]
                dr, dc = other_pos[0] - row, other_pos[1] - col
                if abs(dr) <= r and abs(dc) <= r:
                    agents_map[r + dr, r + dc] = 1.0
        
        return {
            'xy': np.array(list(pos), dtype=np.float32),
            'target_xy': np.array(list(target), dtype=np.float32),
            'obs': np.stack([obstacles_map, agents_map], axis=0).astype(np.float32)
        }
    
    def step(self, actions: np.ndarray) -> List[Tuple]:
        action_map = {
            0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)
        }
        
        new_positions = []
        for agent_id, action in enumerate(actions):
            action = int(action)
            current = self.current_positions[agent_id]
            
            if action in action_map:
                delta = action_map[action]
                new_pos = (current[0] + delta[0], current[1] + delta[1])
                
                if (0 <= new_pos[0] < self.height and 
                    0 <= new_pos[1] < self.width and
                    new_pos not in self.obstacles):
                    
                    conflict = any(new_pos == self.current_positions[j] 
                                   for j in range(self.num_agents) if j != agent_id)
                    
                    new_positions.append(current if conflict else new_pos)
                else:
                    new_positions.append(current)
            else:
                new_positions.append(current)
        
        self.current_positions = new_positions
        return new_positions


class MultiGoalManager:
    def __init__(self, task_instances: List[List[Tuple]], start_positions: List[Tuple]):
        self.task_instances = task_instances
        self.start_positions = start_positions
        self.num_agents = len(task_instances)
        self.current_goal_idx = [0] * self.num_agents
        self.goals_completed = [0] * self.num_agents
        self.total_goals = [len(tasks) for tasks in task_instances]
    
    def get_current_targets(self) -> List[Tuple]:
        targets = []
        for agent_id in range(self.num_agents):
            if self.current_goal_idx[agent_id] < len(self.task_instances[agent_id]):
                targets.append(self.task_instances[agent_id][self.current_goal_idx[agent_id]])
            else:
                targets.append(self.task_instances[agent_id][-1])
        return targets
    
    def check_and_update(self, current_positions: List[Tuple]) -> List[int]:
        completed = []
        for agent_id in range(self.num_agents):
            if self.current_goal_idx[agent_id] >= len(self.task_instances[agent_id]):
                continue
            
            target = self.task_instances[agent_id][self.current_goal_idx[agent_id]]
            if current_positions[agent_id] == target:
                self.goals_completed[agent_id] += 1
                self.current_goal_idx[agent_id] += 1
                completed.append(agent_id)
        return completed
    
    def all_completed(self) -> bool:
        return all(self.goals_completed[i] >= self.total_goals[i] 
                   for i in range(self.num_agents))
    
    def get_progress(self) -> str:
        return " ".join(f"A{i}:{self.goals_completed[i]}/{self.total_goals[i]}" 
                       for i in range(self.num_agents))


def run_mgmapf_python(
    map_name: str,
    task_instances: List[List[Tuple]],
    start_positions: List[Tuple],
    max_steps: int = 2000,
    obs_radius: int = 5,
    save_animation: bool = True,
    animation_path: str = 'renders/animation.svg',
    save_metrics: bool = True,
    metrics_path: str = 'results/metrics.csv'
):
    with open('./env/test-maps.yaml', 'r') as f:
        maps = yaml.safe_load(f)
    map_str = maps[map_name]
    
    env = SimpleMapfEnv(map_str, start_positions, obs_radius)
    goal_manager = MultiGoalManager(task_instances, start_positions)
    
    algo_cfg = FollowerInferenceConfig()
    algo = FollowerInference(algo_cfg)
    algo.reset_states()
    
    metrics = PerformanceMetrics(env.num_agents, task_instances)
    metrics.start()
    
    agent_trajectories = {i: [] for i in range(env.num_agents)}
    
    step_count = 0
    
    while step_count < max_steps:
        step_count += 1
        
        for agent_id in range(env.num_agents):
            agent_trajectories[agent_id].append(env.current_positions[agent_id])
        
        current_targets = goal_manager.get_current_targets()
        
        paths = []
        for agent_id in range(env.num_agents):
            path = env.planner.find_path(
                env.current_positions[agent_id],
                current_targets[agent_id]
            )
            paths.append(path if path else [])
        
        observations = []
        for agent_id in range(env.num_agents):
            obs = env.get_observation(
                agent_id, 
                current_targets[agent_id],
                paths[agent_id]
            )
            observations.append(obs)
        
        actions = algo.act(observations)
        
        prev_positions = list(env.current_positions)
        new_positions = env.step(actions)
        
        for agent_id in range(env.num_agents):
            moved = prev_positions[agent_id] != new_positions[agent_id]
            metrics.record_step(agent_id, moved)
        
        completed = goal_manager.check_and_update(new_positions)
        for agent_id in completed:
            metrics.record_goal_completion(agent_id, step_count)
        if goal_manager.all_completed():
            break
        if step_count % 100 == 0:
            print(f"[Step {step_count}] {goal_manager.get_progress()}")
    metrics.finish(step_count)    
    result_metrics = metrics.print_summary()
    
    if save_metrics:
        from mg_metrics import save_metrics_to_csv
        save_metrics_to_csv(result_metrics, metrics_path)
    
    if save_animation:
        from mg_visualizer import create_svg_animation
        print(f"\nGenerating SVG...")
        create_svg_animation(
            map_size=(env.height, env.width),
            obstacles=env.obstacles,
            agent_trajectories=agent_trajectories,
            task_instances=task_instances,
            output_path=animation_path,
            cell_size=100
        )
    
    return {
        'total_steps': step_count,
        'goals_completed': goal_manager.goals_completed,
        'success': goal_manager.all_completed(),
        'trajectories': agent_trajectories,
        'metrics': result_metrics
    }