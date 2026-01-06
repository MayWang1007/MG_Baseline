import numpy as np
import yaml
import heapq
from typing import List, Tuple, Dict, Optional, Set
from follower.inference import FollowerInferenceConfig, FollowerInference
from mg_metrics import PerformanceMetrics, save_metrics_to_csv
from mg_visualizer import create_svg_animation

class AStarPlanner:
    def __init__(
        self, 
        obstacles: List[Tuple], 
        map_size: Tuple[int, int],
        use_static_cost: bool = True,
        use_dynamic_cost: bool = True,
        static_cost_weight: float = 1.0,
        dynamic_cost_weight: float = 2.0
    ):
        self.obstacles = set(obstacles)
        self.height, self.width = map_size
        self.use_static_cost = use_static_cost
        self.use_dynamic_cost = use_dynamic_cost
        self.static_cost_weight = static_cost_weight
        self.dynamic_cost_weight = dynamic_cost_weight
        
        self.static_penalty = self._precompute_static_penalty() if use_static_cost else None
        self.dynamic_occupations: Set[Tuple] = set()
        
    def _precompute_static_penalty(self, max_penalty: float = 5.0) -> np.ndarray:
        penalty_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) in self.obstacles:
                    penalty_map[row, col] = np.inf
                else:
                    min_dist = float('inf')
                    for obs_row, obs_col in self.obstacles:
                        dist = abs(row - obs_row) + abs(col - obs_col)
                        min_dist = min(min_dist, dist)
                    if min_dist <= 3:
                        penalty_map[row, col] = max_penalty * np.exp(-min_dist / 2.0)
        return penalty_map
    
    def update_dynamic_occupations(self, agent_positions: List[Tuple], exclude_agent: int = None):
        self.dynamic_occupations = set()
        for i, pos in enumerate(agent_positions):
            if exclude_agent is None or i != exclude_agent:
                self.dynamic_occupations.add(pos)
    
    def heuristic(self, pos: Tuple, goal: Tuple) -> float:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def get_cost(self, pos: Tuple) -> float:
        cost = 1.0
        if self.use_static_cost and self.static_penalty is not None:
            row, col = pos
            if 0 <= row < self.height and 0 <= col < self.width:
                cost += self.static_cost_weight * self.static_penalty[row, col]
        if self.use_dynamic_cost and pos in self.dynamic_occupations:
            cost += self.dynamic_cost_weight
        return cost
    
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
    
    def find_path(
        self, 
        start: Tuple, 
        goal: Tuple,
        agent_positions: Optional[List[Tuple]] = None,
        current_agent_id: Optional[int] = None
    ) -> Optional[List[Tuple]]:
        if start == goal:
            return [start]
        
        if self.use_dynamic_cost and agent_positions is not None:
            self.update_dynamic_occupations(agent_positions, current_agent_id)
        
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
                edge_cost = self.get_cost(neighbor)
                tentative_g = g_score[current] + edge_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None

class CompletedAgentManager:
    def __init__(
        self,
        num_agents: int,
        start_positions: List[Tuple],
        avoid_blocking: bool = True
    ):
        self.num_agents = num_agents
        self.start_positions = start_positions
        self.avoid_blocking = avoid_blocking
        
        self.agent_completed = [False] * num_agents
        self.agent_is_static = [False] * num_agents
        self.agent_should_return = [False] * num_agents
        self.agent_at_parking = [False] * num_agents
        
    def mark_completed(self, agent_id: int):
        self.agent_completed[agent_id] = True
        self.agent_is_static[agent_id] = False
        self.agent_should_return[agent_id] = False
        self.agent_at_parking[agent_id] = False
    
    def is_completed(self, agent_id: int) -> bool:
        return self.agent_completed[agent_id]

    def update_states(
        self,
        current_positions: List[Tuple],
        active_targets: List[Tuple]
    ):
        for agent_id in range(self.num_agents):
            if not self.agent_completed[agent_id]:
                continue
            
            current_pos = current_positions[agent_id]
            parking_pos = self.start_positions[agent_id]
            
            # 检查是否在起点
            if current_pos == parking_pos:
                self.agent_at_parking[agent_id] = True
                self.agent_should_return[agent_id] = False
                self.agent_is_static[agent_id] = True
                continue
            
            # 检查是否占用其他智能体的目标
            if self.avoid_blocking:
                is_blocking = self._is_blocking_others(
                    agent_id, current_pos, active_targets
                )
                if is_blocking:
                    self.agent_should_return[agent_id] = True
                    self.agent_is_static[agent_id] = False
                else:
                    self.agent_should_return[agent_id] = False
                    self.agent_is_static[agent_id] = True
            else:
                # 不避让，直接静止
                self.agent_should_return[agent_id] = False
                self.agent_is_static[agent_id] = True
    
    def should_skip_planning(self, agent_id: int) -> bool:
        return self.agent_completed[agent_id] and self.agent_is_static[agent_id]
    
    def should_force_action_zero(self, agent_id: int) -> bool:
        return self.agent_completed[agent_id] and self.agent_is_static[agent_id]
    
    def get_target_for_returning(self, agent_id: int) -> Optional[Tuple]:
        if self.agent_should_return[agent_id]:
            return self.start_positions[agent_id]
        return None
    
    def get_static_obstacles(self, current_positions: List[Tuple]) -> List[Tuple]:
        static_obstacles = []
        for agent_id in range(self.num_agents):
            if self.agent_is_static[agent_id]:
                static_obstacles.append(current_positions[agent_id])
        return static_obstacles
    
    def is_static_obstacle(self, agent_id: int) -> bool:
        return self.agent_is_static[agent_id]
    
    def _is_blocking_others(
        self,
        agent_id: int,
        position: Tuple,
        all_targets: List[Tuple]
    ) -> bool:
        for other_id in range(self.num_agents):
            if other_id == agent_id:
                continue
            if not self.agent_completed[other_id] and all_targets[other_id] == position:
                return True
        return False
    
    def get_status_string(self, agent_id: int) -> str:
        if not self.agent_completed[agent_id]:
            return ""
        if self.agent_is_static[agent_id]:
            return "stay"
        if self.agent_should_return[agent_id]:
            return "return"
        if self.agent_at_parking[agent_id]:
            return "parking"
        return ""
    
class SimpleMapfEnv:
    def __init__(
        self, 
        map_str: str, 
        start_positions: List[Tuple], 
        obs_radius: int = 5,
        use_improved_planner: bool = True,
        static_cost_weight: float = 1.0,
        dynamic_cost_weight: float = 2.0,
        completion_manager: Optional[CompletedAgentManager] = None
    ):
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
        self.completion_manager = completion_manager
        
        if use_improved_planner:
            self.planners = [
                AStarPlanner(
                    self.obstacles, 
                    (self.height, self.width),
                    use_static_cost=True,
                    use_dynamic_cost=True,
                    static_cost_weight=static_cost_weight,
                    dynamic_cost_weight=dynamic_cost_weight
                )
                for _ in range(self.num_agents)
            ]
            self.planner_type = 'ImprovedA*'

    
    def plan_path(self, agent_id: int, target: Tuple) -> List[Tuple]:
        if self.completion_manager and self.completion_manager.should_skip_planning(agent_id):
            return None
        
        if self.completion_manager:
            static_obstacles = self.completion_manager.get_static_obstacles(self.current_positions)
            original_obstacles = self.planners[agent_id].obstacles.copy()
            self.planners[agent_id].obstacles.update(static_obstacles)
            
            path = self.planners[agent_id].find_path(
                start=self.current_positions[agent_id],
                goal=target,
                agent_positions=self.current_positions,
                current_agent_id=agent_id
            )
            
            self.planners[agent_id].obstacles = original_obstacles
            return path
        else:
            return self.planners[agent_id].find_path(
                start=self.current_positions[agent_id],
                goal=target,
                agent_positions=self.current_positions,
                current_agent_id=agent_id
            )

    def get_observation(self, agent_id: int, target: Tuple, path: Optional[List[Tuple]]) -> Dict:
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
                
                if (world_row < 0 or world_row >= self.height or 
                    world_col < 0 or world_col >= self.width or
                    (world_row, world_col) in self.obstacles):
                    obstacles_map[local_row, local_col] = -1.0
        
        if self.completion_manager:
            static_obstacles = self.completion_manager.get_static_obstacles(self.current_positions)
            for static_pos in static_obstacles:
                if static_pos == pos:
                    continue
                dx, dy = static_pos[0] - row, static_pos[1] - col
                if abs(dx) <= r and abs(dy) <= r:
                    obstacles_map[r + dx, r + dy] = -1.0
        
        if path:
            for px, py in path:
                dx, dy = px - row, py - col
                if abs(dx) <= r and abs(dy) <= r:
                    obstacles_map[r + dx, r + dy] = 1.0
        
        for other_id in range(self.num_agents):
            if self.completion_manager and self.completion_manager.is_static_obstacle(other_id):
                continue
            
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
        action_map = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        
        new_positions = []
        for agent_id, action in enumerate(actions):
            action = int(action)
            current = self.current_positions[agent_id]
            
            if action == 0:
                new_positions.append(current)
                continue
            
            if action in action_map:
                delta = action_map[action]
                new_pos = (current[0] + delta[0], current[1] + delta[1])
                
                effective_obstacles = list(self.obstacles)
                if self.completion_manager:
                    static_obstacles = self.completion_manager.get_static_obstacles(self.current_positions)
                    effective_obstacles.extend(static_obstacles)
                
                if (0 <= new_pos[0] < self.height and 
                    0 <= new_pos[1] < self.width and
                    new_pos not in effective_obstacles):
                    
                    conflict = any(
                        new_pos == self.current_positions[j] 
                        for j in range(self.num_agents) 
                        if j != agent_id and not (
                            self.completion_manager and 
                            self.completion_manager.is_static_obstacle(j)
                        )
                    )
                    new_positions.append(current if conflict else new_pos)
                else:
                    new_positions.append(current)
            else:
                new_positions.append(current)
        
        self.current_positions = new_positions
        return new_positions


class MultiGoalManager:
    def __init__(
        self,
        task_instances: List[List[Tuple]],
        start_positions: List[Tuple],
        avoid_blocking: bool = True
    ):
        self.task_instances = task_instances
        self.start_positions = start_positions
        self.num_agents = len(task_instances)
        self.current_goal_idx = [0] * self.num_agents
        self.goals_completed = [0] * self.num_agents
        self.total_goals = [len(tasks) for tasks in task_instances]
        
        self.completion_manager = CompletedAgentManager(
            num_agents=self.num_agents,
            start_positions=start_positions,
            avoid_blocking=avoid_blocking
        )
    
    def get_current_targets(self, current_positions: List[Tuple]) -> List[Tuple]:
        active_targets = []
        for agent_id in range(self.num_agents):
            if self.current_goal_idx[agent_id] < len(self.task_instances[agent_id]):
                active_targets.append(self.task_instances[agent_id][self.current_goal_idx[agent_id]])
            else:
                active_targets.append(current_positions[agent_id])
        
        self.completion_manager.update_states(current_positions, active_targets)
        
        adjusted_targets = []
        for agent_id in range(self.num_agents):
            if not self.completion_manager.agent_completed[agent_id]:
                adjusted_targets.append(active_targets[agent_id])
            else:
                return_target = self.completion_manager.get_target_for_returning(agent_id)
                if return_target:
                    adjusted_targets.append(return_target)
                else:
                    adjusted_targets.append(current_positions[agent_id])
        
        return adjusted_targets
    
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
                
                if self.current_goal_idx[agent_id] >= len(self.task_instances[agent_id]):
                    self.completion_manager.mark_completed(agent_id) 
        return completed
    
    def all_completed(self) -> bool:
        return all(self.goals_completed[i] >= self.total_goals[i] 
                   for i in range(self.num_agents))
    
    def all_settled(self) -> bool:
        for agent_id in range(self.num_agents):
            if self.completion_manager.agent_completed[agent_id]:
                if not (self.completion_manager.agent_is_static[agent_id] or 
                       self.completion_manager.agent_at_parking[agent_id]):
                    return False
        return True
    
    def get_progress(self) -> str:
        parts = []
        for i in range(self.num_agents):
            completed = self.goals_completed[i]
            total = self.total_goals[i]
            status = self.completion_manager.get_status_string(i)
            parts.append(f"A{i}:{completed}/{total}{status}")
        return " ".join(parts)

def run_mgmapf_python(
    map_name: str,
    task_instances: List[List[Tuple]],
    start_positions: List[Tuple],
    max_steps: int = 2000,
    obs_radius: int = 5,
    use_improved_planner: bool = True,
    static_cost_weight: float = 1.0,
    dynamic_cost_weight: float = 2.0,
    avoid_blocking: bool = True,
    save_animation: bool = True,
    animation_path: str = 'renders/animation.svg',
    save_metrics: bool = True,
    metrics_path: str = 'results/metrics.csv'
):
    with open('./env/test-maps.yaml', 'r') as f:
        maps = yaml.safe_load(f)
    map_str = maps[map_name]
    
    goal_manager = MultiGoalManager(
        task_instances, 
        start_positions,
        avoid_blocking=avoid_blocking
    )
    env = SimpleMapfEnv(
        map_str, 
        start_positions, 
        obs_radius,
        use_improved_planner=use_improved_planner,
        static_cost_weight=static_cost_weight,
        dynamic_cost_weight=dynamic_cost_weight,
        completion_manager=goal_manager.completion_manager
    )
    
    algo_cfg = FollowerInferenceConfig()
    algo = FollowerInference(algo_cfg)
    algo.reset_states()
    
    from mg_metrics import PerformanceMetrics
    metrics = PerformanceMetrics(env.num_agents, task_instances)
    metrics.start()
    
    agent_trajectories = {i: [] for i in range(env.num_agents)}
    
    step_count = 0
    
    while step_count < max_steps:
        step_count += 1
        
        for agent_id in range(env.num_agents):
            agent_trajectories[agent_id].append(env.current_positions[agent_id])
        
        current_targets = goal_manager.get_current_targets(env.current_positions)
        
        # path planning
        paths = []
        for agent_id in range(env.num_agents):
            if goal_manager.completion_manager.should_skip_planning(agent_id):
                paths.append([])
            else:
                path = env.plan_path(agent_id, current_targets[agent_id])
                paths.append(path if path else [])
        
        # generate observations
        observations = []
        for agent_id in range(env.num_agents):
            obs = env.get_observation(
                agent_id, 
                current_targets[agent_id],
                paths[agent_id]
            )
            observations.append(obs)
        
        # get actions,static agents stay still
        actions = algo.act(observations)
        for agent_id in range(env.num_agents):
            if goal_manager.completion_manager.should_force_action_zero(agent_id):
                actions[agent_id] = 0

        # environment response
        prev_positions = list(env.current_positions)
        new_positions = env.step(actions)
        for agent_id in range(env.num_agents):
            moved = prev_positions[agent_id] != new_positions[agent_id]
            metrics.record_step(agent_id, moved)
        completed = goal_manager.check_and_update(new_positions)
        for agent_id in completed:
            metrics.record_goal_completion(agent_id, step_count)
        
        # check termination
        if goal_manager.all_completed() and goal_manager.all_settled():
            break
        elif step_count % 100 == 0:
            print(f"[Step {step_count}] {goal_manager.get_progress()}")
    metrics.finish(step_count)
    
    # get results
    result_metrics = metrics.print_summary()
    if save_metrics:
        from mg_metrics import save_metrics_to_csv
        save_metrics_to_csv(result_metrics, metrics_path)
    
    if save_animation:
        from mg_visualizer import create_svg_animation
        print(f"\nGeneraing SVG...")
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
