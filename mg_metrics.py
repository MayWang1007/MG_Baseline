"""
性能指标计算模块
计算多智能体路径规划的各项性能指标
"""

import time
from typing import List, Tuple, Dict
import numpy as np


class PerformanceMetrics:
    def __init__(self, num_agents: int, task_instances: List[List[Tuple]]):
        self.num_agents = num_agents
        self.task_instances = task_instances
        self.total_goals = sum(len(tasks) for tasks in task_instances)
        
        self.start_time = None
        self.end_time = None
        self.agent_start_times = {}
        self.agent_completion_times = {}
        
        self.goals_completed = [0] * num_agents
        self.agent_makespan = {}  
        
        self.total_steps = 0
        self.agent_steps = {}  
    
    def start(self):
        self.start_time = time.time()
        for i in range(self.num_agents):
            self.agent_start_times[i] = self.start_time
            self.agent_steps[i] = 0
    
    def record_goal_completion(self, agent_id: int, step: int):

        self.goals_completed[agent_id] += 1

        if self.goals_completed[agent_id] >= len(self.task_instances[agent_id]):
            if agent_id not in self.agent_completion_times:
                self.agent_completion_times[agent_id] = time.time()
                self.agent_makespan[agent_id] = step
    
    def record_step(self, agent_id: int, moved: bool = True):
        if moved:
            self.agent_steps[agent_id] += 1
    
    def finish(self, total_steps: int):
        self.end_time = time.time()
        self.total_steps = total_steps
    
    def calculate_metrics(self) -> Dict:
        total_runtime = self.end_time - self.start_time

        completed_agents = sum(1 for c in self.goals_completed 
                              if c >= len(self.task_instances[self.goals_completed.index(c)]))
        success_rate = completed_agents / self.num_agents if self.num_agents > 0 else 0
        

        completed_goals = sum(self.goals_completed)
        avg_throughput = completed_goals / total_runtime if total_runtime > 0 else 0

        if self.agent_makespan:
            makespan = max(self.agent_makespan.values())
        else:
            makespan = self.total_steps

        soc = sum(self.agent_steps.values())

        if self.agent_completion_times:
            completion_times = [
                self.agent_completion_times[aid] - self.agent_start_times[aid]
                for aid in self.agent_completion_times
            ]
            avg_completion_time = np.mean(completion_times)
        else:
            avg_completion_time = total_runtime
        
        agent_metrics = []
        for agent_id in range(self.num_agents):
            total_agent_goals = len(self.task_instances[agent_id])
            completed = self.goals_completed[agent_id]
            
            agent_metric = {
                'agent_id': agent_id,
                'total_goals': total_agent_goals,
                'completed_goals': completed,
                'completion_rate': completed / total_agent_goals if total_agent_goals > 0 else 0,
                'steps': self.agent_steps.get(agent_id, 0),
                'makespan': self.agent_makespan.get(agent_id, self.total_steps),
                'completed': completed >= total_agent_goals
            }
            agent_metrics.append(agent_metric)
        
        return {

            'total_runtime': total_runtime,
            'total_steps': self.total_steps,
            'success_rate': success_rate,
            'avg_throughput': avg_throughput,
            'makespan': makespan,
            'sum_of_costs': soc,
            'avg_completion_time': avg_completion_time,
            

            'total_goals': self.total_goals,
            'completed_goals': completed_goals,
            'goal_completion_rate': completed_goals / self.total_goals if self.total_goals > 0 else 0,

            'num_agents': self.num_agents,
            'completed_agents': completed_agents,
            'agent_metrics': agent_metrics
        }
    
    def print_summary(self):
        metrics = self.calculate_metrics()
        print(f"{'='*70}\n")
        print(f"\nSystem Parameters:")
        print(f"  Runtime: {metrics['total_runtime']:.3f}秒")
        print(f"  Total Steps: {metrics['total_steps']}")
        print(f"  Makespan: {metrics['makespan']} 步")
        print(f"  Average Throughput: {metrics['avg_throughput']:.3f} 目标/秒")
        print(f"  Sum of Costs (SoC): {metrics['sum_of_costs']} 步")
        print(f"  Average Makespan: {metrics['avg_completion_time']:.3f}秒")
        
        print(f"\nsuccess rate:")
        print(f" Agents success rate: {metrics['completed_agents']}/{metrics['num_agents']} "
              f"({metrics['success_rate']*100:.1f}%)")
        print(f" Total goal success rate: {metrics['completed_goals']}/{metrics['total_goals']} "
              f"({metrics['goal_completion_rate']*100:.1f}%)")
        
        print(f"\nEach agent:")
        for am in metrics['agent_metrics']:
            print(f"Agent {am['agent_id']}: "
                  f"{am['completed_goals']}/{am['total_goals']} , "
                  f"{am['steps']} , "
                  f"Makespan={am['makespan']}")
        
        print(f"{'='*70}\n")
        
        return metrics


def save_metrics_to_csv(metrics: Dict, output_path: str):
    import csv
    from pathlib import Path
    from datetime import datetime
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    global_file = Path(output_path).with_suffix('.global.csv')
    file_exists = global_file.exists()
    
    with open(global_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'timestamp', 'num_agents', 'total_goals', 'completed_goals',
                'success_rate(%)', 'goal_completion_rate(%)', 
                'runtime(s)', 'total_steps', 'makespan', 'sum_of_costs',
                'avg_throughput(goals/s)', 'avg_completion_time(s)'
            ])
        
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            metrics['num_agents'],
            metrics['total_goals'],
            metrics['completed_goals'],
            round(metrics['success_rate'] * 100, 2),
            round(metrics['goal_completion_rate'] * 100, 2),
            round(metrics['total_runtime'], 3),
            metrics['total_steps'],
            metrics['makespan'],
            metrics['sum_of_costs'],
            round(metrics['avg_throughput'], 3),
            round(metrics['avg_completion_time'], 3)
        ])

    agent_file = Path(output_path).with_suffix('.agents.csv')
    file_exists = agent_file.exists()
    
    with open(agent_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'timestamp', 'agent_id', 'total_goals', 'completed_goals',
                'completion_rate(%)', 'steps', 'makespan', 'completed'
            ])
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for am in metrics['agent_metrics']:
            writer.writerow([
                timestamp,
                am['agent_id'],
                am['total_goals'],
                am['completed_goals'],
                round(am['completion_rate'] * 100, 2),
                am['steps'],
                am['makespan'],
                am['completed']
            ])

    print(f"Global result: {global_file}")
    print(f"Each agent result: {agent_file}")