"""
SVG visualizer for multi-agent pathfinding scenarios.
"""

from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np


class SVGAnimationGenerator:
    def __init__(self, map_size: Tuple[int, int], cell_size: int = 100):
        self.height, self.width = map_size
        self.cell_size = cell_size
        self.canvas_width = self.width * cell_size
        self.canvas_height = self.height * cell_size # cell_size:每个格子的像素大小

        self.agent_colors = [
            '#c1433c',  
            '#2e6f9e',  
            '#6e81af',  
            '#00b9c8',  
            '#f39c12',  
            '#27ae60',  
            '#8e44ad',  
            '#e67e22',  
            '#16a085',  
            '#d35400',  
            '#2980b9',  
            '#c0392b'   
        ]
    
    def _coord_to_pixel(self, row: int, col: int) -> Tuple[float, float]:
        x = col * self.cell_size + self.cell_size / 2
        y = row * self.cell_size + self.cell_size / 2
        return x, y
    
    def _generate_svg_header(self) -> str:
        """生成SVG文件头"""
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="512" height="512" viewBox="0 0 {self.canvas_width} {self.canvas_height}">
<defs>
    <rect id="obstacle" width="{self.cell_size*0.7}" height="{self.cell_size*0.7}" 
          fill="#84A1AE" rx="{self.cell_size*0.15}"/>
    <style>
        .grid-line {{stroke: #84A1AE; stroke-width: {self.cell_size*0.1}; opacity: 0.3;}}
        .agent {{r: {self.cell_size*0.35};}}
        .target {{fill: none; stroke-width: {self.cell_size*0.1}; r: {self.cell_size*0.35};}}
        .path-line {{fill: none; stroke-width: {self.cell_size*0.15}; opacity: 0.4;}}
    </style>
</defs>
'''
    
    def _generate_grid(self) -> str:
        lines = []
        for col in range(self.width + 1):
            x = col * self.cell_size
            lines.append(
                f'<line class="grid-line" x1="{x}" x2="{x}" '
                f'y1="0" y2="{self.canvas_height}" />'
            )
        for row in range(self.height + 1):
            y = row * self.cell_size
            lines.append(
                f'<line class="grid-line" x1="0" x2="{self.canvas_width}" '
                f'y1="{y}" y2="{y}" />'
            )
        
        return '\n'.join(lines)
    
    def _generate_obstacles(self, obstacles: List[Tuple[int, int]]) -> str:
        elements = []
        offset = self.cell_size * 0.15
        
        for row, col in obstacles:
            x, y = self._coord_to_pixel(row, col)
            x -= self.cell_size * 0.35
            y -= self.cell_size * 0.35
            elements.append(f'<use href="#obstacle" x="{x}" y="{y}" />')
        
        return '\n'.join(elements)
    
    def _generate_path_trace(self, trajectory: List[Tuple[int, int]], 
                            color: str) -> str:
        if len(trajectory) < 2:
            return ""
        
        points = []
        for row, col in trajectory:
            x, y = self._coord_to_pixel(row, col)
            points.append(f"{x},{y}")
        
        path_data = " L ".join(points)
        return (f'<path class="path-line" stroke="{color}" '
                f'd="M {path_data}" />')
    
    def _generate_animations(self, agent_id: int, trajectory: List[Tuple[int, int]],
                            goals: List[Tuple[int, int]], color: str,
                            total_steps: int) -> str:
        if not trajectory:
            return ""
        
        # calculate animation duration
        duration = max(10.0, total_steps * 0.05)
        key_times = []
        x_values = []
        y_values = []
        
        for i, (row, col) in enumerate(trajectory):
            t = i / (len(trajectory) - 1) if len(trajectory) > 1 else 0
            key_times.append(f"{t:.10f}")
            x, y = self._coord_to_pixel(row, col)
            x_values.append(f"{x}")
            y_values.append(f"{y}")
        
        key_times.append("1.0")
        x_values.append(x_values[-1])
        y_values.append(y_values[-1])
        
        key_times_str = ";".join(key_times)
        x_values_str = ";".join(x_values)
        y_values_str = ";".join(y_values)
        start_x, start_y = self._coord_to_pixel(*trajectory[0])
        
        agent_circle = f'''<circle class="agent" cx="{start_x}" cy="{start_y}" fill="{color}" r="{self.cell_size*0.35}">
    <animate attributeName="cx" dur="{duration}s" keyTimes="{key_times_str}" 
             repeatCount="indefinite" values="{x_values_str}"/>
    <animate attributeName="cy" dur="{duration}s" keyTimes="{key_times_str}" 
             repeatCount="indefinite" values="{y_values_str}"/>
    <animate attributeName="visibility" dur="{duration}s" 
             keyTimes="0.0;{key_times[-2]};1.0" 
             repeatCount="indefinite" values="visible;visible;hidden"/>
</circle>'''
        
        target_circles = []
        for goal_idx, (goal_row, goal_col) in enumerate(goals):
            goal_x, goal_y = self._coord_to_pixel(goal_row, goal_col)
            
            disappear_time = 1.0
            for i, (traj_row, traj_col) in enumerate(trajectory):
                if (traj_row, traj_col) == (goal_row, goal_col):
                    disappear_time = i / (len(trajectory) - 1) if len(trajectory) > 1 else 1.0
                    break
            
            next_time = min(1.0, disappear_time + 0.01)
            
            target_circle = f'''<circle class="target" cx="{goal_x}" cy="{goal_y}" r="{self.cell_size*0.35}" stroke="{color}">
    <animate attributeName="visibility" dur="{duration}s" 
             keyTimes="0.0;{disappear_time:.10f};{next_time:.10f};1.0" 
             repeatCount="indefinite" values="visible;visible;hidden;hidden"/>
</circle>'''
            target_circles.append(target_circle)
        
        return agent_circle + '\n' + '\n'.join(target_circles)
    
    def generate(self, obstacles: List[Tuple[int, int]],
                agent_trajectories: Dict[int, List[Tuple[int, int]]],
                task_instances: List[List[Tuple[int, int]]],
                output_path: str = 'animation.svg') -> str:
        total_steps = max(len(traj) for traj in agent_trajectories.values()) if agent_trajectories else 1
        
        svg_parts = [
            self._generate_svg_header(),
            self._generate_grid(),
            self._generate_obstacles(obstacles)
        ]
        
        for agent_id in sorted(agent_trajectories.keys()):
            trajectory = agent_trajectories[agent_id]
            color = self.agent_colors[agent_id % len(self.agent_colors)]
            path = self._generate_path_trace(trajectory, color)
            if path:
                svg_parts.append(path)
        
        for agent_id in sorted(agent_trajectories.keys()):
            trajectory = agent_trajectories[agent_id]
            goals = task_instances[agent_id] if agent_id < len(task_instances) else []
            color = self.agent_colors[agent_id % len(self.agent_colors)]
            
            animations = self._generate_animations(
                agent_id, trajectory, goals, color, total_steps
            )
            svg_parts.append(animations)
        
        svg_parts.append('</svg>')
        
        svg_content = '\n'.join(svg_parts)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        print(f"SVG output: {output_path}")
        return str(output_file)


def create_svg_animation(map_size: Tuple[int, int],
                        obstacles: List[Tuple[int, int]],
                        agent_trajectories: Dict[int, List[Tuple[int, int]]],
                        task_instances: List[List[Tuple[int, int]]],
                        output_path: str = 'renders/animation.svg',
                        cell_size: int = 100) -> str:
    generator = SVGAnimationGenerator(map_size, cell_size)
    return generator.generate(obstacles, agent_trajectories, task_instances, output_path)