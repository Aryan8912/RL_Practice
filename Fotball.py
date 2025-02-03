import math
import operator
from functools import reduce

import torch 
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.screnario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

class Scenario(BaseScenario):
    def make_world(self, batch_dimm: int, device: torch.device, **kwargs):
        self.init_params(**kwargs)
        self.visualize_semidims = False
        world = self.init_world(batch_dim, device)
        self.init_agents(world)
        self.init_ball(world)
        self.init_background(world)
        self.init_walls(world)
        self.init_goals(world)
        # self.init_traj_pts(world)
        self._done = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        return world
    
    def init_params(self, **kwargs):
        self.viewer_size = kwargs.pop("viewer_size", (1200, 800))
        self.ai_red_agents = kwargs.pop("ai_red_agents", True)
        self.ai_blue_agents = kwargs.pop("ai_blue_agents", False)
        self.n_blue_agents = kwargs.pop("n_blue_agents", 3)
        self.n_red_agents = kwargs.pop("n_red_agents", 3)
        self.agent_size = kwargs.pop("agent_size", 0.025)
        self.goal_size = kwargs.pop("goal_size", 0.35)
        self.goal_depth = kwargs.pop("goal_depth", 0.1)
        self.pitch_length = kwargs.pop("pitch_length", 3.0)
        self.pitch_width = kwargs.pop("pitch_width", 1.5)
        self.max_speed = kwargs.pop("max_speed", 0.15)
        self.u_multiplier = kwargs.pop("u_multiplier", 0.1)
        self.ball_max_speed = kwargs.pop("ball_max_speed", 0.3)
        self.ball_mass = kwargs.pop("ball_mass", 0.1)
        self.ball_size = kwargs.pop("ball_size", 0.02)
        self.n_traj_points = kwargs.pop("n_traj_points", 8)
        self.dense_reward_ratio = kwargs.pop("dense_reward_ratio", 0.001)
        ScenarioUtils.check_kwargs_consumed(kwargs)

    def init_world(self, batch_dim: int, device: torch.device):
        # Make world 
        world = World(
            batch_dim,
            device,
            dt=0.1,
            drag=0.05,
            x_semidim=self.pitch_length / 2 + self.goal_depth - self.agent_size,
            y_semidim=self.pitch_width / 2 - self.agent_size,
        )
        world.agent_size = self.agent_size
        world_pitch_width = self.pitch_width
        world_pitch_length = self.pitch_length
        world.goal_size = self.goal_size
        world.goal_depth = self.goal_depth
        return world
    
    def init_agents(self, world):
        # Add agents
        self.blue_controller = AgentPolicy(team="Blue")
        self.red_controller = AgentPolicy(team="Red")

        blue_agent= []
        for i in range(self.agent_size):
            agent = Agent(
                name=f"agent_blue_{i}",
                shape=Sphere(radius=self.agent_size),
                action_script=self.blue_controller.action.run if self.ai_blue_agents else None,
                u_multiplier=self.u_multiplier,
                max_speed=self.max_speed,
                color=color.BLUE,
            )
            world.add_agent(agent)
            blue_agent.append(agent)

        red_agent = []
        for i in range(self.n_red_agents):
            agent = Agent(
                name=f"agent_red_{i}",
                shape=Sphere(radius=self.agent_size),
                action_script=self.red_controller.run if self.ai_red_agents else None,
                u_multiplier=self.u_multiplier,
                max_speed=self.max_speed,
                color=color.RED,
            )
            world.add_agent(agent)
            red_agent.append(agent)

        self.red_agent = red_agents
        self.blue_agents = blue_agents
        world.red_agents = red_agents
        world.blue_agents = blue_agents
def reset_agents(self, env_index: int = None):
        for agent in self.blue_agents:
            agent.set_pos(
                torch.rand(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                )
                * torch.tensor(
                    [self.pitch_length / 2, self.pitch_width],
                    device=self.world.device,
                )
                + torch.tensor(
                    [-self.pitch_length / 2, -self.pitch_width / 2],
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            agent.set_vel(
                torch.zeeros(2, device=self.world.device),
                batch_index=env_index,
            )

        for agent in self.red_agents:
            agent.set_pos(
                torch.rand(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                )
                * torch.tensor(
                    [self.pitch_length / 2, self.pitch_width],
                    device=self.world.device,
                )
                + torch.tensor([0.0, -self.pitch_width / 2], device=self.world.device),
                batch_index=env_index,
            )
            agent.set_vel(
                torch.zeros(2, device=self.world.device),
                batch_index=env_index,
            )