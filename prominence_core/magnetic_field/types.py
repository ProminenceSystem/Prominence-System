"""
Magnetic Field Types Module
Defines the core types and structures for magnetic field representations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import numpy as np

@dataclass
class MagneticField:
    """
    Represents a magnetic field in the Prominence System.
    """
    strength: torch.Tensor
    polarity: str
    gradient: torch.Tensor
    topology: Optional[List[Tuple[int, int]]] = None
    
    @classmethod
    def create(cls, strength: float, polarity: str = "north") -> 'MagneticField':
        """
        Create a new magnetic field instance with default parameters.
        """
        return cls(
            strength=torch.tensor([strength]),
            polarity=polarity,
            gradient=torch.zeros(3),  # 3D gradient
            topology=[]
        )
    
    def calculate_flux(self, area: torch.Tensor) -> torch.Tensor:
        """
        Calculate magnetic flux through a given area.
        Φ = B · A
        """
        return torch.dot(self.strength, area)
    
    def calculate_energy_density(self) -> torch.Tensor:
        """
        Calculate magnetic energy density.
        u = B² / (2μ₀)
        """
        mu0 = 4 * torch.pi * 1e-7
        return torch.sum(self.strength ** 2) / (2 * mu0)
    
    def calculate_tension(self) -> torch.Tensor:
        """
        Calculate magnetic tension force.
        T = (B · ∇)B / μ₀
        """
        mu0 = 4 * torch.pi * 1e-7
        return torch.matmul(self.strength, self.gradient) / mu0
    
    def calculate_pressure(self) -> torch.Tensor:
        """
        Calculate magnetic pressure.
        P = B² / (2μ₀)
        """
        mu0 = 4 * torch.pi * 1e-7
        return torch.sum(self.strength ** 2) / (2 * mu0)
    
    def update_gradient(self, new_gradient: torch.Tensor):
        """
        Update the magnetic field gradient.
        """
        self.gradient = new_gradient
    
    def add_topology_connection(self, start: int, end: int):
        """
        Add a topological connection in the magnetic field.
        """
        if self.topology is None:
            self.topology = []
        self.topology.append((start, end))
    
    def to_dict(self) -> dict:
        """
        Convert the magnetic field to a dictionary representation.
        """
        return {
            "strength": self.strength.tolist(),
            "polarity": self.polarity,
            "gradient": self.gradient.tolist(),
            "topology": self.topology
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MagneticField':
        """
        Create a magnetic field instance from a dictionary.
        """
        return cls(
            strength=torch.tensor(data["strength"]),
            polarity=data["polarity"],
            gradient=torch.tensor(data["gradient"]),
            topology=data.get("topology")
        )

@dataclass
class MagneticFieldLine:
    """
    Represents a magnetic field line in 3D space.
    """
    points: List[torch.Tensor]
    strength: torch.Tensor
    curvature: Optional[torch.Tensor] = None
    
    def calculate_length(self) -> float:
        """
        Calculate the length of the field line.
        """
        length = 0.0
        for i in range(len(self.points) - 1):
            length += torch.norm(self.points[i + 1] - self.points[i]).item()
        return length
    
    def calculate_curvature(self) -> torch.Tensor:
        """
        Calculate the curvature of the field line.
        """
        if len(self.points) < 3:
            return torch.zeros(1)
        
        curvature = []
        for i in range(1, len(self.points) - 1):
            v1 = self.points[i] - self.points[i - 1]
            v2 = self.points[i + 1] - self.points[i]
            angle = torch.acos(torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2)))
            curvature.append(angle)
        
        self.curvature = torch.tensor(curvature)
        return self.curvature
    
    def resample(self, num_points: int) -> 'MagneticFieldLine':
        """
        Resample the field line to have a specific number of points.
        """
        if len(self.points) == num_points:
            return self
        
        total_length = self.calculate_length()
        segment_length = total_length / (num_points - 1)
        
        new_points = [self.points[0]]
        current_point = 0
        current_distance = 0.0
        
        for i in range(1, num_points - 1):
            target_distance = i * segment_length
            while current_distance < target_distance and current_point < len(self.points) - 1:
                current_point += 1
                current_distance += torch.norm(self.points[current_point] - self.points[current_point - 1]).item()
            
            # Linear interpolation
            alpha = (target_distance - (current_distance - torch.norm(self.points[current_point] - self.points[current_point - 1]).item())) / \
                   torch.norm(self.points[current_point] - self.points[current_point - 1]).item()
            interpolated_point = (1 - alpha) * self.points[current_point - 1] + alpha * self.points[current_point]
            new_points.append(interpolated_point)
        
        new_points.append(self.points[-1])
        return MagneticFieldLine(points=new_points, strength=self.strength) 