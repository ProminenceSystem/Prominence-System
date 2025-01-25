"""
Prominence System Configuration
Core configuration settings for the HELIOS system and magnetic field parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class MagneticFieldConfig:
    field_strength: int = 1000
    polarity: str = "north"
    diffusivity: float = 0.001
    reynolds_number: int = 1000
    alfven_speed: float = 0.1

@dataclass
class PlasmaConfig:
    temperature: float = 1e6  # Kelvin
    density: float = 1e12     # particles/cm^3
    pressure: float = 1.0     # normalized units
    conductivity: float = 0.9 # normalized units

@dataclass
class HELIOSConfig:
    model_version: str = "1.0.0"
    attention_heads: int = 8
    hidden_dim: int = 512
    num_layers: int = 6
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_sequence_length: int = 1024

@dataclass
class BlockchainConfig:
    network: str = "solana-mainnet"
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    commitment: str = "confirmed"
    max_retries: int = 3
    timeout: int = 30

@dataclass
class ProminenceConfig:
    magnetic_field: MagneticFieldConfig = MagneticFieldConfig()
    plasma: PlasmaConfig = PlasmaConfig()
    helios: HELIOSConfig = HELIOSConfig()
    blockchain: BlockchainConfig = BlockchainConfig()
    debug: bool = False
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict:
        return {
            "magnetic_field": self.magnetic_field.__dict__,
            "plasma": self.plasma.__dict__,
            "helios": self.helios.__dict__,
            "blockchain": self.blockchain.__dict__,
            "debug": self.debug,
            "log_level": self.log_level
        }

# Default configuration instance
default_config = ProminenceConfig() 