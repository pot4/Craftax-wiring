from dataclasses import dataclass
from typing import Tuple, Any

import jax.random
from flax import struct
import jax.numpy as jnp


# @struct.dataclass
# class Inventory: # experimental setup
#     wood: int = 9
#     stone: int = 9
#     coal: int = 0
#     iron: int = 0
#     diamond: int = 0
#     sapling: int = 0
#     wood_pickaxe: int = 1
#     stone_pickaxe: int = 1
#     iron_pickaxe: int = 1
#     wood_sword: int = 0
#     stone_sword: int = 0
#     iron_sword: int = 1
#     wire: int = 8
#     power: int = 0
#     extender: int = 0
#     junction: int = 9
#     AND: int = 1
#     OR: int = 0
#     XOR: int = 1
#     NOT: int = 0
#     pressure_plate: int = 0
#     input: int = 2
#     output: int = 2

@struct.dataclass
class Inventory:
    wood: int = 0
    stone: int = 0
    coal: int = 0
    iron: int = 0
    diamond: int = 0
    sapling: int = 0
    wood_pickaxe: int = 0
    stone_pickaxe: int = 0
    iron_pickaxe: int = 0
    wood_sword: int = 0
    stone_sword: int = 0
    iron_sword: int = 0
    wire: int = 0
    power: int = 0
    extender: int = 0
    junction: int = 0
    AND: int = 0
    OR: int = 0
    XOR: int = 0
    NOT: int = 0
    pressure_plate: int = 0
    input: int = 0
    output: int = 0

@struct.dataclass
class Mobs:
    position: jnp.ndarray
    health: int
    mask: bool
    attack_cooldown: int


@struct.dataclass
class EnvState:
    map: jnp.ndarray
    mob_map: jnp.ndarray

    player_position: jnp.ndarray
    player_direction: int

    # Intrinsics
    player_health: int
    player_food: int
    player_drink: int
    player_energy: int
    is_sleeping: bool

    # Second order intrinsics
    player_recover: float
    player_hunger: float
    player_thirst: float
    player_fatigue: float

    inventory: Inventory

    zombies: Mobs
    cows: Mobs
    skeletons: Mobs
    arrows: Mobs
    arrow_directions: jnp.ndarray

    growing_plants_positions: jnp.ndarray
    growing_plants_age: jnp.ndarray
    growing_plants_mask: jnp.ndarray

    wires_positions: jnp.ndarray
    wires_charge: jnp.ndarray
    wires_mask: jnp.ndarray

    extenders_positions: jnp.ndarray
    extenders_direction: jnp.ndarray
    extenders_mask: jnp.ndarray

    logic_gates_positions: jnp.ndarray
    logic_gates_type: jnp.ndarray
    logic_gates_direction: jnp.ndarray
    logic_gates_power: jnp.ndarray
    logic_gates_mask: jnp.ndarray

    inputs_positions: jnp.ndarray
    inputs_mask: jnp.ndarray

    outputs_positions: jnp.ndarray
    outputs_mask: jnp.ndarray

    light_level: float

    achievements: jnp.ndarray
    # input_switched: int
    # old_outputs: jnp.ndarray
    truth_table_half_adder: jnp.ndarray
    truth_table_full_adder: jnp.ndarray
    truth_table_bin_to_gray: jnp.ndarray
    half_adder_curriculum_stage: int
    state_rng: Any

    timestep: int

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class EnvParams:
    max_timesteps: int = 10000
    day_length: int = 300

    always_diamond: bool = True

    zombie_health: int = 5
    cow_health: int = 3
    skeleton_health: int = 3

    mob_despawn_distance: int = 14

    spawn_cow_chance: float = 0.1
    spawn_zombie_base_chance: float = 0.02
    spawn_zombie_night_chance: float = 0.1
    spawn_skeleton_chance: float = 0.05

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class StaticEnvParams:
    map_size: Tuple[int, int] = (64, 64)

    # Mobs
    max_zombies: int = 3
    max_cows: int = 3
    max_growing_plants: int = 10
    max_skeletons: int = 2
    max_arrows: int = 3

    max_wires: int = 20

    max_wire_charge: int = 10

    max_extenders: int = 2

    max_logic_gates: int = 5

    max_inputs: int = 3
    max_outputs: int = 3