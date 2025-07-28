import os
import pathlib
from enum import Enum

import jax
import jax.numpy as jnp
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageEnhance
from craftax.environment_base.util import load_compressed_pickle, save_compressed_pickle

# GAME CONSTANTS
OBS_DIM = (7, 9)
MAX_OBS_DIM = max(OBS_DIM)
assert OBS_DIM[0] % 2 == 1 and OBS_DIM[1] % 2 == 1
BLOCK_PIXEL_SIZE_HUMAN = 64
BLOCK_PIXEL_SIZE_IMG = 16
BLOCK_PIXEL_SIZE_AGENT = 7
INVENTORY_OBS_HEIGHT = 3
TEXTURE_CACHE_FILE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "assets", "texture_cache_wiring.pbz2"
)

# ENUMS
class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16
    WIRE = 17
    POWER = 18
    EXTENDER = 19
    EXTENSION = 20
    JUNCTION = 21
    AND = 22
    OR = 23
    XOR = 24
    NOT = 25
    PRESSURE_PLATE = 26
    INPUT_OFF = 27
    INPUT_ON = 28
    OUTPUT = 29
    


class Action(Enum):
    NOOP = 0  # q
    LEFT = 1  # a
    RIGHT = 2  # d
    UP = 3  # w
    DOWN = 4  # s
    DO = 5  # space
    SLEEP = 6  # tab
    PLACE_STONE = 7  # r
    PLACE_TABLE = 7  # t
    PLACE_FURNACE = 9  # f
    PLACE_PLANT = 10  # p
    MAKE_WOOD_PICKAXE = 11  # 1
    MAKE_STONE_PICKAXE = 12  # 2
    MAKE_IRON_PICKAXE = 13  # 3
    MAKE_WOOD_SWORD = 14  # 4
    MAKE_STONE_SWORD = 15  # 5
    MAKE_IRON_SWORD = 16  # 6
    MAKE_WIRE = 8  # 7
    MAKE_POWER = 18  # 8
    MAKE_EXTENDER = 19 # e
    MAKE_JUNCTION = 20 # r
    MAKE_AND = 21 # t
    MAKE_OR = 22 # y
    MAKE_XOR = 23 # u
    MAKE_NOT = 24 # i
    MAKE_PRESSURE_PLATE = 25 # o
    MAKE_INPUT = 26 # [
    MAKE_OUTPUT = 27 # ]
    PLACE_WIRE = 8 # 9
    PLACE_POWER = 29 # 0
    PLACE_EXTENDER = 11 # lshift
    PLACE_JUNCTION = 9 # z
    PLACE_AND = 10 # x
    PLACE_OR = 14 # c
    PLACE_XOR = 11 # v
    PLACE_NOT = 16 # b
    PLACE_PRESSURE_PLATE = 17 # n
    PLACE_INPUT = 12 # ,
    PLACE_OUTPUT = 13 # .
    SWITCH_INPUT = 14 # /

# GAME MECHANICS
DIRECTIONS = jnp.concatenate(
    (
        jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32),
        jnp.zeros((11, 2), dtype=jnp.int32),
    ),
    axis=0,
)

CLOSE_BLOCKS = jnp.array(
    [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    dtype=jnp.int32,
)

# Can't walk through these
SOLID_BLOCKS = jnp.array(
    [
        BlockType.WATER.value,
        BlockType.STONE.value,
        BlockType.TREE.value,
        BlockType.COAL.value,
        BlockType.IRON.value,
        BlockType.DIAMOND.value,
        BlockType.CRAFTING_TABLE.value,
        BlockType.FURNACE.value,
        BlockType.PLANT.value,
        BlockType.RIPE_PLANT.value,
        BlockType.EXTENDER.value,
        BlockType.EXTENSION.value,
        BlockType.INPUT_OFF.value,
        BlockType.INPUT_ON.value,
        BlockType.OUTPUT.value, 
    ],
    dtype=jnp.int32,
)

WIRE_BLOCKS = jnp.array(
    [
        BlockType.WIRE.value,
        BlockType.POWER.value,
        BlockType.JUNCTION.value,
        BlockType.AND.value,
        BlockType.OR.value,
        BlockType.XOR.value,
        BlockType.NOT.value,
        BlockType.PRESSURE_PLATE.value,
    ],
    dtype=jnp.int32,
)

# ACHIEVEMENTS
class Achievement(Enum):
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    MAKE_WOOD_PICKAXE = 5
    MAKE_WOOD_SWORD = 6
    PLACE_PLANT = 7
    DEFEAT_ZOMBIE = 8
    COLLECT_STONE = 9
    PLACE_STONE = 10
    EAT_PLANT = 11
    DEFEAT_SKELETON = 12
    MAKE_STONE_PICKAXE = 13
    MAKE_STONE_SWORD = 14
    WAKE_UP = 15
    PLACE_FURNACE = 16
    COLLECT_COAL = 17
    COLLECT_IRON = 18
    COLLECT_DIAMOND = 19
    MAKE_IRON_PICKAXE = 20
    MAKE_IRON_SWORD = 21
    MAKE_WIRE = 22
    PLACE_WIRE = 23
    MAKE_INPUT = 24
    PLACE_INPUT = 25
    MAKE_OUTPUT = 26
    PLACE_OUTPUT = 27
    ACTIVATE_INPUT = 28
    ACTIVATE_OUTPUT = 29
    ACTIVATE_LOGIC_GATE = 30
    HALF_ADDER = 31
    FULL_ADDER = 32
    BIN_TO_GRAY = 33
    TRAP = 34
    DOOR = 35
    STAGE0 = 36
    STAGE1 = 37
    STAGE2 = 38
    STAGE3 = 39
    STAGE4 = 40
    STAGE5 = 41
    STAGE6 = 42
    STAGE7 = 43
    STAGE8 = 44
    STAGE9 = 45
    STAGE10 = 46
    
    

INTERMEDIATE_ACHIEVEMENTS = [
    Achievement.ACTIVATE_OUTPUT.value,
    Achievement.ACTIVATE_LOGIC_GATE.value,
    Achievement.TRAP.value,
]

VERY_ADVANCED_ACHIEVEMENTS = [
    Achievement.DOOR.value,
    Achievement.HALF_ADDER.value,
    Achievement.FULL_ADDER.value,
    Achievement.BIN_TO_GRAY.value,
]

DISABLED_ACHIEVEMETS = [
    # Achievement.COLLECT_WOOD.value,
    # Achievement.PLACE_TABLE.value,
    # Achievement.EAT_COW.value,
    # Achievement.COLLECT_SAPLING.value,
    # Achievement.COLLECT_DRINK.value,
    # Achievement.MAKE_WOOD_PICKAXE.value,
    # Achievement.MAKE_WOOD_SWORD.value,
    # Achievement.PLACE_PLANT.value,
    # Achievement.DEFEAT_ZOMBIE.value,
    # Achievement.COLLECT_STONE.value,
    # Achievement.PLACE_STONE.value,
    # Achievement.EAT_PLANT.value,
    # Achievement.DEFEAT_SKELETON.value,
    # Achievement.MAKE_STONE_PICKAXE.value,
    # Achievement.MAKE_STONE_SWORD.value,
    # Achievement.WAKE_UP.value,
    # Achievement.PLACE_FURNACE.value,
    # Achievement.COLLECT_COAL.value,
    # Achievement.COLLECT_IRON.value,
    # Achievement.COLLECT_DIAMOND.value,
    # Achievement.MAKE_IRON_PICKAXE.value,
    # Achievement.MAKE_IRON_SWORD.value,
    # Achievement.MAKE_WIRE.value,
    # Achievement.MAKE_INPUT.value,
    Achievement.STAGE0.value,
    Achievement.STAGE1.value,
    Achievement.STAGE2.value,
    Achievement.STAGE3.value,
    Achievement.STAGE4.value,
    Achievement.STAGE5.value,
    Achievement.STAGE6.value,
    Achievement.STAGE7.value,
    Achievement.STAGE8.value,
    Achievement.STAGE9.value,
    Achievement.STAGE10.value,
]

def achievement_mapping(achievement_value):
    if achievement_value in INTERMEDIATE_ACHIEVEMENTS:
        return 3
    elif achievement_value in VERY_ADVANCED_ACHIEVEMENTS:
        return 8
    elif achievement_value in DISABLED_ACHIEVEMETS:
        return 0
    else:
        return 1


ACHIEVEMENT_REWARD_MAP = jnp.array(
    [achievement_mapping(i) for i in range(len(Achievement))]
)

# TEXTURES
def load_texture(filename, block_pixel_size, clamp_alpha=True):
    filename = os.path.join(pathlib.Path(__file__).parent.resolve(), "assets", filename)
    img = iio.imread(filename)
    jnp_img = jnp.array(img).astype(int)
    assert jnp_img.shape[:2] == (16, 16)

    if jnp_img.shape[2] == 4 and clamp_alpha:
        jnp_img = jnp_img.at[:, :, 3].set(jnp_img[:, :, 3] // 255)

    if block_pixel_size != 16:
        img = np.array(jnp_img, dtype=np.uint8)
        image = Image.fromarray(img)
        image = image.resize(
            (block_pixel_size, block_pixel_size), resample=Image.NEAREST
        )
        jnp_img = jnp.array(image, dtype=jnp.int32)

    return jnp_img


def load_all_textures(block_pixel_size):
    small_block_pixel_size = int(block_pixel_size * 0.8)

    # blocks
    texture_names = [
        "debug_tile.png",
        "debug_tile.png",
        "grass.png",
        "water.png",
        "stone.png",
        "tree.png",
        "wood.png",
        "path.png",
        "coal.png",
        "iron.png",
        "diamond.png",
        "table.png",
        "furnace.png",
        "sand.png",
        "lava.png",
        "plant_on_grass.png",
        "ripe_plant_on_grass.png",
        "wire.png",
        "power.png",
        "extender.png",
        "extension.png",
        "junction.png",
        "AND.png",
        "OR.png",
        "XOR.png",
        "NOT.png",
        "pressure_plate.png",
        "input_off.png",
        "input_on.png",
        "output.png",
    ]

    block_textures = jnp.array(
        [
            load_texture("debug_tile.png", block_pixel_size),
            jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32) * 128,
            load_texture("grass.png", block_pixel_size),
            load_texture("water.png", block_pixel_size),
            load_texture("stone.png", block_pixel_size),
            load_texture("tree.png", block_pixel_size),
            load_texture("wood.png", block_pixel_size)[:, :, :3],
            load_texture("path.png", block_pixel_size)[:, :, :3],
            load_texture("coal.png", block_pixel_size)[:, :, :3],
            load_texture("iron.png", block_pixel_size)[:, :, :3],
            load_texture("diamond.png", block_pixel_size)[:, :, :3],
            load_texture("table.png", block_pixel_size)[:, :, :3],
            load_texture("furnace.png", block_pixel_size)[:, :, :3],
            load_texture("sand.png", block_pixel_size)[:, :, :3],
            load_texture("lava.png", block_pixel_size)[:, :, :3],
            load_texture("plant_on_grass.png", block_pixel_size)[:, :, :3],
            load_texture("ripe_plant_on_grass.png", block_pixel_size)[:, :, :3],
            load_texture("wire.png", block_pixel_size)[:, :, :3],
            load_texture("power.png", block_pixel_size)[:, :, :3],
            load_texture("extender.png", block_pixel_size)[:, :, :3],
            load_texture("extension.png", block_pixel_size)[:, :, :3],
            load_texture("junction.png", block_pixel_size)[:, :, :3],
            load_texture("AND.png", block_pixel_size)[:, :, :3],
            load_texture("OR.png", block_pixel_size)[:, :, :3],
            load_texture("XOR.png", block_pixel_size)[:, :, :3],
            load_texture("NOT.png", block_pixel_size)[:, :, :3],
            load_texture("pressure_plate.png", block_pixel_size)[:, :, :3],
            load_texture("input_off.png", block_pixel_size)[:, :, :3],
            load_texture("input_on.png", block_pixel_size)[:, :, :3],
            load_texture("output.png", block_pixel_size)[:, :, :3],
        ]
    )

    block_textures = jnp.array(
        [load_texture(fname, block_pixel_size)[:, :, :3] for fname in texture_names]
    )
    block_textures = block_textures.at[1].set(
        jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32) * 128
    )

    # rng = jax.random.prngkey(0)
    # block_textures = jax.random.permutation(rng, block_textures)

    smaller_block_textures = jnp.array(
        [
            load_texture(fname, int(block_pixel_size * 0.8))[:, :, :3]
            for fname in texture_names
        ]
    )

    full_map_block_textures = jnp.array(
        [jnp.tile(block_textures[block.value], (*OBS_DIM, 1)) for block in BlockType]
    )

    # player
    pad_pixels = (
        (OBS_DIM[0] // 2) * block_pixel_size,
        (OBS_DIM[1] // 2) * block_pixel_size,
    )

    player_textures = [
        load_texture("player-left.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-right.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-up.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-down.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-sleep.png", block_pixel_size, clamp_alpha=False),
    ]

    full_map_player_textures_rgba = [
        jnp.pad(
            player_texture,
            ((pad_pixels[0], pad_pixels[0]), (pad_pixels[1], pad_pixels[1]), (0, 0)),
        )
        for player_texture in player_textures
    ]

    full_map_player_textures = jnp.array(
        [player_texture[:, :, :3] for player_texture in full_map_player_textures_rgba]
    )

    full_map_player_textures_alpha = jnp.array(
        [
            jnp.repeat(
                jnp.expand_dims(player_texture[:, :, 3], axis=-1).astype(float) / 255,
                repeats=3,
                axis=2,
            )
            for player_texture in full_map_player_textures_rgba
        ]
    )

    # inventory

    empty_texture = jnp.zeros((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)
    smaller_empty_texture = jnp.zeros(
        (int(block_pixel_size * 0.8), int(block_pixel_size * 0.8), 3), dtype=jnp.int32
    )

    ones_texture = jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)

    number_size = int(block_pixel_size * 0.6)

    number_textures_rgba = [
        jnp.zeros((number_size, number_size, 3), dtype=jnp.int32),
        load_texture("1.png", number_size),
        load_texture("2.png", number_size),
        load_texture("3.png", number_size),
        load_texture("4.png", number_size),
        load_texture("5.png", number_size),
        load_texture("6.png", number_size),
        load_texture("7.png", number_size),
        load_texture("8.png", number_size),
        load_texture("9.png", number_size),
    ]

    number_textures = jnp.array(
        [
            number_texture[:, :, :3]
            * jnp.repeat(jnp.expand_dims(number_texture[:, :, 3], axis=-1), 3, axis=-1)
            for number_texture in number_textures_rgba
        ]
    )

    number_textures_alpha = jnp.array(
        [
            jnp.repeat(
                jnp.expand_dims(number_texture[:, :, 3], axis=-1), repeats=3, axis=2
            )
            for number_texture in number_textures_rgba
        ]
    )

    health_texture = jnp.array(
        load_texture("health.png", small_block_pixel_size)[:, :, :3]
    )
    hunger_texture = jnp.array(
        load_texture("food.png", small_block_pixel_size)[:, :, :3]
    )
    thirst_texture = jnp.array(
        load_texture("drink.png", small_block_pixel_size)[:, :, :3]
    )
    energy_texture = jnp.array(
        load_texture("energy.png", small_block_pixel_size)[:, :, :3]
    )

    # get rid of the cow ghost
    def apply_alpha(texture):
        return texture[:, :, :3] * jnp.repeat(
            jnp.expand_dims(texture[:, :, 3], axis=-1), 3, axis=-1
        )

    wood_pickaxe_texture = jnp.array(
        load_texture("wood_pickaxe.png", small_block_pixel_size)[:, :, :3]
    )  # no ghosts :)
    stone_pickaxe_texture = jnp.array(
        load_texture("stone_pickaxe.png", small_block_pixel_size)
    )
    stone_pickaxe_texture = apply_alpha(stone_pickaxe_texture)
    iron_pickaxe_texture = jnp.array(
        load_texture("iron_pickaxe.png", small_block_pixel_size)
    )
    iron_pickaxe_texture = apply_alpha(iron_pickaxe_texture)

    wood_sword_texture = jnp.array(
        load_texture("wood_sword.png", small_block_pixel_size)
    )
    wood_sword_texture = apply_alpha(wood_sword_texture)
    stone_sword_texture = jnp.array(
        load_texture("stone_sword.png", small_block_pixel_size)
    )
    stone_sword_texture = apply_alpha(stone_sword_texture)
    iron_sword_texture = jnp.array(
        load_texture("iron_sword.png", small_block_pixel_size)
    )
    iron_sword_texture = apply_alpha(iron_sword_texture)

    sapling_texture = jnp.array(
        load_texture("sapling.png", small_block_pixel_size)[:, :, :3]
    )

    wire_texture = jnp.array(
        load_texture("wire.png", small_block_pixel_size)
    )
    wire_texture = apply_alpha(wire_texture)
    
    power_texture = jnp.array(
        load_texture("power.png", small_block_pixel_size)
    )
    power_texture = apply_alpha(power_texture)
     
    extender_texture = jnp.array(
        load_texture("extender.png", small_block_pixel_size)
    )
    extender_texture = apply_alpha(extender_texture)

    extension_texture = jnp.array(
        load_texture("extension.png", small_block_pixel_size)
    )
    extension_texture = apply_alpha(extension_texture)
     
    junction_texture = jnp.array(
        load_texture("junction.png", small_block_pixel_size)
    )
    junction_texture = apply_alpha(junction_texture)
     
    AND_texture = jnp.array(
        load_texture("AND.png", small_block_pixel_size)
    )
    AND_texture = apply_alpha(AND_texture)
     
    OR_texture = jnp.array(
        load_texture("OR.png", small_block_pixel_size)
    )
    OR_texture = apply_alpha(OR_texture)
     
    XOR_texture = jnp.array(
        load_texture("XOR.png", small_block_pixel_size)
    )
    XOR_texture = apply_alpha(XOR_texture)
     
    NOT_texture = jnp.array(
        load_texture("NOT.png", small_block_pixel_size)
    )
    NOT_texture = apply_alpha(NOT_texture)
     
    pressure_plate_texture = jnp.array(
        load_texture("pressure_plate.png", small_block_pixel_size)
    )
    pressure_plate_texture = apply_alpha(pressure_plate_texture)
     
    input_off_texture = jnp.array(
        load_texture("input_off.png", small_block_pixel_size)
    )
    input_off_texture = apply_alpha(input_off_texture)

    input_on_texture = jnp.array(
        load_texture("input_on.png", small_block_pixel_size)
    )
    input_on_texture = apply_alpha(input_on_texture)

    output_texture = jnp.array(
        load_texture("output.png", small_block_pixel_size)
    )
    output_texture = apply_alpha(output_texture)

    # entities
    zombie_texture_rgba = jnp.array(
        load_texture("zombie.png", block_pixel_size, clamp_alpha=False)
    )
    zombie_texture = zombie_texture_rgba[:, :, :3]
    zombie_texture_alpha = jnp.repeat(
        jnp.expand_dims(zombie_texture_rgba[:, :, 3], axis=-1).astype(float) / 255,
        repeats=3,
        axis=2,
    )

    cow_texture_rgba = jnp.array(
        load_texture("cow.png", block_pixel_size, clamp_alpha=False)
    )
    cow_texture = cow_texture_rgba[:, :, :3]
    cow_texture_alpha = jnp.repeat(
        jnp.expand_dims(cow_texture_rgba[:, :, 3], axis=-1).astype(float) / 255,
        repeats=3,
        axis=2,
    )

    skeleton_texture_rgba = jnp.array(
        load_texture("skeleton.png", block_pixel_size, clamp_alpha=False)
    )
    skeleton_texture = skeleton_texture_rgba[:, :, :3]
    skeleton_texture_alpha = jnp.repeat(
        jnp.expand_dims(skeleton_texture_rgba[:, :, 3], axis=-1).astype(float) / 255,
        repeats=3,
        axis=2,
    )

    arrow_texture_rgba = jnp.array(load_texture("arrow-up.png", block_pixel_size))
    arrow_texture = apply_alpha(arrow_texture_rgba)
    arrow_texture_alpha = jnp.repeat(
        jnp.expand_dims(arrow_texture_rgba[:, :, 3], axis=-1), repeats=3, axis=2
    )

    night_texture = (
        jnp.array([[[0, 16, 64]]])
        .repeat(OBS_DIM[0] * block_pixel_size, axis=0)
        .repeat(OBS_DIM[1] * block_pixel_size, axis=1)
    )

    xs, ys = np.meshgrid(
        np.linspace(-1, 1, OBS_DIM[0] * block_pixel_size),
        np.linspace(-1, 1, OBS_DIM[1] * block_pixel_size),
    )
    night_noise_intensity_texture = (
        1 - np.exp(-0.5 * (xs**2 + ys**2) / (0.5**2)).T
    )

    night_noise_intensity_texture = jnp.expand_dims(
        night_noise_intensity_texture, axis=-1
    ).repeat(3, axis=-1)

    return {
        "block_textures": block_textures,
        "smaller_block_textures": smaller_block_textures,
        "full_map_block_textures": full_map_block_textures,
        "player_textures": player_textures,
        "full_map_player_textures": full_map_player_textures,
        "full_map_player_textures_alpha": full_map_player_textures_alpha,
        "empty_texture": empty_texture,
        "smaller_empty_texture": smaller_empty_texture,
        "ones_texture": ones_texture,
        "number_textures": number_textures,
        "number_textures_alpha": number_textures_alpha,
        "health_texture": health_texture,
        "hunger_texture": hunger_texture,
        "thirst_texture": thirst_texture,
        "energy_texture": energy_texture,
        "wood_pickaxe_texture": wood_pickaxe_texture,
        "stone_pickaxe_texture": stone_pickaxe_texture,
        "iron_pickaxe_texture": iron_pickaxe_texture,
        "wood_sword_texture": wood_sword_texture,
        "stone_sword_texture": stone_sword_texture,
        "iron_sword_texture": iron_sword_texture,
        "sapling_texture": sapling_texture,
        "wire_texture": wire_texture,
        "power_texture": power_texture,
        "extender_texture": extender_texture,
        "extension_texture": extension_texture,
        "junction_texture": junction_texture,
        "AND_texture": AND_texture,
        "OR_texture": OR_texture,
        "XOR_texture": XOR_texture,
        "NOT_texture": NOT_texture,
        "pressure_plate_texture": pressure_plate_texture,
        "input_off_texture": input_off_texture,
        "input_on_texture": input_on_texture,
        "output_texture": output_texture,
        "zombie_texture": zombie_texture,
        "zombie_texture_alpha": zombie_texture_alpha,
        "cow_texture": cow_texture,
        "cow_texture_alpha": cow_texture_alpha,
        "skeleton_texture": skeleton_texture,
        "skeleton_texture_alpha": skeleton_texture_alpha,
        "arrow_texture": arrow_texture,
        "arrow_texture_alpha": arrow_texture_alpha,
        "night_texture": night_texture,
        "night_noise_intensity_texture": night_noise_intensity_texture,
    }


load_cached_textures_success = True
if os.path.exists(TEXTURE_CACHE_FILE) and not os.environ.get(
    "CRAFTAX_RELOAD_TEXTURES", False
):
    print("Loading Craftax-Wiring textures from cache.")
    TEXTURES = load_compressed_pickle(TEXTURE_CACHE_FILE)
    # Check validity of texture cache
    for ts in (BLOCK_PIXEL_SIZE_AGENT, BLOCK_PIXEL_SIZE_IMG, BLOCK_PIXEL_SIZE_HUMAN):
        tex_shape = TEXTURES[ts]["full_map_block_textures"].shape
        if (
            tex_shape[0] != len(BlockType)
            or tex_shape[1] != OBS_DIM[0] * ts
            or tex_shape[2] != OBS_DIM[1] * ts
            or tex_shape[3] != 3
        ):
            load_cached_textures_success = False
            print("Invalid texture cache, going to reload textures.")
            break
    print("Textures successfully loaded from cache.")
else:
    load_cached_textures_success = False

if not load_cached_textures_success:
    print(
        "Processing Craftax-Wiring textures. This will take a minute but will be cached for future use."
    )
    TEXTURES = {
        BLOCK_PIXEL_SIZE_AGENT: load_all_textures(BLOCK_PIXEL_SIZE_AGENT),
        BLOCK_PIXEL_SIZE_IMG: load_all_textures(BLOCK_PIXEL_SIZE_IMG),
        BLOCK_PIXEL_SIZE_HUMAN: load_all_textures(BLOCK_PIXEL_SIZE_HUMAN),
    }

    save_compressed_pickle(TEXTURE_CACHE_FILE, TEXTURES)
    print("Textures loaded and saved to cache.")
