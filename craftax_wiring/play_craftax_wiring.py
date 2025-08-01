import argparse
import sys

import pygame

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_wiring.constants import (
    OBS_DIM,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement,
    BLOCK_PIXEL_SIZE_HUMAN,
)
from craftax.craftax_wiring.envs.craftax_symbolic_env import (
    CraftaxWiringSymbolicEnv as CraftaxEnv,
)
from craftax.craftax_wiring.renderer import render_craftax_pixels
from craftax.craftax_env import make_craftax_env_from_name

from craftax.craftax_wiring.wire_logic import *

# KEY_MAPPING = {
#     pygame.K_q: Action.NOOP,
#     pygame.K_w: Action.UP,
#     pygame.K_d: Action.RIGHT,
#     pygame.K_s: Action.DOWN,
#     pygame.K_a: Action.LEFT,
#     pygame.K_SPACE: Action.DO,
#     pygame.K_LCTRL: Action.PLACE_TABLE,
#     pygame.K_TAB: Action.SLEEP,
#     pygame.K_9: Action.PLACE_WIRE,
#     pygame.K_z: Action.PLACE_JUNCTION,
#     pygame.K_x: Action.PLACE_AND,
#     pygame.K_v: Action.PLACE_XOR,
#     pygame.K_COMMA: Action.PLACE_INPUT,
#     pygame.K_PERIOD: Action.PLACE_OUTPUT,
#     pygame.K_SLASH: Action.SWITCH_INPUT,
# }

KEY_MAPPING = {
    pygame.K_q: Action.NOOP,
    pygame.K_w: Action.UP,
    pygame.K_d: Action.RIGHT,
    pygame.K_s: Action.DOWN,
    pygame.K_a: Action.LEFT,
    pygame.K_SPACE: Action.DO,
    pygame.K_LCTRL: Action.PLACE_TABLE,
    pygame.K_TAB: Action.SLEEP,
    pygame.K_CAPSLOCK: Action.PLACE_STONE,
    pygame.K_f: Action.PLACE_FURNACE,
    pygame.K_p: Action.PLACE_PLANT,
    pygame.K_1: Action.MAKE_WOOD_PICKAXE,
    pygame.K_2: Action.MAKE_STONE_PICKAXE,
    pygame.K_3: Action.MAKE_IRON_PICKAXE,
    pygame.K_4: Action.MAKE_WOOD_SWORD,
    pygame.K_5: Action.MAKE_STONE_SWORD,
    pygame.K_6: Action.MAKE_IRON_SWORD,
    pygame.K_7: Action.MAKE_WIRE,
    pygame.K_8: Action.MAKE_POWER,
    pygame.K_e: Action.MAKE_EXTENDER,
    pygame.K_r: Action.MAKE_JUNCTION,
    pygame.K_t: Action.MAKE_AND,
    pygame.K_y: Action.MAKE_OR,
    pygame.K_u: Action.MAKE_XOR,
    pygame.K_i: Action.MAKE_NOT,
    pygame.K_o: Action.MAKE_PRESSURE_PLATE,
    pygame.K_LEFTBRACKET: Action.MAKE_INPUT,
    pygame.K_RIGHTBRACKET: Action.MAKE_OUTPUT,
    pygame.K_9: Action.PLACE_WIRE,
    pygame.K_0: Action.PLACE_POWER,
    pygame.K_LSHIFT: Action.PLACE_EXTENDER,
    pygame.K_z: Action.PLACE_JUNCTION,
    pygame.K_x: Action.PLACE_AND,
    pygame.K_c: Action.PLACE_OR,
    pygame.K_v: Action.PLACE_XOR,
    pygame.K_b: Action.PLACE_NOT,
    pygame.K_n: Action.PLACE_PRESSURE_PLATE,
    pygame.K_COMMA: Action.PLACE_INPUT,
    pygame.K_PERIOD: Action.PLACE_OUTPUT,
    pygame.K_SLASH: Action.SWITCH_INPUT,
}


class CraftaxRenderer:
    def __init__(self, env: CraftaxEnv, env_params, pixel_render_size=4):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size,
        )

        # Init rendering
        pygame.init()
        pygame.key.set_repeat(250, 75)

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self._render = jax.jit(render_craftax_pixels, static_argnums=(1,))

    def update(self):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Update screen
        pygame.display.flip()
        # time.sleep(0.01)

    def render(self, env_state):
        # Clear
        self.screen_surface.fill((0, 0, 0))

        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        surface = pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, 0))

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

    def get_action_from_keypress(self, state):
        if state.is_sleeping:
            return Action.NOOP.value
        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key in KEY_MAPPING:
                    return KEY_MAPPING[event.key].value

        return None


def print_new_achievements(old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(f"{Achievement(i).name} ({new_achievements.sum()}/{47})")


def main(args):
    env = make_craftax_env_from_name("Craftax-Wiring-Symbolic-v1", auto_reset=True)
    env_params = env.default_params

    print("Controls")
    for k, v in KEY_MAPPING.items():
        print(f"{pygame.key.name(k)}: {v.name.lower()}")

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    _, env_state = env.reset(_rng, env_params)

    pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN

    renderer = CraftaxRenderer(env, env_params, pixel_render_size=pixel_render_size)
    renderer.render(env_state)

    step_fn = jax.jit(env.step)

    clock = pygame.time.Clock()

    while not renderer.is_quit_requested():
        action = renderer.get_action_from_keypress(env_state)

        if action is not None:
            rng, _rng = jax.random.split(rng)
            old_achievements = env_state.achievements
            obs, env_state, reward, done, info = step_fn(
                _rng, env_state, action, env_params
            )
            new_achievements = env_state.achievements
            print_new_achievements(old_achievements, new_achievements)

            if reward > 0.01 or reward < -0.01:
                print(f"Reward: {reward}\n")

            renderer.render(env_state)

        renderer.update()
        clock.tick(args.fps)


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fps", type=int, default=60)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)


if __name__ == "__main__":
    entry_point()
