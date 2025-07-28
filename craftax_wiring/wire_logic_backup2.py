from craftax.craftax_classic.constants import *
from craftax.craftax_classic.envs.craftax_state import *
from jax import checkpoint

def in_bounds(state, position):
    in_bounds_x = jnp.logical_and(0 <= position[0], position[0] < state.map.shape[0])
    in_bounds_y = jnp.logical_and(0 <= position[1], position[1] < state.map.shape[1])
    in_bounds = jnp.logical_and(in_bounds_x, in_bounds_y)
    # jax.debug.print("in_bounds: {}", position)
    return jax.lax.select(in_bounds, position, jnp.array([0, 0]))




def add_new_wire(state, position, is_placing_wire, static_params):
    def _is_empty(unused, index):
        return None, jnp.logical_not(state.wires_mask[index])

    _, is_empty = jax.lax.scan(
        _is_empty, None, jnp.arange(static_params.max_wires)
    )

    wire_index = jnp.argmax(is_empty)
    is_an_empty_slot = is_empty.sum() > 0

    is_adding_wire = jnp.logical_and(is_an_empty_slot, is_placing_wire)

    new_wires_positions = jax.lax.select(
        is_adding_wire,
        state.wires_positions.at[wire_index].set(position),
        state.wires_positions,
    )
    new_wires_charge = jax.lax.select(
        is_adding_wire,
        state.wires_charge.at[wire_index].set(0),
        state.wires_charge,
    )
    new_wires_mask = jax.lax.select(
        is_adding_wire,
        state.wires_mask.at[wire_index].set(True),
        state.wires_mask,
    )

    return new_wires_positions, new_wires_charge, new_wires_mask

def remove_wire(state, position, is_mining_wire, static_params):

    wire_index = jnp.argmax(jnp.all(state.wires_positions == position, axis=1))

    new_wires_positions = jax.lax.select(
        is_mining_wire,
        state.wires_positions.at[wire_index].set([-100, -100]),
        state.wires_positions,
    )
    new_wires_charge = jax.lax.select(
        is_mining_wire,
        state.wires_charge.at[wire_index].set(0),
        state.wires_charge,
    )
    new_wires_mask = jax.lax.select(
        is_mining_wire,
        state.wires_mask.at[wire_index].set(False),
        state.wires_mask,
    )

    return new_wires_positions, new_wires_charge, new_wires_mask

def add_new_extender(state, position, is_placing_extender, static_params):
    def _is_empty(unused, index):
        return None, jnp.logical_not(state.extenders_mask[index])

    _, is_empty = jax.lax.scan(
        _is_empty, None, jnp.arange(static_params.max_extenders)
    )

    extender_index = jnp.argmax(is_empty)
    is_an_empty_slot = is_empty.sum() > 0

    is_adding_extender = jnp.logical_and(is_an_empty_slot, is_placing_extender)

    new_extenders_positions = jax.lax.select(
        is_adding_extender,
        state.extenders_positions.at[extender_index].set(position),
        state.extenders_positions,
    )
    new_extenders_direction = jax.lax.select(
        is_adding_extender,
        state.extenders_direction.at[extender_index].set(state.player_direction),
        state.extenders_direction,
    )
    new_extenders_mask = jax.lax.select(
        is_adding_extender,
        state.extenders_mask.at[extender_index].set(True),
        state.extenders_mask,
    )

    return new_extenders_positions, new_extenders_direction, new_extenders_mask

def remove_extender(state, position, is_mining_extender, static_params):
    extender_index = jnp.argmax(jnp.all(state.logic_gates_positions == position, axis=1))

    new_extenders_positions = jax.lax.select(
        is_mining_extender,
        state.extenders_positions.at[extender_index].set([0, 0]),
        state.extenders_positions,
    )
    new_extenders_direction = jax.lax.select(
        is_mining_extender,
        state.extenders_direction.at[extender_index].set(state.player_direction),
        state.extenders_direction,
    )
    new_extenders_mask = jax.lax.select(
        is_mining_extender,
        state.extenders_mask.at[extender_index].set(False),
        state.extenders_mask,
    )

    return new_extenders_positions, new_extenders_direction, new_extenders_mask

def add_new_logic_gate(state, position, is_placing_logic_gate, static_params, type):
    def _is_empty(unused, index):
        return None, jnp.logical_not(state.logic_gates_mask[index])

    _, is_empty = jax.lax.scan(
        _is_empty, None, jnp.arange(static_params.max_logic_gates)
    )

    logic_gate_index = jnp.argmax(is_empty)
    is_an_empty_slot = is_empty.sum() > 0

    is_adding_logic_gate = jnp.logical_and(is_an_empty_slot, is_placing_logic_gate)

    new_logic_gates_positions = jax.lax.select(
        is_adding_logic_gate,
        state.logic_gates_positions.at[logic_gate_index].set(position),
        state.logic_gates_positions,
    )
    new_logic_gates_type = jax.lax.select(
        is_adding_logic_gate,
        state.logic_gates_type.at[logic_gate_index].set(type),
        state.logic_gates_type,
    )
    new_logic_gates_direction = jax.lax.select(
        is_adding_logic_gate,
        state.logic_gates_direction.at[logic_gate_index].set(state.player_direction),
        state.logic_gates_direction,
    )
    new_logic_gates_power = jax.lax.select(
        is_adding_logic_gate,
        state.logic_gates_power.at[logic_gate_index].set(False),
        state.logic_gates_power,
    )
    new_logic_gates_mask = jax.lax.select(
        is_adding_logic_gate,
        state.logic_gates_mask.at[logic_gate_index].set(True),
        state.logic_gates_mask,
    )

    return new_logic_gates_positions, new_logic_gates_type, new_logic_gates_direction, new_logic_gates_power, new_logic_gates_mask

def remove_logic_gate(state, position, is_mining_logic_gate, static_params):
    logic_gate_index = jnp.argmax(jnp.all(state.logic_gates_positions == position, axis=1))

    new_logic_gates_positions = jax.lax.select(
        is_mining_logic_gate,
        state.logic_gates_positions.at[logic_gate_index].set([0, 0]),
        state.logic_gates_positions,
    )
    new_logic_gates_type = jax.lax.select(
        is_mining_logic_gate,
        state.logic_gates_type.at[logic_gate_index].set(-1),
        state.logic_gates_type,
    )
    new_logic_gates_direction = jax.lax.select(
        is_mining_logic_gate,
        state.logic_gates_direction.at[logic_gate_index].set(state.player_direction),
        state.logic_gates_direction,
    )
    new_logic_gates_power = jax.lax.select(
        is_mining_logic_gate,
        state.logic_gates_power.at[logic_gate_index].set(False),
        state.logic_gates_power,
    )
    new_logic_gates_mask = jax.lax.select(
        is_mining_logic_gate,
        state.logic_gates_mask.at[logic_gate_index].set(False),
        state.logic_gates_mask,
    )

    return new_logic_gates_positions, new_logic_gates_type, new_logic_gates_direction, new_logic_gates_power, new_logic_gates_mask

def add_new_input(state, position, is_placing_input, static_params):
    def _is_empty(unused, index):
        return None, jnp.logical_not(state.inputs_mask[index])

    _, is_empty = jax.lax.scan(
        _is_empty, None, jnp.arange(static_params.max_inputs)
    )

    input_index = jnp.argmax(is_empty)
    is_an_empty_slot = is_empty.sum() > 0

    is_adding_input = jnp.logical_and(is_an_empty_slot, is_placing_input)

    new_inputs_positions = jax.lax.select(
        is_adding_input,
        state.inputs_positions.at[input_index].set(position),
        state.inputs_positions,
    )
    new_inputs_mask = jax.lax.select(
        is_adding_input,
        state.inputs_mask.at[input_index].set(True),
        state.inputs_mask,
    )

    return new_inputs_positions, new_inputs_mask

# def update_input(state, position, is_switching_input):
#     input_index = jnp.argmax(jnp.all(state.inputs_positions == position, axis=1))
#     new_inputs_charge = jax.lax.select(
#         is_switching_input,
#         state.inputs_charge.at[input_index].set(state.map[position]),
#         state.inputs_charge,
#     )
#     return new_inputs_charge

def retrieve_input(state, index):
    input_position = state.inputs_positions[index]
    is_on = state.map[input_position[0], input_position[1]] == BlockType.INPUT_ON.value
    return state, is_on

def remove_input(state, position, is_mining_input, static_params):

    input_index = jnp.argmax(jnp.all(state.inputs_positions == position, axis=1))

    new_inputs_positions = jax.lax.select(
        is_mining_input,
        state.inputs_positions.at[input_index].set([0, 0]),
        state.inputs_positions,
    )

    new_inputs_mask = jax.lax.select(
        is_mining_input,
        state.inputs_mask.at[input_index].set(False),
        state.inputs_mask,
    )

    return new_inputs_positions, new_inputs_mask

def add_new_output(state, position, is_placing_output, static_params):
    def _is_empty(unused, index):
        return None, jnp.logical_not(state.outputs_mask[index])

    _, is_empty = jax.lax.scan(
        _is_empty, None, jnp.arange(static_params.max_outputs)
    )

    output_index = jnp.argmax(is_empty)
    is_an_empty_slot = is_empty.sum() > 0

    is_adding_output = jnp.logical_and(is_an_empty_slot, is_placing_output)

    new_outputs_positions = jax.lax.select(
        is_adding_output,
        state.outputs_positions.at[output_index].set(position),
        state.outputs_positions,
    )
    new_outputs_mask = jax.lax.select(
        is_adding_output,
        state.outputs_mask.at[output_index].set(True),
        state.outputs_mask,
    )

    return new_outputs_positions, new_outputs_mask

def retrieve_output(state, index):
    output_position = state.outputs_positions[index]
    output_index = jnp.argmax(jnp.all(state.wires_positions == output_position, axis=1))
    output = state.wires_charge[output_index] > 0
    return state, output

def remove_output(state, position, is_mining_output, static_params):

    output_index = jnp.argmax(jnp.all(state.outputs_positions == position, axis=1))

    new_outputs_positions = jax.lax.select(
        is_mining_output,
        state.outputs_positions.at[output_index].set([0, 0]),
        state.outputs_positions,
    )
    new_outputs_mask = jax.lax.select(
        is_mining_output,
        state.outputs_mask.at[output_index].set(False),
        state.outputs_mask,
    )

    return new_outputs_positions, new_outputs_mask

@checkpoint
def update_wires(state, static_params):
    neighboring_wire_area = jnp.array([
        [-1, 0], [0, -1], [0, 1], [1, 0]
    ], dtype=jnp.int32)

    map_h, map_w = state.map.shape

    def in_bounds(pos):
        return jnp.array([
            jnp.clip(pos[0], 0, map_h - 1),
            jnp.clip(pos[1], 0, map_w - 1)
        ], dtype=jnp.int32)

    def safe_map_lookup(pos):
        safe_pos = in_bounds(pos)
        return state.map[safe_pos[0], safe_pos[1]]

    def get_wire_charge(pos):
        is_wire = safe_map_lookup(pos) == 4  # BlockType.WIRE
        wire_matches = jnp.all(state.wires_positions == pos, axis=1)
        wire_found = jnp.any(wire_matches)
        wire_index = jnp.argmax(wire_matches)
        return jax.lax.select(jnp.logical_and(is_wire, wire_found), state.wires_charge[wire_index], 0)

    def check_max_charge(wires_charge, index):
        wire_pos = state.wires_positions[index]
        nearby_charge = 0

        for i in range(neighboring_wire_area.shape[0]):
            loc_add = neighboring_wire_area[i]
            pos = wire_pos + loc_add

            # Follow junctions
            def cond_fn(carry):
                pos, count = carry
                return safe_map_lookup(pos) == 6  # BlockType.JUNCTION

            def body_fn(carry):
                pos, count = carry
                return pos + loc_add, count + 1

            pos, num_junctions = jax.lax.while_loop(cond_fn, body_fn, (pos, 0))

            map_val = safe_map_lookup(pos)
            is_power = jnp.logical_or(
                jnp.logical_or(
                    map_val == 2,  # BlockType.POWER
                    map_val == 8   # BlockType.INPUT_ON
                ),
                jnp.logical_and(
                    map_val == 14,  # BlockType.PRESSURE_PLATE
                    jnp.logical_or(
                        jnp.all(state.player_position == pos),
                        state.mob_map[pos[0], pos[1]]
                    )
                )
            )

            # Logic gate check (guarded)
            gate_matches = jnp.all(state.logic_gates_positions == pos, axis=1)
            gate_found = jnp.any(gate_matches)
            logic_gate_index = jnp.where(gate_found, jnp.argmax(gate_matches), 0)

            powered_gate = jax.lax.select(gate_found, state.logic_gates_power[logic_gate_index], False)
            gate_dir = jax.lax.select(gate_found, state.logic_gates_direction[logic_gate_index], 0)
            gate_pos = jax.lax.select(gate_found, state.logic_gates_positions[logic_gate_index], jnp.array([0, 0]))

            is_output = jnp.all(
                gate_pos + DIRECTIONS[gate_dir] * (1 + num_junctions) == wire_pos
            )
            receive_output = jnp.logical_and(powered_gate, is_output)
            source_nearby = jnp.logical_or(is_power, receive_output)

            new_charge = jax.lax.select(source_nearby, static_params.max_wire_charge, nearby_charge)
            decay_charge = jnp.maximum(0, get_wire_charge(pos) - 1)
            nearby_charge = jnp.maximum(new_charge, decay_charge)

        wires_charge = wires_charge.at[index].set(nearby_charge)
        return wires_charge

    def loop_body(i, wires_charge):
        return check_max_charge(wires_charge, i)

    wires_charge = jax.lax.fori_loop(0, static_params.max_wires, loop_body, state.wires_charge)

    return state.replace(wires_charge=wires_charge)

# def update_wires(state, static_params):
#     neighboring_wire_area = jnp.array([
#         [-1, 0], [0, -1], [0, 1], [1, 0]
#     ], dtype=jnp.int32)

#     def safe_get_charge(pos):
#         is_wire = state.map[pos[0], pos[1]] == BlockType.WIRE.value
#         wire_matches = jnp.all(state.wires_positions == pos, axis=1)
#         wire_found = jnp.any(wire_matches)
#         wire_index = jnp.argmax(wire_matches)
#         valid = jnp.logical_and(is_wire, wire_found)
#         return jax.lax.select(valid, state.wires_charge[wire_index], 0)


#     def check_max_charge(carry, loc_add):
#         wires_charge, nearby_charge, index = carry
#         pos = state.wires_positions[index] + loc_add
#         pos = in_bounds(state, pos)

#         # Step through junctions
#         def cond(carry):
#             pos, count = carry
#             return state.map[pos[0], pos[1]] == BlockType.JUNCTION.value

#         def body(carry):
#             pos, count = carry
#             return pos + loc_add, count + 1

#         pos, num_junctions = jax.lax.while_loop(cond, body, (pos, 0))

#         # Power source check
#         is_power = jnp.logical_or(
#             jnp.logical_or(
#                 state.map[pos[0], pos[1]] == BlockType.POWER.value,
#                 state.map[pos[0], pos[1]] == BlockType.INPUT_ON.value
#             ),
#             jnp.logical_and(
#                 state.map[pos[0], pos[1]] == BlockType.PRESSURE_PLATE.value,
#                 jnp.logical_or(
#                     jnp.all(state.player_position == pos),
#                     state.mob_map[pos[0], pos[1]]
#                 )
#             )
#         )

#         # Logic gate match
#         gate_matches = jnp.all(state.logic_gates_positions == pos, axis=1)
#         gate_found = jnp.any(gate_matches)
#         logic_gate_index = jnp.argmax(gate_matches)

#         powered_gate = jax.lax.select(
#             gate_found,
#             state.logic_gates_power[logic_gate_index],
#             False
#         )
#         gate_dir = jax.lax.select(
#             gate_found,
#             state.logic_gates_direction[logic_gate_index],
#             0
#         )
#         gate_pos = jax.lax.select(
#             gate_found,
#             state.logic_gates_positions[logic_gate_index],
#             jnp.array([0, 0], dtype=jnp.int32)
#         )

#         is_output = jnp.all(
#             gate_pos + DIRECTIONS[gate_dir] * (1 + num_junctions) == state.wires_positions[index]
#         )
#         receive_output = jnp.logical_and(powered_gate, is_output)
#         source_nearby = jnp.logical_or(is_power, receive_output)

#         new_charge = jax.lax.select(source_nearby, static_params.max_wire_charge, nearby_charge)

#         # Safe wire access
#         is_wire = state.map[pos[0], pos[1]] == BlockType.WIRE.value
#         wire_matches = jnp.all(state.wires_positions == pos, axis=1)
#         wire_found = jnp.logical_and(is_wire, jnp.any(wire_matches))
#         wire_index = jnp.argmax(wire_matches)

#         safe_neighbor_charge = jax.lax.select(
#             wire_found,
#             state.wires_charge[wire_index],
#             0
#         )
#         decay_charge = jnp.maximum(0, safe_neighbor_charge - 1)
#         new_charge = jnp.maximum(new_charge, decay_charge)
#         nearby_charge = jnp.maximum(nearby_charge, new_charge)

#         wires_charge = wires_charge.at[index].set(nearby_charge)
#         return (wires_charge, nearby_charge, index), None


#     def check_one_wire(wires_charge, index):
#         nearby_charge = 0
#         (wires_charge, _, _), _ = jax.lax.scan(
#             check_max_charge,
#             (wires_charge, nearby_charge, index),
#             neighboring_wire_area
#         )
#         return wires_charge, None

#     new_wires_charge, _ = jax.lax.scan(
#         check_one_wire,
#         state.wires_charge,
#         jnp.arange(20) # max wires
#     )

#     state = state.replace(
#         wires_charge=new_wires_charge,
#     )
#     return state

# @jax.jit
# def update_wires(state, static_params):
#     neighboring_wire_area = jnp.array([
#         [-1, 0], [0, -1], [0, 1], [1, 0]
#     ], dtype=jnp.int32)

#     def safe_get_charge(pos):
#         is_wire = state.map[pos[0], pos[1]] == BlockType.WIRE.value
#         wire_matches = jnp.all(state.wires_positions == pos, axis=1)
#         wire_found = jnp.any(wire_matches)
#         wire_index = jnp.argmax(wire_matches)
#         valid = jnp.logical_and(is_wire, wire_found)
#         return jax.lax.select(valid, state.wires_charge[wire_index], 0)


#     def check_max_charge(carry, loc_add):
#         wires_charge, nearby_charge, index = carry
#         pos = state.wires_positions[index] + loc_add

#         def cond(carry):
#             pos, count = carry
#             return state.map[pos[0], pos[1]] == BlockType.JUNCTION.value

#         def body(carry):
#             pos, count = carry
#             return pos + loc_add, count + 1

#         pos, num_junctions = jax.lax.while_loop(cond, body, (pos, 0))

#         is_power = jnp.logical_or(
#             jnp.logical_or(
#                 state.map[pos[0], pos[1]] == BlockType.POWER.value,
#                 state.map[pos[0], pos[1]] == BlockType.INPUT_ON.value
#             ),
#             jnp.logical_and(
#                 state.map[pos[0], pos[1]] == BlockType.PRESSURE_PLATE.value,
#                 jnp.logical_or(
#                     jnp.all(state.player_position == pos),
#                     state.mob_map[pos[0], pos[1]]
#                 )
#             )
#         )

#         gate_matches = jnp.all(state.logic_gates_positions == pos, axis=1)
#         gate_found = jnp.any(gate_matches)
#         logic_gate_index = jnp.argmax(gate_matches)

#         powered_gate = jax.lax.select(gate_found, state.logic_gates_power[logic_gate_index], False)
#         gate_dir = jax.lax.select(gate_found, state.logic_gates_direction[logic_gate_index], 0)
#         gate_pos = jax.lax.select(gate_found, state.logic_gates_positions[logic_gate_index], jnp.array([0, 0]))

#         is_output = jnp.all(
#             gate_pos + DIRECTIONS[gate_dir] * (1 + num_junctions) == state.wires_positions[index]
#         )

#         receive_output = jnp.logical_and(powered_gate, is_output)
#         source_nearby = jnp.logical_or(is_power, receive_output)

#         new_charge = jax.lax.select(source_nearby, static_params.max_wire_charge, nearby_charge)

#         is_wire = state.map[pos[0], pos[1]] == BlockType.WIRE.value
#         wire_matches = jnp.all(state.wires_positions == pos, axis=1)
#         wire_found = jnp.logical_and(is_wire, jnp.any(wire_matches))
#         wire_index = jnp.argmax(wire_matches)

#         valid_index = jnp.logical_and(wire_found, wire_index < 20)
#         neighbor_charge = jax.lax.select(valid_index, state.wires_charge[wire_index], 0)
#         decay_charge = jnp.maximum(0, neighbor_charge - 1)

#         new_charge = jnp.maximum(new_charge, decay_charge)
#         nearby_charge = jnp.maximum(nearby_charge, new_charge)

#         wires_charge = wires_charge.at[index].set(nearby_charge)
#         return (wires_charge, nearby_charge, index), None

#     def check_one_wire(wires_charge, index):
#         nearby_charge = 0
#         (wires_charge, _, _), _ = jax.lax.scan(
#             check_max_charge,
#             (wires_charge, nearby_charge, index),
#             neighboring_wire_area
#         )
#         return wires_charge, None

#     new_wires_charge, _ = jax.lax.scan(
#         check_one_wire,
#         state.wires_charge,
#         jnp.arange(20) # max wires
#     )

#     state = state.replace(
#         wires_charge=new_wires_charge,
#     )
#     return state

# @jax.jit
def update_extenders(state, static_params):
    neighboring_wire_area = jnp.array([
        [-1, 0], [0, -1], [0, 1], [1, 0]
    ], dtype=jnp.int32)
    
    def nearby_wire(carry, loc_add):
        is_extending, index = carry
        pos = state.extenders_positions[index] + loc_add
        pos = in_bounds(state, pos)
        # jax.debug.print("map: {}", pos)
        is_wire = state.map[pos[0], pos[1]] == BlockType.WIRE.value
        wire_index = jax.lax.select(is_wire, jnp.argmax(jnp.all(state.wires_positions == pos, axis=1)), static_params.max_wires + 1)
        valid_index = jnp.logical_and(wire_index >= 0, wire_index <= static_params.max_wire_charge)
        
        extending = jnp.logical_or(is_extending, state.wires_charge[wire_index] > 0)
        
        return (extending, index), None
        

    def check_one_extender(new_map, index):
        extension_position = state.extenders_positions[index] + DIRECTIONS[state.extenders_direction[index]]
        is_extending = False
        (is_extending, _), _ = jax.lax.scan(
            nearby_wire,
            (is_extending, index),
            neighboring_wire_area
        )

        placed_extender_block = jax.lax.select(
            is_extending,
            BlockType.EXTENSION.value,
            BlockType.PATH.value,
        )
        # jax.debug.print("{}", placed_extender_block)
        new_map = new_map.at[extension_position[0], extension_position[1]].set(
            placed_extender_block
        )

        return new_map, None
        
    new_map = state.map
    new_map, _ = jax.lax.scan(
        check_one_extender,
        new_map,
        jnp.arange(static_params.max_extenders)
        # jnp.arange(1)
    )

    
    state = state.replace(
        map=new_map,
    )

    return state

# @jax.jit
def update_logic(state, static_params):
    # neighboring_wire_area = jnp.array([
    #     [-1, 0], [0, -1], [0, 1], [1, 0]
    # ], dtype=jnp.int32)

    def rotate_left(direction):
        return jnp.array([-direction[1], direction[0]])

    def rotate_right(direction):
        return jnp.array([direction[1], -direction[0]])
    
    def input_gate(position, direction):
        pos_input_1 = position + rotate_left(DIRECTIONS[direction])
        pos_input_2 = position + rotate_right(DIRECTIONS[direction])
        pos_input_1 = in_bounds(state, pos_input_1)
        pos_input_2 = in_bounds(state, pos_input_2)

        # jax.debug.print("{}", pos_input_1)
        is_wire_1 = state.map[pos_input_1[0], pos_input_1[1]] == BlockType.WIRE.value
        is_wire_2 = state.map[pos_input_2[0], pos_input_2[1]] == BlockType.WIRE.value
        wire_index_1 = jax.lax.select(is_wire_1, jnp.argmax(jnp.all(state.wires_positions == pos_input_1, axis=1)), static_params.max_wires + 1)
        wire_index_2 = jax.lax.select(is_wire_2, jnp.argmax(jnp.all(state.wires_positions == pos_input_2, axis=1)), static_params.max_wires + 1)
        input_1 = state.wires_charge[wire_index_1] > 0
        input_2 = state.wires_charge[wire_index_2] > 0
        return jnp.array([input_1, input_2])

    def input_NOT(position, direction):
        pos_input = position - DIRECTIONS[direction]
        pos_input = in_bounds(state, pos_input)
        # jax.debug.print("{}", pos_input)
        is_wire = state.map[pos_input[0], pos_input[1]] == BlockType.WIRE.value
        wire_index = jax.lax.select(is_wire, jnp.argmax(jnp.all(state.wires_positions == pos_input, axis=1)), static_params.max_wires + 1)
        input = state.wires_charge[wire_index] > 0
        return jnp.array([input, False])
    
    def update_NONE(input):
        return False
    
    def update_AND(input):
        input_1, input_2 = input
        return jnp.logical_and(input_1, input_2)

    def update_OR(input):
        input_1, input_2 = input
        return jnp.logical_or(input_1, input_2)
    
    def update_XOR(input):
        input_1, input_2 = input
        return jnp.logical_xor(input_1, input_2)

    def update_NOT(input):
        return jnp.logical_not(input[0])

    def update_logic_gate(carry):
        update_functions = [update_NONE, update_AND, update_OR, update_XOR, update_NOT]
        power, index = carry
        type = state.logic_gates_type[index]
        pos_output = state.logic_gates_positions[index] + DIRECTIONS[state.logic_gates_direction[index]] # doesnt matter i think

        input = jax.lax.select(type == 4, 
            input_NOT(state.logic_gates_positions[index], state.logic_gates_direction[index]),
            input_gate(state.logic_gates_positions[index], state.logic_gates_direction[index]) 
        )
        output = jax.lax.switch(type, update_functions, input)
        power = power.at[index].set(output)
        return power

    def check_one_logic_gate(logic_gates_power, index):
        
        new_logic_gates_power = update_logic_gate((logic_gates_power, index))

        return new_logic_gates_power, None

    new_logic_gates_power = state.logic_gates_power
    new_logic_gates_power, _ = jax.lax.scan(
        check_one_logic_gate,
        new_logic_gates_power,
        jnp.arange(5) #static_params.max_logic_gates
    )

    
    state = state.replace(
        logic_gates_power=new_logic_gates_power,
    )

    return state

# @jax.jit    
def update_all_wiring_components(i, carry):
    state, static_params = carry
    new_state = state
        
    # Wires
    # new_state = update_wires(new_state, static_params)
    new_state = update_wires(new_state, static_params)

    # Logic_gates
    new_state = update_logic(new_state, static_params)
    # new_state = update_logic(new_state, static_params)

    return new_state, static_params

# @jax.jit
def update_wiring_max(state, static_params, n=1):
    new_state = state
    
    new_state, _ = jax.lax.fori_loop(0, n, update_all_wiring_components, (state, static_params))
    # Extenders
    new_state = update_extenders(new_state, static_params)

    # new_state = update_extenders(new_state, static_params)
    new_state = update_wiring_achievements(new_state, static_params)
    # _, new_outputs = jax.lax.scan(retrieve_output, state, jnp.arange(static_params.max_outputs))
    # new_state = new_state.replace(old_outputs=new_outputs)
    return new_state

# @jax.jit
def update_wiring_achievements(state, static_params):
    new_state = state
    # Achievements
    # new_state = jax.lax.select(new_state.achievements[Achievement.ACTIVATE_OUTPUT.value], activate_output(new_state, static_params), new_state)
    # new_state = jax.lax.select(new_state.achievements[Achievement.ACTIVATE_LOGIC_GATE.value], activate_logic_gate(new_state, static_params), new_state)
    # new_state = jax.lax.select(new_state.achievements[Achievement.HALF_ADDER.value], half_adder(new_state, static_params), new_state)
    # new_state = jax.lax.select(new_state.achievements[Achievement.FULL_ADDER.value], full_adder(new_state, static_params), new_state)
    # new_state = jax.lax.select(new_state.achievements[Achievement.BIN_TO_GRAY.value], bin_to_gray(new_state, static_params), new_state)
    # new_state = jax.lax.select(new_state.achievements[Achievement.TRAP.value], trap(new_state, static_params), new_state)
    # new_state = jax.lax.select(new_state.achievements[Achievement.DOOR.value], door(new_state, static_params), new_state)
    new_state = activate_output(new_state, static_params)

    new_state = activate_logic_gate(new_state, static_params)

    new_state = half_adder(new_state, static_params)
    # new_state, reward = jax.lax.cond(
    #     new_state.achievements[Achievement.HALF_ADDER.value],
    #     lambda _: new_state, reward,
    #     lambda _: half_adder_curriculum(new_state, static_params),
    #     operand=None
    # )
    new_state = full_adder(new_state, static_params)

    new_state = bin_to_gray(new_state, static_params)

    new_state = trap(new_state, static_params)

    new_state = door(new_state, static_params)


    # new_state = activate_logic_gate(new_state, static_params)
    # # new_state = implement_AND_gate(new_state, static_params)
    # new_state = half_adder(new_state, static_params)
    # new_state = full_adder(new_state, static_params)
    # new_state = bin_to_gray(new_state, static_params)
    # new_state = trap(new_state, static_params)
    # new_state = door(new_state, static_params)

    # new_input_switched = jax.lax.select(new_state.input_switched > 0, new_state.input_switched - 1, new_state.input_switched)

    # new_state = new_state.replace(
    #     input_switched=new_input_switched,
    # )
    return new_state


# Reward functions
def activate_output(state, static_params):
    new_achievements = state.achievements
    all_outputs = jnp.arange(static_params.max_outputs)
    outputs_on = jax.lax.scan(retrieve_output, state, all_outputs)[1]
    # jax.debug.print("{}", outputs_on)
    is_powering_output = jnp.any(outputs_on)
    new_achievements = new_achievements.at[Achievement.ACTIVATE_OUTPUT.value].set(
        jnp.logical_or(
            new_achievements[Achievement.ACTIVATE_OUTPUT.value], is_powering_output
        )
    )
    state = state.replace(
        achievements=new_achievements,
    )
    return state

def activate_logic_gate(state, static_params):
    # new_achievements = state.achievements
    no_not_mask = state.logic_gates_type != 4
    no_not_active = jnp.logical_and(state.logic_gates_power, no_not_mask)
    logic_gate_active = jnp.any(no_not_active)
    new_achievements = state.achievements.at[Achievement.ACTIVATE_LOGIC_GATE.value].set(
        jnp.logical_or(
            state.achievements[Achievement.ACTIVATE_LOGIC_GATE.value], logic_gate_active
        )
    )
    state = state.replace(
        achievements=new_achievements,
    )
    return state

# def implement_AND_gate(state, static_params):
#     all_inputs = jnp.arange(static_params.max_inputs)
#     all_outputs = jnp.arange(static_params.max_outputs)
#     AND_gate = jnp.argmax(jnp.all(state.logic_gates_type == 1)) # DOESNT WORK XD
#     # jax.debug.print("AND_gate{}", AND_gate)
#     inputs_on = jax.lax.scan(retrieve_input, state, all_inputs)[1]
#     outputs_on = jax.lax.scan(retrieve_output, state, all_outputs)[1]
#     AND_on = state.logic_gates_power[AND_gate]

#     two_inputs_on = jnp.sum(inputs_on) >= 2
#     any_outputs_on = jnp.any(outputs_on)
#     # jax.debug.print("{}", two_inputs_on)
#     implemented_AND_gate = jnp.logical_and(
#         jnp.logical_and(
#             two_inputs_on,
#             any_outputs_on
#         ),
#         AND_on
#     )
#     new_achievements = state.achievements.at[Achievement.IMPLEMENT_AND_GATE.value].set(
#         jnp.logical_or(
#             state.achievements[Achievement.IMPLEMENT_AND_GATE.value], implemented_AND_gate
#         )
#     )
#     state = state.replace(
#         achievements=new_achievements,
#     )
#     return state

# def binary_adder(state, static_params):
#     n_inputs = jnp.sum(state.inputs_mask) == 2
#     n_outputs = jnp.sum(state.inputs_mask) == 2
#     all_inputs = jnp.arange(static_params.max_inputs)
#     all_outputs = jnp.arange(static_params.max_outputs)
#     AND_gate = jnp.argmax(state.logic_gates_type == 1)
#     XOR_gate = jnp.argmax(state.logic_gates_type == 3)

#     inputs_on = jax.lax.scan(retrieve_input, state, all_inputs)[1]
#     outputs_on = jax.lax.scan(retrieve_output, state, all_outputs)[1]
#     one_output = jnp.sum(outputs_on) == 1
#     AND_on = state.logic_gates_power[AND_gate]
#     XOR_on = state.logic_gates_power[XOR_gate]

#     n_correct = jnp.logical_and(n_inputs, n_outputs)
#     functional_sum = jnp.logical_and(
#         jnp.logical_and(
#             XOR_on,
#             jnp.sum(inputs_on) == 1
#         ),
#         one_output
#     )
#     functional_carry = jnp.logical_and(
#         jnp.logical_and(
#             AND_on,
#             jnp.sum(inputs_on) == 2
#         ),
#         one_output
#     )
#     new_achievements = state.achievements
#     new_achievements = new_achievements.at[Achievement.FUNCTIONAL_SUM.value].set(
#         jnp.logical_or(
#             new_achievements[Achievement.FUNCTIONAL_SUM.value], functional_sum
#         )
#     )
#     new_achievements = new_achievements.at[Achievement.FUNCTIONAL_CARRY.value].set(
#         jnp.logical_or(
#             new_achievements[Achievement.FUNCTIONAL_CARRY.value], functional_carry
#         )
#     )

#     binary_adder = jnp.logical_and(
#         jnp.logical_and(
#             new_achievements[Achievement.FUNCTIONAL_SUM.value],
#             new_achievements[Achievement.FUNCTIONAL_CARRY.value]
#         ),
#         n_correct
#     )
#     new_achievements = new_achievements.at[Achievement.BINARY_ADDER.value].set(
#         jnp.logical_or(
#             new_achievements[Achievement.BINARY_ADDER.value], binary_adder
#         )
#     )
#     state = state.replace(
#         achievements=new_achievements,
#     )
#     return state

def half_adder(state, static_params):

    input_A = retrieve_input(state, 0)[1]
    input_B = retrieve_input(state, 1)[1]

    output_sum = retrieve_output(state, 0)[1]
    output_carry = retrieve_output(state, 1)[1]

    truth_table = state.truth_table_half_adder

    inputs = jnp.array([input_A, input_B, output_sum, output_carry], dtype=bool)

    real_truth_table = jnp.array([
        [False, False, False, False],
        [False, True,  True,  False],
        [True,  False, True,  False],
        [True,  True,  False, True ]
    ], dtype=bool)

    matches = jnp.all(inputs == real_truth_table, axis=1) 

    truth_table = jnp.logical_or(truth_table, matches)
    full_truth_table = jnp.all(truth_table)

    new_achievements = state.achievements.at[Achievement.HALF_ADDER.value].set(
        jnp.logical_or(
            state.achievements[Achievement.HALF_ADDER.value], full_truth_table
        )
    )
    state = state.replace(truth_table_half_adder=truth_table,
                          achievements=new_achievements)

    return state

# def half_adder_curriculum(state, static_params):
#     stage = state.half_adder_curriculum_stage
#     achievements = state.achievements

#     input_A = retrieve_input(state, 0)[1]
#     input_B = retrieve_input(state, 1)[1]
#     output_sum = retrieve_output(state, 0)[1]
#     output_carry = retrieve_output(state, 1)[1]
#     inputs = jnp.array([input_A, input_B, output_sum, output_carry], dtype=bool)

#     gate_types = state.logic_gates_type
#     gate_power = state.logic_gates_power
#     gate_mask = state.logic_gates_mask
#     gate_positions = state.logic_gates_positions
#     gate_directions = state.logic_gates_direction
#     outputs = state.outputs_positions

#     xor_mask = jnp.logical_and(gate_types == 3, gate_mask)
#     and_mask = jnp.logical_and(gate_types == 1, gate_mask)

#     xor_index = jnp.argmax(xor_mask.astype(jnp.float32))
#     and_index = jnp.argmax(and_mask.astype(jnp.float32))
#     xor_powered = gate_power[xor_index]
#     and_powered = gate_power[and_index]

#     def advance(condition, reward_val, achievement_key=None):
#         new_stage = jax.lax.select(condition, stage + 1, stage)
#         reward = jax.lax.select(condition, reward_val, 0.0)
#         new_state = state.replace(half_adder_curriculum_stage=new_stage)
#         if achievement_key is not None:
#             updated = achievements.at[achievement_key].set(
#                 jnp.logical_or(achievements[achievement_key], condition)
#             )
#             new_state = new_state.replace(achievements=updated)
#         return new_state, reward

#     def stage0(s): return advance(jnp.sum(s.inputs_mask) == 2, 0.5)
#     def stage1(s): return advance(jnp.any(xor_mask), 1.0)
#     def stage2(s): return advance(jnp.any(and_mask), 1.0)

#     def output_near_gate(gate_idx, output_idx, threshold=3.0):
#         dist = jnp.sum(jnp.abs(outputs[output_idx] - gate_positions[gate_idx]))
#         return dist < threshold

#     def stage3(s): return advance(output_near_gate(xor_index, 0), 2.0)
#     def stage4(s): return advance(output_near_gate(and_index, 1), 2.0)
#     def stage5(s): return advance(achievements[Achievement.ACTIVATE_INPUT.value], 0.5)
#     def stage6(s): return advance(achievements[Achievement.ACTIVATE_LOGIC_GATE.value], 1.0)

#     def stage7(s):
#         xor_out = gate_positions[xor_index] + DIRECTIONS[gate_directions[xor_index]]
#         and_out = gate_positions[and_index] + DIRECTIONS[gate_directions[and_index]]
#         match_xor = jnp.all(outputs[0] == xor_out)
#         match_and = jnp.all(outputs[1] == and_out)
#         condition = jnp.logical_and(match_xor, match_and)
#         reward = jnp.where(condition, 3.0, match_xor.astype(jnp.float32) * 1.5 + match_and.astype(jnp.float32) * 1.5)

#         updated_achievements = achievements.at[Achievement.STAGE7.value].set(
#             jnp.logical_or(achievements[Achievement.STAGE7.value], condition)
#         )
#         new_state = state.replace(
#             half_adder_curriculum_stage=jax.lax.select(condition, stage + 1, stage),
#             achievements=updated_achievements
#         )
#         return new_state, reward


#     def stage8(s): return advance(xor_powered, 1.0, Achievement.STAGE8.value)

#     def truth_table_stage(target_inputs, target_gates, achv):
#         def fn(s):
#             io_match = jnp.all(inputs == target_inputs)
#             power_match = jnp.all(jnp.array([xor_powered, and_powered]) == target_gates)
#             return advance(jnp.logical_and(io_match, power_match), 3.0, achv)
#         return fn

#     stage9  = truth_table_stage(jnp.array([1, 0, 1, 0], dtype=bool), jnp.array([1, 0], dtype=bool), Achievement.STAGE9.value)
#     stage10 = truth_table_stage(jnp.array([0, 1, 1, 0], dtype=bool), jnp.array([1, 0], dtype=bool), Achievement.STAGE10.value)
#     stage11 = truth_table_stage(jnp.array([1, 1, 0, 1], dtype=bool), jnp.array([0, 1], dtype=bool), Achievement.STAGE11.value)
#     stage12 = truth_table_stage(jnp.array([0, 0, 0, 0], dtype=bool), jnp.array([0, 0], dtype=bool), Achievement.STAGE12.value)

#     def complete(s): return s, 0.0

#     stages = [
#         stage0, stage1, stage2, stage3, stage4, stage5, stage6,
#         stage7, stage8, stage9, stage10, stage11, stage12, complete
#     ]

#     return jax.lax.switch(stage, stages, state)

def half_adder_curriculum(state, static_params):
    input_A = retrieve_input(state, 0)[1]
    input_B = retrieve_input(state, 1)[1]
    output_sum = retrieve_output(state, 0)[1]
    output_carry = retrieve_output(state, 1)[1]
    inputs = jnp.array([input_A, input_B, output_sum, output_carry], dtype=bool)
    stage = state.half_adder_curriculum_stage
    neighboring_wire_area = jnp.array([
        [-1, 0], [0, -1], [0, 1], [1, 0]
    ], dtype=jnp.int32)


    # jax.debug.print("stage: {}", stage)
    # return state, 0.0
    def check_area(carry, loc_add):
        pos, target_value, n_targets = carry
        target_pos = pos + loc_add
        n_targets += state.map[target_pos[0], target_pos[1]] == target_value
        return (pos, target_value, n_targets), None

    def stage0(state):
        condition = jnp.sum(state.inputs_mask) == 2
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 0.5, 0.0)
        new_achievements = state.achievements.at[Achievement.STAGE0.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE0.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward
    
    def stage1(state):
        n_wires_0 = 0
        n_wires_1 = 0
        (_, _, n_wires_0), _ = jax.lax.scan(
            check_area,
            (state.inputs_positions[0], BlockType.WIRE.value, n_wires_0),
            neighboring_wire_area
        )
        (_, _, n_wires_1), _ = jax.lax.scan(
            check_area,
            (state.inputs_positions[1], BlockType.WIRE.value, n_wires_1),
            neighboring_wire_area
        )
        condition = jnp.logical_and(n_wires_0 >= 2, n_wires_1 >= 2)
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 3.0, 0.0)
        new_achievements = state.achievements.at[Achievement.STAGE1.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE1.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward
    
    def stage2(state):
        condition = jnp.any(state.logic_gates_type == 3) # XOR
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 1.0, 0.0)
        new_achievements = state.achievements.at[Achievement.STAGE2.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE2.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward

    def stage3(state):
        condition = state.achievements[Achievement.ACTIVATE_INPUT.value]
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 0.5, 0.0)
        new_achievements = state.achievements.at[Achievement.STAGE3.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE3.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward

    def stage4(state):
        XOR_index = jnp.argmax(state.logic_gates_type == 3)
        XOR_powered = state.logic_gates_power[XOR_index]
        condition = XOR_powered
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 1.0, 0.0)
        new_achievements = state.achievements.at[Achievement.STAGE4.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE4.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward

    def stage5(state):
        condition = jnp.any(state.logic_gates_type == 1) # AND
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 1.0, 0.0)
        new_achievements = state.achievements.at[Achievement.STAGE5.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE5.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward
    
    # def stage6(state):
    #     condition = jnp.sum(state.outputs_mask) == 2
    #     new_stage = jax.lax.select(condition, stage + 1, stage)
    #     reward = jax.lax.select(condition, 0.5, -0.1)
    #     new_achievements = state.achievements.at[Achievement.STAGE6.value].set(
    #         jnp.logical_or(
    #             state.achievements[Achievement.STAGE6.value], condition
    #         )
    #     )

    #     return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward
    
    def stage6(state):
        XOR_gate = 0
        AND_gate = 0
        (_, _, XOR_gate), _ = jax.lax.scan(
            check_area,
            (state.outputs_positions[0], BlockType.XOR.value, XOR_gate),
            neighboring_wire_area
        )
        (_, _, AND_gate), _ = jax.lax.scan(
            check_area,
            (state.outputs_positions[1], BlockType.AND.value, AND_gate),
            neighboring_wire_area
        )
        condition = jnp.logical_and(XOR_gate, AND_gate)
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 5.0, 0.0)
        new_achievements = state.achievements.at[Achievement.STAGE6.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE6.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward
    
    # def stage5(state):
    #     condition = state.achievements[Achievement.ACTIVATE_INPUT.value]
    #     new_stage = jax.lax.select(condition, stage + 1, stage)
    #     reward = jax.lax.select(condition, 0.5, 0.0)
    #     return state.replace(half_adder_curriculum_stage=new_stage), reward
    
    # def stage8(state):
    #     condition = state.achievements[Achievement.ACTIVATE_LOGIC_GATE.value]
    #     new_stage = jax.lax.select(condition, stage + 1, stage)
    #     reward = jax.lax.select(condition, 1.0, 0.0)
    #     new_achievements = state.achievements.at[Achievement.STAGE8.value].set(
    #         jnp.logical_or(
    #             state.achievements[Achievement.STAGE8.value], condition
    #         )
    #     )

    #     return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward
    
    # def stage5(state):
    #     condition = jnp.sum(state.outputs_mask) == 2
    #     new_stage = jax.lax.select(condition, stage + 1, stage)
    #     reward = jax.lax.select(condition, 0.5, 0.0)
    #     return state.replace(half_adder_curriculum_stage=new_stage), reward
    
    # def stage6(state):
    #     condition = state.achievements[Achievement.ACTIVATE_OUTPUT.value]
    #     new_stage = jax.lax.select(condition, stage + 1, stage)
    #     reward = jax.lax.select(condition, 1.0, 0.0)
    #     return state.replace(half_adder_curriculum_stage=new_stage), reward
    
    # def stage7(state):
    #     XOR_index = jnp.argmax(state.logic_gates_type == 3)
    #     AND_index = jnp.argmax(state.logic_gates_type == 1)
    #     XOR_out_pos = state.logic_gates_positions[XOR_index] + DIRECTIONS[state.logic_gates_direction[XOR_index]]
    #     AND_out_pos = state.logic_gates_positions[AND_index] + DIRECTIONS[state.logic_gates_direction[AND_index]]
    #     XOR_match_out = jnp.all(state.outputs_positions[0] == XOR_out_pos)
    #     AND_match_out = jnp.all(state.outputs_positions[1] == AND_out_pos)

    #     reward = XOR_match_out.astype(jnp.float32) * 1.5 + AND_match_out.astype(jnp.float32) * 1.5
    #     condition = jnp.logical_and(XOR_match_out, AND_match_out)

    #     new_stage = jax.lax.select(condition, stage + 1, stage)
    #     reward = jax.lax.select(condition, 3.0, reward)

    #     new_achievements = state.achievements.at[Achievement.STAGE7.value].set( # this will not be in the final product, just to see which stage it reaches
    #         jnp.logical_or(
    #             state.achievements[Achievement.STAGE7.value], condition
    #         )
    #     )
    #     return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward

    def stage7(state):
        truth_table1 = jnp.array([True,  False, True,  False], dtype=bool)
        correct_logic_gates = jnp.array([True, False], dtype=bool)
        XOR_index = jnp.argmax(state.logic_gates_type == 3)
        XOR_powered = state.logic_gates_power[XOR_index]
        AND_index = jnp.argmax(state.logic_gates_type == 1)
        AND_powered = state.logic_gates_power[AND_index]
        current_gates = jnp.array([XOR_powered, AND_powered])
        correct_gates_powered = jnp.all(current_gates == correct_logic_gates)
        condition = jnp.logical_and(jnp.all(inputs == truth_table1), correct_gates_powered)
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 3.0, 0.0)

        new_achievements = state.achievements.at[Achievement.STAGE7.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE7.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward

    def stage8(state):
        truth_table2 = jnp.array([False, True,  True,  False], dtype=bool)
        correct_logic_gates = jnp.array([True, False], dtype=bool)
        XOR_index = jnp.argmax(state.logic_gates_type == 3)
        XOR_powered = state.logic_gates_power[XOR_index]
        AND_index = jnp.argmax(state.logic_gates_type == 1)
        AND_powered = state.logic_gates_power[AND_index]
        current_gates = jnp.array([XOR_powered, AND_powered])
        correct_gates_powered = jnp.all(current_gates == correct_logic_gates)
        condition = jnp.logical_and(jnp.all(inputs == truth_table2), correct_gates_powered)
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 3.0, 0.0)

        new_achievements = state.achievements.at[Achievement.STAGE8.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE8.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward

    def stage9(state):
        truth_table3 = jnp.array([True,  True,  False, True ], dtype=bool)
        correct_logic_gates = jnp.array([False, True], dtype=bool)
        XOR_index = jnp.argmax(state.logic_gates_type == 3)
        XOR_powered = state.logic_gates_power[XOR_index]
        AND_index = jnp.argmax(state.logic_gates_type == 1)
        AND_powered = state.logic_gates_power[AND_index]
        current_gates = jnp.array([XOR_powered, AND_powered])
        correct_gates_powered = jnp.all(current_gates == correct_logic_gates)
        condition = jnp.logical_and(jnp.all(inputs == truth_table3), correct_gates_powered)
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 3.0, 0.0)

        new_achievements = state.achievements.at[Achievement.STAGE9.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE9.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward
    
    def stage10(state):
        truth_table0 = jnp.array([False, False, False, False], dtype=bool)
        correct_logic_gates = jnp.array([False, False], dtype=bool)
        XOR_index = jnp.argmax(state.logic_gates_type == 3)
        XOR_powered = state.logic_gates_power[XOR_index]
        AND_index = jnp.argmax(state.logic_gates_type == 1)
        AND_powered = state.logic_gates_power[AND_index]
        current_gates = jnp.array([XOR_powered, AND_powered])
        correct_gates_powered = jnp.all(current_gates == correct_logic_gates)
        condition = jnp.logical_and(jnp.all(inputs == truth_table0), correct_gates_powered)
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 3.0, 0.0)

        new_achievements = state.achievements.at[Achievement.STAGE10.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE10.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward
    
    def complete(state):
        # state = state.replace(
        #     achievements=state.achievements.at[Achievement.HALF_ADDER.value].set(True)
        # )
        # reward = 0.0
        # new_stage = 13
        return state, 0.0

    # Wrap the stage functions into a switch table
    stages = [stage0, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, stage10, complete]

    # Use `jax.lax.switch` to dispatch
    state, reward = jax.lax.switch(stage, stages, state)

    return state, reward

def full_adder(state, static_params):
    
    # input_switched = state.input_switched # might be needed

    input_A = retrieve_input(state, 0)[1]
    input_B = retrieve_input(state, 1)[1]
    input_C = retrieve_input(state, 2)[1]

    output_sum = retrieve_output(state, 0)[1]
    output_carry = retrieve_output(state, 1)[1]

    truth_table = state.truth_table_full_adder

    inputs = jnp.array([input_A, input_B, input_C, output_sum, output_carry], dtype=bool)

    real_truth_table = jnp.array([
        [False, False, False, False, False],
        [False, False, True , True , False],
        [False, True , False, True , False],
        [False, True , True , False, True ],
        [True , False, False, True , False],
        [True , False, True , False, True ],
        [True , True , False, False, True ],
        [True , True , True , True , True ],
    ], dtype=bool)

    matches = jnp.all(inputs == real_truth_table, axis=1) 

    # _, new_outputs = jax.lax.scan(retrieve_output, state, jnp.arange(static_params.max_outputs))

    # did_update = jnp.logical_and(
    #     input_switched > 0,
    #     jnp.any(new_outputs != state.old_outputs)
    # )

    # truth_table = jax.lax.select(did_update, jnp.logical_or(truth_table, matches), truth_table)
    truth_table = jnp.logical_or(truth_table, matches)
    full_truth_table = jnp.all(truth_table)

    new_achievements = state.achievements.at[Achievement.FULL_ADDER.value].set(
        jnp.logical_or(
            state.achievements[Achievement.FULL_ADDER.value], full_truth_table
        )
    )
    state = state.replace(truth_table_full_adder=truth_table,
                          achievements=new_achievements)

    return state

def bin_to_gray(state, static_params):
    input_A = retrieve_input(state, 0)[1]
    input_B = retrieve_input(state, 1)[1]
    input_C = retrieve_input(state, 2)[1]

    output_0 = retrieve_output(state, 0)[1]
    output_1 = retrieve_output(state, 1)[1]
    output_2 = retrieve_output(state, 2)[1]

    truth_table = state.truth_table_bin_to_gray

    inputs = jnp.array([input_A, input_B, input_C, output_0, output_1, output_2], dtype=bool)

    real_truth_table = jnp.array([ 
        [False, False, False, False, False, False],
        [False, False, True , False, False, True ],
        [False, True , False, False, True , True ],
        [False, True , True , False, True , False],
        [True , False, False, True , True , False],
        [True , False, True , True , True , True ],
        [True , True , False, True , False, True ],
        [True , True , True , True , False, False]
    ], dtype=bool)

    matches = jnp.all(inputs == real_truth_table, axis=1) 

    truth_table = jnp.logical_or(truth_table, matches)

    full_truth_table = jnp.all(truth_table)

    new_achievements = state.achievements.at[Achievement.BIN_TO_GRAY.value].set(
        jnp.logical_or(
            state.achievements[Achievement.BIN_TO_GRAY.value], full_truth_table
        )
    )
    state = state.replace(truth_table_bin_to_gray=truth_table,
                          achievements=new_achievements)

    return state

def trap(state, static_params):
    neighboring_area = jnp.array([
        [-1, 0], [0, -1], [0, 1], [1, 0]
    ], dtype=jnp.int32)

    def check_area(carry, loc_add):
        is_pressure_plate, index = carry
        pos = state.extenders_positions[index] + DIRECTIONS[state.extenders_direction[index]] + loc_add
        pressure_plate = jnp.logical_or(is_pressure_plate, state.map[pos[0], pos[1]] == BlockType.PRESSURE_PLATE.value)
        return (pressure_plate, index), None

    def check_extended(is_pressure_plate, index):
        (is_pressure_plate, _), _ = jax.lax.scan(
            check_area,
            (is_pressure_plate, index),
            neighboring_area
        )
        return is_pressure_plate, None
    
    is_pressure_plate = False
    is_pressure_plate, _ = jax.lax.scan(
        check_extended,
        is_pressure_plate,
        jnp.arange(static_params.max_extenders)
    )

    new_achievements = state.achievements.at[Achievement.TRAP.value].set(
        jnp.logical_or(
            state.achievements[Achievement.TRAP.value], is_pressure_plate
        )
    )

    state = state.replace(
        achievements=new_achievements,
    )
    return state

def door(state, static_params):
    neighboring_area = jnp.array([
        [-1, 0], [0, -1], [0, 1], [1, 0]
    ], dtype=jnp.int32)

    def check_area(carry, loc_add):
        is_extended = carry
        pos = state.player_position + loc_add
        extended = jnp.logical_or(is_extended, state.map[pos[0], pos[1]] == BlockType.EXTENSION.value)
        return extended, None
        
    is_on_pressure_plate = state.map[state.player_position[0], state.player_position[1]] == BlockType.PRESSURE_PLATE.value
    NOT_gate = jnp.argmax(state.logic_gates_type == 4)
    NOT_off = jnp.logical_and(state.logic_gates_power[NOT_gate] == False, state.logic_gates_mask[NOT_gate] == True)
    is_extended = False
    is_extended, _ = jax.lax.scan(
        check_area,
        is_extended,
        neighboring_area
    )

    is_door = jnp.logical_and(
        jnp.logical_and(
            is_on_pressure_plate,
            NOT_off
        ),
        is_extended
    )

    new_achievements = state.achievements.at[Achievement.DOOR.value].set(
        jnp.logical_or(
            state.achievements[Achievement.DOOR.value], is_door
        )
    )

    state = state.replace(
        achievements=new_achievements,
    )
    return state