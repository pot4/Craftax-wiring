from craftax.craftax_classic.constants import *
from craftax.craftax_classic.envs.craftax_state import *

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

def update_wires(state, static_params):
    neighboring_wire_area = jnp.array([
        [-1, 0], [0, -1], [0, 1], [1, 0]
    ], dtype=jnp.int32)

    def check_max_charge(carry, loc_add):
        wires_charge, nearby_charge, index = carry
        # jax.debug.print("loc_add: {}", loc_add)
        pos = state.wires_positions[index] + loc_add
        
        def cond(carry):
            pos, count = carry
            return state.map[pos[0], pos[1]] == BlockType.JUNCTION.value

        def body(carry):
            pos, count = carry
            return (pos + loc_add, count + 1)

        (pos, num_junctions) = jax.lax.while_loop(cond, body, (pos, 0))

        is_power = jnp.logical_or(
            jnp.logical_or(
                state.map[pos[0], pos[1]] == BlockType.POWER.value,
                state.map[pos[0], pos[1]] == BlockType.INPUT_ON.value
            ),
            jnp.logical_and(
                state.map[pos[0], pos[1]] == BlockType.PRESSURE_PLATE.value,
                jnp.logical_or(
                    jnp.all(state.player_position == pos),
                    jnp.all(state.mob_map[pos[0], pos[1]] != False)
                )

            )
        )

        logic_gate_index = jnp.argmax(jnp.all(state.logic_gates_positions == pos, axis=1)) # DOESNT WORK MAYBE IT DOES IDK
        powered_gate = state.logic_gates_power[logic_gate_index]
        is_output = jnp.all(
            state.logic_gates_positions[logic_gate_index] +
            DIRECTIONS[state.logic_gates_direction[logic_gate_index]] * (1+num_junctions)
            == state.wires_positions[index]
        )
        receive_output = jnp.logical_and(powered_gate, is_output)

        source_nearby = jnp.logical_or(is_power, receive_output)
        new_charge = jax.lax.select(source_nearby, static_params.max_wire_charge, nearby_charge)

        is_wire = state.map[pos[0], pos[1]] == BlockType.WIRE.value

        wire_index = jax.lax.select(is_wire, jnp.argmax(jnp.all(state.wires_positions == pos, axis=1)), static_params.max_wires + 1) #static_params.max_wires + 1
        
        valid_index = jnp.logical_and(wire_index >= 0, wire_index <= static_params.max_wire_charge)

        higher_charge = jnp.logical_and(
            valid_index, 
            state.wires_charge[wire_index] > nearby_charge
        )
        
        new_charge = jax.lax.select(
            higher_charge, 
            state.wires_charge[wire_index] - 1,
            new_charge
        )


        nearby_charge = new_charge
        wires_charge = wires_charge.at[index].set(new_charge)
        return (wires_charge, nearby_charge, index), None

    def check_one_wire(wires_charge, index):
        nearby_charge = 0
        (wires_charge, nearby_charge, _), _ = jax.lax.scan(
            check_max_charge,
            (wires_charge, nearby_charge, index),
            neighboring_wire_area
        )
        
        return wires_charge, None

    new_wires_charge, _ = jax.lax.scan(
        check_one_wire,
        state.wires_charge,
        jnp.arange(static_params.max_wires)
    )

    
    state = state.replace(
        wires_charge=new_wires_charge,
    )

    return state
# def update_wires1(state, static_params):
#     neighboring_wire_area = jnp.array([
#         [-1, 0], [0, -1], [0, 1], [1, 0]
#     ], dtype=jnp.int32)

#     def check_max_charge(carry, loc_add):
#         wires_charge, nearby_charge, index = carry
#         pos = state.wires_positions[index] + loc_add

#         def cond(carry):
#             pos, count = carry
#             return jnp.logical_and(
#                 (0 <= pos[0]) & (pos[0] < state.map.shape[0]) &
#                 (0 <= pos[1]) & (pos[1] < state.map.shape[1]),
#                 state.map[pos[0], pos[1]] == BlockType.JUNCTION.value
#             )

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
#         logic_gate_index = jnp.argmax(gate_matches)
#         gate_found = jnp.any(gate_matches)

#         powered_gate = jax.lax.select(gate_found, state.logic_gates_power[logic_gate_index], False)
#         gate_dir = jax.lax.select(gate_found, state.logic_gates_direction[logic_gate_index], 0)
#         gate_pos = jax.lax.select(gate_found, state.logic_gates_positions[logic_gate_index], jnp.array([0, 0]))

#         output_pos = gate_pos + DIRECTIONS[gate_dir] * (1 + num_junctions)
#         is_output = jnp.all(output_pos == state.wires_positions[index])

#         receive_output = jnp.logical_and(powered_gate, is_output)
#         source_nearby = jnp.logical_or(is_power, receive_output)

#         new_charge = jax.lax.select(source_nearby, static_params.max_wire_charge, nearby_charge)

#         is_wire = state.map[pos[0], pos[1]] == BlockType.WIRE.value
#         wire_matches = jnp.all(state.wires_positions == pos, axis=1)
#         wire_index = jnp.argmax(wire_matches)
#         wire_found = jnp.logical_and(is_wire, jnp.any(wire_matches))

#         valid_index = jnp.logical_and(wire_found, wire_index < static_params.max_wires)

#         higher_charge = jnp.logical_and(
#             valid_index,
#             state.wires_charge[wire_index] > nearby_charge
#         )

#         new_charge = jax.lax.select(
#             higher_charge,
#             state.wires_charge[wire_index] - 1,
#             new_charge
#         )

#         nearby_charge = new_charge
#         wires_charge = wires_charge.at[index].set(new_charge)
#         return (wires_charge, nearby_charge, index), None

#     def check_one_wire(wires_charge, index):
#         nearby_charge = 0
#         (wires_charge, nearby_charge, _), _ = jax.lax.scan(
#             check_max_charge,
#             (wires_charge, nearby_charge, index),
#             neighboring_wire_area
#         )
#         return wires_charge, None

#     new_wires_charge, _ = jax.lax.scan(
#         check_one_wire,
#         state.wires_charge,
#         jnp.arange(static_params.max_wires)
#     )

#     state = state.replace(
#         wires_charge=new_wires_charge,
#     )

#     return state

# def update_wires1(state, static_params):
#     neighboring_wire_area = jnp.array([
#         [-1, 0], [0, -1], [0, 1], [1, 0]
#     ], dtype=jnp.int32)

#     def check_max_charge(carry, loc_add):
#         wires_charge, nearby_charge, index = carry
#         pos = state.wires_positions[index] + loc_add

#         def cond(carry):
#             pos, count = carry
#             in_bounds = (
#                 (0 <= pos[0]) & (pos[0] < state.map.shape[0]) &
#                 (0 <= pos[1]) & (pos[1] < state.map.shape[1])
#             )
#             return jnp.logical_and(
#                 in_bounds,
#                 state.map[pos[0], pos[1]] == BlockType.JUNCTION.value
#             )

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
#                     state.mob_map[pos[0], pos[1]] == True
#                 )
#             )
#         )

#         logic_gate_index = jnp.argmax(jnp.all(state.logic_gates_positions == pos, axis=1))
#         gate_matches = jnp.all(state.logic_gates_positions == pos, axis=1)
#         gate_found = jnp.any(gate_matches)

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
#         wire_index = jnp.argmax(wire_matches)
#         wire_found = jnp.logical_and(is_wire, jnp.any(wire_matches))

#         valid_index = jnp.logical_and(wire_found, wire_index < static_params.max_wires)

#         higher_charge = jnp.logical_and(
#             valid_index,
#             state.wires_charge[wire_index] > nearby_charge
#         )

#         new_charge = jax.lax.select(
#             higher_charge,
#             state.wires_charge[wire_index] - 1,
#             new_charge
#         )

#         nearby_charge = jnp.maximum(nearby_charge, new_charge)
#         wires_charge = wires_charge.at[index].set(nearby_charge)
#         return (wires_charge, nearby_charge, index), None

#     def check_one_wire(wires_charge, index):
#         nearby_charge = 0
#         (wires_charge, nearby_charge, _), _ = jax.lax.scan(
#             check_max_charge,
#             (wires_charge, nearby_charge, index),
#             neighboring_wire_area
#         )
#         return wires_charge, None

#     new_wires_charge, _ = jax.lax.scan(
#         check_one_wire,
#         state.wires_charge,
#         jnp.arange(30) #static_params.max_wires
#     )

#     state = state.replace(
#         wires_charge=new_wires_charge,
#     )

    # return state
def update_wires(state, static_params):
    neighboring_wire_area = jnp.array([
        [-1, 0], [0, -1], [0, 1], [1, 0]
    ], dtype=jnp.int32)

    def check_max_charge(carry, loc_add):
        wires_charge, nearby_charge, index = carry
        pos = state.wires_positions[index] + loc_add

        def cond(carry):
            pos, count = carry
            return state.map[pos[0], pos[1]] == BlockType.JUNCTION.value

        def body(carry):
            pos, count = carry
            return pos + loc_add, count + 1

        pos, num_junctions = jax.lax.while_loop(cond, body, (pos, 0))

        is_power = jnp.logical_or(
            jnp.logical_or(
                state.map[pos[0], pos[1]] == BlockType.POWER.value,
                state.map[pos[0], pos[1]] == BlockType.INPUT_ON.value
            ),
            jnp.logical_and(
                state.map[pos[0], pos[1]] == BlockType.PRESSURE_PLATE.value,
                jnp.logical_or(
                    jnp.all(state.player_position == pos),
                    state.mob_map[pos[0], pos[1]]
                )
            )
        )

        gate_matches = jnp.all(state.logic_gates_positions == pos, axis=1)
        gate_found = jnp.any(gate_matches)
        logic_gate_index = jnp.argmax(gate_matches)

        powered_gate = jax.lax.select(gate_found, state.logic_gates_power[logic_gate_index], False)
        gate_dir = jax.lax.select(gate_found, state.logic_gates_direction[logic_gate_index], 0)
        gate_pos = jax.lax.select(gate_found, state.logic_gates_positions[logic_gate_index], jnp.array([0, 0]))

        is_output = jnp.all(
            gate_pos + DIRECTIONS[gate_dir] * (1 + num_junctions) == state.wires_positions[index]
        )

        receive_output = jnp.logical_and(powered_gate, is_output)
        source_nearby = jnp.logical_or(is_power, receive_output)

        new_charge = jax.lax.select(source_nearby, static_params.max_wire_charge, nearby_charge)

        is_wire = state.map[pos[0], pos[1]] == BlockType.WIRE.value
        wire_matches = jnp.all(state.wires_positions == pos, axis=1)
        wire_found = jnp.logical_and(is_wire, jnp.any(wire_matches))
        wire_index = jnp.argmax(wire_matches)

        valid_index = jnp.logical_and(wire_found, wire_index < 30)
        neighbor_charge = jax.lax.select(valid_index, state.wires_charge[wire_index], 0)
        decay_charge = jnp.maximum(0, neighbor_charge - 1)

        new_charge = jnp.maximum(new_charge, decay_charge)
        nearby_charge = jnp.maximum(nearby_charge, new_charge)

        wires_charge = wires_charge.at[index].set(nearby_charge)
        return (wires_charge, nearby_charge, index), None

    def check_one_wire(wires_charge, index):
        nearby_charge = 0
        (wires_charge, _, _), _ = jax.lax.scan(
            check_max_charge,
            (wires_charge, nearby_charge, index),
            neighboring_wire_area
        )
        return wires_charge, None

    new_wires_charge, _ = jax.lax.scan(
        check_one_wire,
        state.wires_charge,
        jnp.arange(30)
    )

    state = state.replace(
        wires_charge=new_wires_charge,
    )
    return state

def update_extenders(state, static_params):
    neighboring_wire_area = jnp.array([
        [-1, 0], [0, -1], [0, 1], [1, 0]
    ], dtype=jnp.int32)
    
    def nearby_wire(carry, loc_add):
        is_extending, index = carry
        pos = state.extenders_positions[index] + loc_add
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
    
def update_all_wiring_components(i, carry):
    state, static_params = carry
    new_state = state
        
    # Wires
    # new_state = update_wires(new_state, static_params)
    new_state = jax.lax.cond(
        jnp.sum(new_state.wires_mask) > 0,
        lambda _: update_wires(new_state, static_params),
        lambda _: new_state,
        operand=None
    )

    # Logic_gates
    new_state = jax.lax.cond(
        jnp.sum(new_state.logic_gates_mask) > 0,
        lambda _: update_logic(new_state, static_params),
        lambda _: new_state,
        operand=None
    )
    # new_state = update_logic(new_state, static_params)

    return new_state, static_params

def update_wiring_max(state, static_params, n=1):
    new_state = state
    
    new_state, _ = jax.lax.fori_loop(0, n, update_all_wiring_components, (state, static_params))
    # Extenders
    new_state = jax.lax.cond(
        jnp.sum(new_state.extenders_mask) > 0,
        lambda _: update_extenders(new_state, static_params),
        lambda _: new_state,
        operand=None
    )
    # new_state = update_extenders(new_state, static_params)
    # new_state = update_wiring_achievements(new_state, static_params)
    # _, new_outputs = jax.lax.scan(retrieve_output, state, jnp.arange(static_params.max_outputs))
    # new_state = new_state.replace(old_outputs=new_outputs)
    return new_state

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
    new_state = jax.lax.cond(
        new_state.achievements[Achievement.ACTIVATE_OUTPUT.value],
        lambda _: new_state,
        lambda _: activate_output(new_state, static_params),
        operand=None
    )

    new_state = jax.lax.cond(
        new_state.achievements[Achievement.ACTIVATE_LOGIC_GATE.value],
        lambda _: new_state,
        lambda _: activate_logic_gate(new_state, static_params),
        operand=None
    )

    new_state = jax.lax.cond(
        new_state.achievements[Achievement.HALF_ADDER.value],
        lambda _: new_state,
        lambda _: half_adder(new_state, static_params),
        operand=None
    )
    # new_state, reward = jax.lax.cond(
    #     new_state.achievements[Achievement.HALF_ADDER.value],
    #     lambda _: new_state, reward,
    #     lambda _: half_adder_curriculum(new_state, static_params),
    #     operand=None
    # )
    new_state = jax.lax.cond(
        new_state.achievements[Achievement.FULL_ADDER.value],
        lambda _: new_state,
        lambda _: full_adder(new_state, static_params),
        operand=None
    )

    new_state = jax.lax.cond(
        new_state.achievements[Achievement.BIN_TO_GRAY.value],
        lambda _: new_state,
        lambda _: bin_to_gray(new_state, static_params),
        operand=None
    )

    new_state = jax.lax.cond(
        new_state.achievements[Achievement.TRAP.value],
        lambda _: new_state,
        lambda _: trap(new_state, static_params),
        operand=None
    )

    new_state = jax.lax.cond(
        new_state.achievements[Achievement.DOOR.value],
        lambda _: new_state,
        lambda _: door(new_state, static_params),
        operand=None
    )


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


def half_adder_curriculum(state, static_params):
    input_A = retrieve_input(state, 0)[1]
    input_B = retrieve_input(state, 1)[1]
    output_sum = retrieve_output(state, 0)[1]
    output_carry = retrieve_output(state, 1)[1]
    inputs = jnp.array([input_A, input_B, output_sum, output_carry], dtype=bool)
    stage = state.half_adder_curriculum_stage
    # jax.debug.print("stage: {}", stage)

    def stage0(state):
        condition = jnp.sum(state.inputs_mask) == 2
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 0.5, 0.0)
        return state.replace(half_adder_curriculum_stage=new_stage), reward
    

    def stage1(state):
        condition = jnp.any(state.logic_gates_type == 3) # XOR
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 1.0, 0.0)
        return state.replace(half_adder_curriculum_stage=new_stage), reward

    def stage2(state):
        condition = jnp.any(state.logic_gates_type == 1) # AND
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 1.0, 0.0)
        return state.replace(half_adder_curriculum_stage=new_stage), reward
    
    def stage3(state):
        condition = state.achievements[Achievement.ACTIVATE_INPUT.value]
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 0.5, 0.0)
        return state.replace(half_adder_curriculum_stage=new_stage), reward
    
    def stage4(state):
        condition = state.achievements[Achievement.ACTIVATE_LOGIC_GATE.value]
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 1.0, 0.0)
        return state.replace(half_adder_curriculum_stage=new_stage), reward
    
    def stage5(state):
        condition = jnp.sum(state.outputs_mask) == 2
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 0.5, 0.0)
        return state.replace(half_adder_curriculum_stage=new_stage), reward
    
    def stage6(state):
        condition = state.achievements[Achievement.ACTIVATE_OUTPUT.value]
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 1.0, 0.0)
        return state.replace(half_adder_curriculum_stage=new_stage), reward
    
    def stage7(state):
        XOR_index = jnp.argmax(state.logic_gates_type == 3)
        condition = state.logic_gates_power[XOR_index]
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 1.0, 0.0)

        new_achievements = state.achievements.at[Achievement.STAGE7.value].set( # this will not be in the final product, just to see which stage it reaches
            jnp.logical_or(
                state.achievements[Achievement.STAGE7.value], condition
            )
        )
        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward
    
    def stage8(state):
        AND_index = jnp.argmax(state.logic_gates_type == 1)
        condition = state.logic_gates_power[AND_index]
        new_stage = jax.lax.select(condition, stage + 1, stage)
        reward = jax.lax.select(condition, 1.0, 0.0)

        new_achievements = state.achievements.at[Achievement.STAGE8.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE7.value], condition
            )
        )
        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward

    def stage9(state):
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

        new_achievements = state.achievements.at[Achievement.STAGE9.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE9.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward

    def stage10(state):
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

        new_achievements = state.achievements.at[Achievement.STAGE10.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE10.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward

    def stage11(state):
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

        new_achievements = state.achievements.at[Achievement.STAGE11.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE11.value], condition
            )
        )

        return state.replace(half_adder_curriculum_stage=new_stage, achievements=new_achievements), reward
    
    def stage12(state):
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

        new_achievements = state.achievements.at[Achievement.STAGE12.value].set(
            jnp.logical_or(
                state.achievements[Achievement.STAGE12.value], condition
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
    stages = [stage0, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, 
              stage10, stage11, stage12, complete]

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