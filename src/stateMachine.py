##[MC:2025-03-06] Definitions for a state machine for the simulation loop

# Define states
states = {
    "INIT": 0,
    "SEARCHING_OBJECT": 1,
    "MOVING_TO_OBJECT":2,
    "GENERATING_GRASP": 3,
    "PRE_GRASPING": 4,
    "GRASPING": 5,
    "LIFTING": 6,
    "GENERATING_TRAJECTORY": 7,
    "MOVING_TO_TARGET": 8,
    "FINISHED": 9,
}

# Define events
events = {
    "NONE":0,
    "SIMULATION_STABLE": 1,
    "OBJECT_POSITION_ESTIMATED": 2,
    "REACHED_OBJECT": 3,
    "GRASP_GENERATED": 4,
    "PREGRASP_SUCCESS": 5,
    "GRASP_SUCCESS": 6,
    "OBJECT_LIFTED": 7,
    "TRAJECTORY_GENERATED": 8,
    "TARGET_REACHED": 9,

}

# Define transitions
transitions = {
    state: {event: state for event in events.values()} for state in states.values()
}

# Update specific transitions
transitions[states["INIT"]][events["SIMULATION_STABLE"]] = states["SEARCHING_OBJECT"]
transitions[states["SEARCHING_OBJECT"]][events["OBJECT_POSITION_ESTIMATED"]] = states["MOVING_TO_OBJECT"]
transitions[states["MOVING_TO_OBJECT"]][events["REACHED_OBJECT"]] = states["GENERATING_GRASP"]
transitions[states["GENERATING_GRASP"]][events["GRASP_GENERATED"]] = states["PRE_GRASPING"]
transitions[states["PRE_GRASPING"]][events["PREGRASP_SUCCESS"]] = states["GRASPING"]
transitions[states["GRASPING"]][events["GRASP_SUCCESS"]] = states["LIFTING"]
transitions[states["LIFTING"]][events["OBJECT_LIFTED"]] = states["GENERATING_TRAJECTORY"]
transitions[states["GENERATING_TRAJECTORY"]][events["TRAJECTORY_GENERATED"]] = states["MOVING_TO_TARGET"]
transitions[states["MOVING_TO_TARGET"]][events["TARGET_REACHED"]] = states["FINISHED"]
