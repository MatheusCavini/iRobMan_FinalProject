##[MC:2025-03-06] Definitions for a state machine for the simulation loop

# Define states
states = {
    "INIT": 0,
    "SEARCHING_OBJECT": 1,
    "MOVING_TO_OBJECT":2,
    "GENERATING_GRASP": 3,
    "GRASPING": 4,
    "LIFTING": 5,
    "GENERATING_TRAJECTORY": 6,
    "MOVING_TO_TARGET": 7,
    "FINISHED": 8,
    "GENERATING_CORRECTOR":9,
    "CORRECTING_POSITION": 10,

}

# Define events
events = {
    "NONE":0,
    "SIMULATION_STABLE": 1,
    "OBJECT_POSITION_ESTIMATED": 2,
    "REACHED_OBJECT": 3,
    "GRASP_GENERATED": 4,
    "GRASP_SUCCESS": 5,
    "OBJECT_LIFTED": 6,
    "TRAJECTORY_GENERATED": 7,
    "TARGET_REACHED": 8,
    "CORRECTION_GENERATED": 9,
    "POSITION_CORRECTED": 10,

}

# Define transitions
transitions = {
    state: {event: state for event in events.values()} for state in states.values()
}

# Update specific transitions
transitions[states["INIT"]][events["SIMULATION_STABLE"]] = states["SEARCHING_OBJECT"]
transitions[states["SEARCHING_OBJECT"]][events["OBJECT_POSITION_ESTIMATED"]] = states["MOVING_TO_OBJECT"]
transitions[states["MOVING_TO_OBJECT"]][events["REACHED_OBJECT"]] = states["GENERATING_CORRECTOR"]
transitions[states["GENERATING_CORRECTOR"]][events["CORRECTION_GENERATED"]] = states["CORRECTING_POSITION"]
transitions[states["CORRECTING_POSITION"]][events["POSITION_CORRECTED"]] = states["GENERATING_GRASP"]
transitions[states["GENERATING_GRASP"]][events["GRASP_GENERATED"]] = states["GRASPING"]
transitions[states["GRASPING"]][events["GRASP_SUCCESS"]] = states["LIFTING"]
transitions[states["LIFTING"]][events["OBJECT_LIFTED"]] = states["GENERATING_TRAJECTORY"]
transitions[states["GENERATING_TRAJECTORY"]][events["TRAJECTORY_GENERATED"]] = states["MOVING_TO_TARGET"]
transitions[states["MOVING_TO_TARGET"]][events["TARGET_REACHED"]] = states["FINISHED"]

