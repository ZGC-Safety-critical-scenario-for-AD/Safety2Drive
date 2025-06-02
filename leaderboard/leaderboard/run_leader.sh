# export SCENARIOS=${OP_BRIDGE_ROOT}/data/all_towns_traffic_scenarios_public.json
# export ROUTES=${OP_BRIDGE_ROOT}/data/routes_devtest.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=0
# export TEAM_AGENT=${OP_BRIDGE_ROOT}/op_bridge/op_ros2_agent.py
export TEAM_AGENT=npc_agent.py

# export TEAM_AGENT=/home/amin/carla/carla_garage/team_code/sensor_agent.py
# export TEAM_CONFIG=/home/amin/carla/carla_garage/pretrained_models/all_towns
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${OP_BRIDGE_ROOT}":${PYTHONPATH}
# export CHECKPOINT_ENDPOINT=${OP_BRIDGE_ROOT}/results.json
# export CHALLENGE_TRACK_CODENAME=MAP
export AGENT_ROLE_NAME="ego_vehicle"
export OP_BRIDGE_MODE="leaderboard"
# export OPEN_SCENARIO="/mnt/data/carla_safebench/scenario_runner/srunner/examples/FollowLeadingVehicle.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/Regulation_and_control/Target_car_stationary_straight_lane.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/Regulation_and_control/Target_vehicle_decelerates_straight_lane.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/Regulation_and_control/Target_car_cuts_into_straight_lane_at_high_speed.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/Regulation_and_control/Target_car_cut_out_after_the_front_car_straight_lane.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/Regulation_and_control/Speed_limit_sign_recognition_and_response.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/Regulation_and_control/Crosswalk_line_recognition_and_response_with_pedestrians.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/Regulation_and_control/The_pedestrian_crosses_the_road.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/Regulation_and_control/Straight_traffic_collision.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/Regulation_and_control/Pedestrians_and_passenger_cars_moving_slowly_Straight.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/Regulation_and_control/Straight_road_traffic_accident_Cross_passenger_car.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/test/test1.xosc"
# export OPEN_SCENARIO="Stop_and_go.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/Standards_and_regulations/Target_vehicle_decelerates_straight_lane.xosc"
export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/Straight_road_vehicle_changes_lanes_and_encounters_a_slanted_cone.xosc"
python3 leaderboard_xosc_test.py \
--repetitions=${REPETITIONS} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--openscenario=${OPEN_SCENARIO} \
# --track=${CHALLENGE_TRACK_CODENAME} \
# --checkpoint=${CHECKPOINT_ENDPOINT} \
# --routes=${ROUTES} \
# --scenarios=${SCENARIOS}  \


