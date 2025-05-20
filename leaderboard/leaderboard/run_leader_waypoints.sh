# export SCENARIOS=${OP_BRIDGE_ROOT}/data/all_towns_traffic_scenarios_public.json
# export ROUTES=${OP_BRIDGE_ROOT}/data/routes_devtest.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=0
# export TEAM_AGENT=${OP_BRIDGE_ROOT}/op_bridge/op_ros2_agent.py
# export TEAM_AGENT=npc_agent_behavior.py

# export TEAM_AGENT=/home/amin/carla/carla_garage/team_code/sensor_agent.py
# export TEAM_CONFIG=/home/amin/carla/carla_garage/pretrained_models

# export TEAM_AGENT=/mnt/data/wtc/Bench2DriveZoo/team_code/vad_b2d_agent.py
# export TEAM_CONFIG=/mnt/data/wtc/Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+/mnt/data/wtc/Bench2DriveZoo/ckpts/vad_b2d_base.pth

export TEAM_AGENT=/mnt/data/wtc/Bench2DriveZoo/team_code/uniad_b2d_agent.py
export TEAM_CONFIG=/mnt/data/wtc/Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+/mnt/data/wtc/Bench2DriveZoo/ckpts/uniad_base_b2d.pth

# export TEAM_AGENT=/data1/wtc/LMDrive/leaderboard/team_code/lmdriver_agent_xosc.py
# export TEAM_CONFIG=/data1/wtc/LMDrive/leaderboard/team_code/lmdriver_config.py

# export TEAM_CONFIG=/home/amin/carla/carla_garage/pretrained_models/test


# export TEAM_AGENT=/mnt/data/simulator/scenario_util/op_carla_local/op_bridge/leaderboard/npc_agent1.py

# export TEAM_AGENT=/mnt/data/wtc/transfuser/team_code_transfuser/submission_agent.py
# export TEAM_CONFIG=/mnt/data/wtc/transfuser/model_ckpt/transfuser

export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${OP_BRIDGE_ROOT}":${PYTHONPATH}

export PYTHONPATH=$PYTHONPATH:/mnt/data/Bench2Drive/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
# export CHECKPOINT_ENDPOINT=${OP_BRIDGE_ROOT}/results.json
# export CHALLENGE_TRACK_CODENAME=MAP
export AGENT_ROLE_NAME="ego_vehicle"
export OP_BRIDGE_MODE="leaderboard"
# export OPEN_SCENARIO="/mnt/data/carla_safebench/scenario_runner/srunner/examples/FollowLeadingVehicle.xosc"
# export OPEN_SCENARIO="/mnt/data/carla_safebench/scenario_runner/srunner/examples/LaneChangeSimple.xosc"
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
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/test/test3.xosc"
# export OPEN_SCENARIO="/mnt/data/tcwang/An_opposing_vehicle_passes_through_the_lane.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/Ghost_probe.xosc"
# export OPEN_SCENARIO="/mnt/data/tcwang/Ghost_probe.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/Straight_road_vehicle_changes_lanes_and_encounters_a_slanted_cone.xosc"
# export OPEN_SCENARIO="/mnt/data/simulator/scenario_util/op_carla_local/op_bridge/leaderboard/test.xosc"

# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/SimpleScene/Left_deviation_test_curve.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/SimpleScene/Speed_limit_sign_recognition_and_response.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/MediumScene/Target_car_stationary_straight_lane.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/MediumScene/Straight_road_vehicle_changes_lanes.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/MediumScene/Target_car_cuts_into_straight_lane_at_high_speed.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/MediumScene/Target_vehicle_decelerates_straight_lane.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/MediumScene/The_pedestrian_walks_along_the_road.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/MediumScene/The_two-wheeler_crosses_the_road.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/RiskMigration/Ghost_probe.xosc"
# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/RiskMigration/TwowheelerRetrograde.xosc"

# export OPEN_SCENARIO="/mnt/data/cjc/Scene/agent_test/Cruising_corners.xosc"
# export OPEN_SCENARIO="Stop_and_go.xosc"

# export OPEN_SCENARIO="/data1/wtc/Scene/original_scene/original_scene/ExitZoneLeading.xosc"
# export OPEN_SCENARIO="/data1/wtc/Scene/original_scene/original_scene/StraightLaneDeceleration.xosc"

# export OPEN_SCENARIO="/data1/wtc/Scene/original_scene/original_scene/OpposingPass.xosc"
# export OPEN_SCENARIO="/data1/wtc/Scene/original_scene/original_scene/StationaryStraightObstacle.xosc"
# export OPEN_SCENARIO="/data1/wtc/Scene/original_scene/original_scene/StraightLaneChanging.xosc"
# export OPEN_SCENARIO="/data1/wtc/Scene/original_scene/original_scene/StraightLaneCut-in.xosc"
# export OPEN_SCENARIO="/data1/wtc/Scene/original_scene/original_scene/StraightLanePost-cutout.xosc"
# export OPEN_SCENARIO="/data1/wtc/Scene/original_scene/original_scene/Turn-Left.xosc"
# export OPEN_SCENARIO="/data1/wtc/Scene/original_scene/original_scene/Turn-Right.xosc"
export OPEN_SCENARIO="/data1/wtc/Scene/original_scene/original_scene/SuddenPedestrianCrossing.xosc"


python3 leaderboard_xosc_test_waypoints.py \
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


