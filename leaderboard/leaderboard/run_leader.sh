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
export OPEN_SCENARIO="Scene_leaderboard/Overtaking/AccidentWarningObjectDetection.xosc"
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


