# export SCENARIOS=${OP_BRIDGE_ROOT}/data/all_towns_traffic_scenarios_public.json
# export ROUTES=${OP_BRIDGE_ROOT}/data/routes_devtest.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=0
# export TEAM_AGENT=${OP_BRIDGE_ROOT}/op_bridge/op_ros2_agent.py
# export TEAM_AGENT=npc_agent_behavior.py

# export TEAM_AGENT=/home/amin/carla/carla_garage/team_code/sensor_agent.py
# export TEAM_CONFIG=/home/amin/carla/carla_garage/pretrained_models

# export TEAM_AGENT=../Bench2DriveZoo/team_code/vad_b2d_agent.py
# export TEAM_CONFIG=../Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+/mnt/data/wtc/Bench2DriveZoo/ckpts/vad_b2d_base.pth

export TEAM_AGENT=../Bench2DriveZoo/team_code/uniad_b2d_agent.py
export TEAM_CONFIG=../Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+/mnt/data/wtc/Bench2DriveZoo/ckpts/uniad_base_b2d.pth

# export TEAM_AGENT=../LMDrive/leaderboard/team_code/lmdriver_agent_xosc.py
# export TEAM_CONFIG=../LMDrive/leaderboard/team_code/lmdriver_config.py

# export TEAM_CONFIG=/home/amin/carla/carla_garage/pretrained_models/test


# export TEAM_AGENT=/mnt/data/simulator/scenario_util/op_carla_local/op_bridge/leaderboard/npc_agent1.py

# export TEAM_AGENT=../transfuser/team_code_transfuser/submission_agent.py
# export TEAM_CONFIG=../transfuser/model_ckpt/transfuser

export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${OP_BRIDGE_ROOT}":${PYTHONPATH}

export PYTHONPATH=$PYTHONPATH:/mnt/data/Bench2Drive/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
# export CHECKPOINT_ENDPOINT=${OP_BRIDGE_ROOT}/results.json
# export CHALLENGE_TRACK_CODENAME=MAP
export AGENT_ROLE_NAME="ego_vehicle"
export OP_BRIDGE_MODE="leaderboard"

export OPEN_SCENARIO="Scene_leaderboard/Overtaking/AccidentWarningObjectDetection.xosc"


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


