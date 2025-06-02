 #!/bin/bash

# export TEAM_AGENT=$LEADERBOARD_ROOT/leaderboard/autoagents/human_agent.py
export TEAM_AGENT=autoagents/autonomous_agent.py

export ROUTES=$LEADERBOARD_ROOT/data/routes_devtest.xml
export ROUTES_SUBSET=0
export REPETITIONS=1

export DEBUG_CHALLENGE=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT="${LEADERBOARD_ROOT}/results.json"
export RECORD_PATH=
export RESUME=

#!/bin/bash

python3 leaderboard_xosc.py \
--agent=${TEAM_AGENT} \
--scenarios=/mnt/data/cjc/Scene/Standards_and_regulations/Two-wheeler_cuts_out_in_front_of_a_walking_pedestrian.xosc