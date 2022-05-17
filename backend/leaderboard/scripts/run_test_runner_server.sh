#!/bin/bash

rm ./results/sample_result.json

#export CARLA_ROOT=carla
export CARLA_ROOT=/opt/carla-simulator
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
#export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export PYTHONPATH=$PYTHONPATH:carlaviz

export LEADERBOARD_ROOT=leaderboard
#export CHALLENGE_TRACK_CODENAME=SENSORS
#export PORT=2000 # same as the carla server port
#export TM_PORT=8000 # port for traffic manager, required when spawning multiple servers/clients
#export DEBUG_CHALLENGE=0
#export REPETITIONS=1 # multiple evaluation runs


#export ROUTES=leaderboard/data/additional_routes/routes_town03_long.xml

#export TEAM_AGENT=leaderboard/team_code/transfuser_agent.py # agent
#export TEAM_CONFIG=model/transfuser # model checkpoint, not required for expert
#export CHECKPOINT_ENDPOINT=results/sample_result.json # results file
#export SCENARIOS=leaderboard/data/scenarios/no_scenarios.json
#export SAVE_PATH=data/transfuser # path for saving episodes while evaluating
#export RESUME=True

python3 test_runner_server.py
#python3 server.py 


#export FLASK_APP=carla_api.py
#export FLASK_ENV=development
#flask run