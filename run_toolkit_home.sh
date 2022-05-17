#!/bin/bash

rm backend/results/sample_result.json

#export CARLA_ROOT=carla
#export CARLA_ROOT=/auto/homes/sn611/saasha/carla-simulator
export CARLA_ROOT=/opt/carla-simulator
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg

export PYTHONPATH=$PYTHONPATH:frontend
export PYTHONPATH=$PYTHONPATH:backend
export PYTHONPATH=$PYTHONPATH:backend/leaderboard
export PYTHONPATH=$PYTHONPATH:backend/leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:backend/scenario_runner
export PYTHONPATH=$PYTHONPATH:backend/carlaviz
export PYTHONPATH=$PYTHONPATH:backend/advGAN

export LEADERBOARD_ROOT=backend/leaderboard
export LEADERBOARD_ROUTE_ROOT=${LEADERBOARD_ROOT}/data
export LEADERBOARD_SCENARIO_ROOT=${LEADERBOARD_ROUTE_ROOT}/scenarios
export LEADERBOARD_AGENT_ROOT=${LEADERBOARD_ROOT}/team_code
export LEADERBOARD_AGENT_MODEL_WEIGHTS_ROOT=backend/model

export GAN_MODELS_ROOT=backend/advGAN/models

export TOOLKIT_FRONTEND=frontend
export USE_CARLAVIZ=False ## can take values: True, False

export RESULTS_BASEPATH=results
export REP_NUM=0 ## experiments are repeated 10 times per route with a different seed value
#export PYTHONHASHSEED='10' ## seed values used per rep: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90

streamlit run app.py