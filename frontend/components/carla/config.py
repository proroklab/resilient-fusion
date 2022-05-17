import argparse
import os

MODEL_NAME_TO_FILE = {
    'transfuser': 'transfuser_agent.py',
    'auto pilot': 'auto_pilot.py',
    'late fusion': 'late_fusion_agent.py',
    'conditional imitation learning': 'cilrs_agent.py',
    'expert': 'auto_pilot.py',
    'route following': 'route_following_agent.py'
    #'our model v1'
}

MODEL_NAME_TO_WEIGHTS_FOLDER = {
    'transfuser': 'transfuser',
    'expert': '',
    'route following': '',
    'late fusion': 'late_fusion',
    'conditional imitation learning': 'cilrs',
    #'our model v1'
}

def get_carla_args(route_path, route_num, scenario_path, agent_name):
    args = argparse.Namespace()
    args.host = 'localhost'
    args.port = '2000'
    args.trafficManagerPort = '8000'
    args.trafficManagerSeed = '0'
    args.debug = 0
    args.timeout = '600.0'
    #args.routes = 'leaderboard/data/validation_routes/routes_town05_short.xml'
    #args.routes = 'backend/leaderboard/data/additional_routes/routes_town03_long.xml'
    #args.routes = 'backend/leaderboard/data/{}'.format(route_path)
    args.routes = '/'.join((os.environ['LEADERBOARD_ROUTE_ROOT'], route_path))
    #args.scenarios = 'backend/leaderboard/data/scenarios/no_scenarios.json'
    #args.scenarios = 'backend/leaderboard/data/scenarios/{}'.format(scenario_path)
    args.scenarios = '/'.join((os.environ['LEADERBOARD_SCENARIO_ROOT'], scenario_path))
    args.repetitions = 2
    #args.agent = 'leaderboard/team_code/auto_pilot.py'
    #args.agent = 'backend/leaderboard/team_code/transfuser_agent.py'
    #args.agent = 'backend/leaderboard/team_code/{}'.format(MODEL_NAME_TO_FILE[agent_name])
    args.agent = '/'.join((os.environ['LEADERBOARD_AGENT_ROOT'], MODEL_NAME_TO_FILE[agent_name]))
    args.resume = False
    args.checkpoint = 'backend/results/sample_result.json'
    #args.agent_config = 'backend/model/transfuser'
    #args.agent_config = 'backend/model/{}'.format(MODEL_NAME_TO_WEIGHTS_FOLDER[agent_name])
    args.agent_config = '/'.join((os.environ['LEADERBOARD_AGENT_MODEL_WEIGHTS_ROOT'], MODEL_NAME_TO_WEIGHTS_FOLDER[agent_name]))
    args.track = 'SENSORS'
    args.record = ''

    args.route_num = route_num

    return args