import streamlit as st

import os
import state
import components.carla.config as carla_config
import components.data.functional as data_fn
import components.metrics.functional as metrics_fn

from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
from leaderboard.utils.statistics_manager import StatisticsManager

def connect_to_carla(route_path, route_num, scenario_path, agent_name):
    print('Connect button was clicked')

    st.session_state.carla_args = carla_config.get_carla_args(route_path, route_num, scenario_path, agent_name.lower())
    st.session_state.carla_statistics_manager = StatisticsManager()
    st.session_state.carla_leaderboard_evaluator = LeaderboardEvaluator(st.session_state.carla_args, st.session_state.carla_statistics_manager)
    ri = st.session_state.carla_leaderboard_evaluator.setup_route_and_scenario(st.session_state.carla_args)
    st.session_state.carla_leaderboard_evaluator.manager._running = True
    dense_traj = st.session_state.carla_leaderboard_evaluator.get_dense_trajectory()

    #print('DENSE TRAJ: ', dense_traj)

    #st.session_state.carla_driving_metrics = metrics_fn.DrivingMetrics(dense_traj, None, route_path)
    #potential_static_collisions, potential_pedestrian_collisions, potential_vehicle_collisions = st.session_state.carla_leaderboard_evaluator.get_total_potential_collisions()
    
    st.session_state.carla_leaderboard_evaluator._get_total_potential_collisions()
    potential_vehicle_collisions = st.session_state.carla_leaderboard_evaluator.vehicles
    potential_pedestrian_collisions = st.session_state.carla_leaderboard_evaluator.pedestrians
    potential_static_collisions = []
    st.session_state.carla_driving_metrics = metrics_fn.DrivingMetrics(dense_traj, potential_static_collisions, potential_vehicle_collisions, potential_pedestrian_collisions)
    
    ### TEMPORARILY COMMENTED TO ALLOW WORKING ON LAB PC! Needed for working with carlaviz ###
    if os.environ['USE_CARLAVIZ']=='True':
        st.session_state.carla_leaderboard_evaluator.manager.set_carla_visualizer(dense_traj)
    ### TEMPORARILY COMMENTED TO ALLOW WORKING ON LAB PC! ###

    #if agent_name.lower() != 'expert':
    #    st.session_state.carla_expert_stats = metrics_fn.read_expert_data()
    #    st.session_state.carla_driving_metrics = metrics_fn.DrivingMetrics(dense_traj, st.session_state.carla_expert_stats['speed'], route_path)
    #    st.session_state.carla_leaderboard_evaluator.manager.set_carla_visualizer(dense_traj)

    data_fn.get_data_from_carla()
    
    st.session_state.carla_connected = True


def disconnect_from_carla():
    print('Disconnect button was clicked')
    le = st.session_state.carla_leaderboard_evaluator
    le.__del__()
    #del le
    print('Leaderboard Eval object deleted')
    state.reset_carla_state()
    st.session_state.carla_connected = False