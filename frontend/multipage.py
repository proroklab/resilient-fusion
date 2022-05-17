import streamlit as st

import os
import math
import numpy as np
import copy
import argparse

import torch
torch.cuda.empty_cache()
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from state import *
import components.carla.connect as carla_fn

import xml.etree.ElementTree as ET

def get_available_route_nums():
    route_file = '/'.join((os.environ['LEADERBOARD_ROUTE_ROOT'], st.session_state.carla_route_path_selected))
    root = ET.parse(route_file).getroot()
    print('Root: ', root)
    print('Total available routes: ', len(root))
    st.session_state.total_routes = len(root)


class MultiPage: 
    def __init__(self) -> None:
        self.pages = []
        init_session_state()
    
    def add_page(self, page_name, func) -> None: 
        self.pages.append(
            {
                "page_name": page_name, 
                "function": func
            }
        )

    def run(self):

        ###st.write(st.session_state)

        st.sidebar.title('<Toolkit name here>')

        st.session_state.pages = self.pages

        page = st.sidebar.radio(
            label='Navigate to:',
            options=self.pages,
            format_func=lambda page: page['page_name'],
            key='page_selected'
            )

        num_frames_to_perturb = st.sidebar.number_input(
            label='Number of frames to perform inference on',
            min_value=1, max_value=10000, value=1, step=1,
            key='num_frames_to_perturb_selected', 
            on_change=reset_control_state_if_min_frame_selected,
            )

        if num_frames_to_perturb > 1:
            num_frames_to_skip = st.sidebar.number_input(
            label='Number of frames to skip between perturbations',
            min_value=0, max_value=100, value=0, step=1,
            key='num_frames_to_skip_selected',
            )
            
            perturbation_settings_mode = st.sidebar.selectbox(
                label='Perturbation Settings Mode',
                options=('Exact', 'Interval'),
                key='perturbation_settings_mode_selected',
                on_change=reset_control_state,
                )
        
        data_source = st.sidebar.selectbox(
            label='Data Source',
            options=('Static dataset', 'CARLA',),
            key='data_source_selected',
            )

        '''
        processing_mode = st.sidebar.selectbox(
            label='Mode of processing',
            options=('Single Frame', 'Multiple Frames',),
            key='processing_mode_selected',
            )
        '''

        model = st.sidebar.selectbox(
            label='Model to use',
            options=('Expert', 'TransFuser', 'Conditional Imitation Learning', 'Late Fusion'), #, 'Our Model v1'),
            key='model_selected',
            )

        if data_source == 'CARLA':
            carla_route_path = st.sidebar.selectbox(
                label='Route',
                options=('', 'validation_routes/routes_town05_short.xml', 'validation_routes/routes_town05_tiny.xml'),
                key='carla_route_path_selected',
                on_change=get_available_route_nums,
                )

            
            carla_route_num = st.sidebar.selectbox(
                label='Route number',
                options=list(range(st.session_state.total_routes)),
                key='carla_route_num_selected',
                )

            carla_scenario_path = st.sidebar.selectbox(
                label='Scenario',
                options=('no_scenarios.json', 'town05_all_scenarios.json'),
                key='carla_scenario_path_selected',
                )

            #st.text('Mode: {}, Model: {}, Route: {}, Scenario: {}, Carla connected: {}'.format(st.session_state.perturbation_settings_mode_selected, model, carla_route_path, carla_scenario_path, st.session_state.carla_connected))

            if carla_route_num is not None:
                RESULTS_BASEPATH = os.environ['RESULTS_BASEPATH']
                REP_NUM = os.environ['REP_NUM']

                results_path_route = carla_route_path.split('.')[0].replace('/', '_')
                results_path_scenario = carla_scenario_path.split('.')[0].replace('/', '_')
                results_path_agent = model.lower()

                st.session_state.results_folder_path = '/'.join((RESULTS_BASEPATH, results_path_route, results_path_scenario, 'route_idx_{}'.format(carla_route_num), results_path_agent, REP_NUM))
                os.makedirs(st.session_state.results_folder_path, exist_ok=True)

            if st.session_state.carla_connected:
                carla_disconnect_btn = st.sidebar.button('Disconnect', on_click=carla_fn.disconnect_from_carla)
            else:
                carla_connect_btn = st.sidebar.button('Connect', on_click=carla_fn.connect_to_carla, args=(carla_route_path, carla_route_num, carla_scenario_path, model,)) 

        page['function']()
