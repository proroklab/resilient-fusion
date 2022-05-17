import streamlit as st
import pandas as pd

def init_session_state():
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
        st.session_state.perturbed_image = None

    if 'original_lidar' not in st.session_state:
        st.session_state.original_lidar = None
        st.session_state.perturbed_lidar = None

    if 'data_source_selected' not in st.session_state:
        st.session_state.data_source_selected = 'Static dataset'
        st.session_state.prev_data_source_selected = 'Static dataset'

    if 'image_perturbation_settings' not in st.session_state:
        st.session_state.image_perturbation_settings = {}

    if 'lidar_perturbation_settings' not in st.session_state:
        st.session_state.lidar_perturbation_settings = {}

    if 'carla_args' not in st.session_state:
        st.session_state.carla_args = None

    if 'carla_leaderboard_evaluator' not in st.session_state:
        st.session_state.carla_leaderboard_evaluator = None

    if 'carla_ego_velocity' not in st.session_state:
        st.session_state.carla_ego_velocity = []

    if 'carla_ego_acceleration' not in st.session_state:
        st.session_state.carla_ego_acceleration = []

    if 'carla_ego_stats' not in st.session_state:
        st.session_state.carla_ego_stats = pd.DataFrame(columns=['acceleration', 'speed', 'x', 'y', 'heading', 'expected_heading', 'expected_speed', 'projected_x', 'projected_y'])

    if 'carla_connected' not in st.session_state:
        st.session_state.carla_connected = False

    if 'data' not in st.session_state:
        st.session_state.data = {}

    if 'perform_inference_flag' not in st.session_state:
        st.session_state.perform_inference_flag = False

    if 'inference_step_cntr' not in st.session_state:
        st.session_state.inference_step_cntr = 0

    if 'steps_since_last_perturbation' not in st.session_state:
        st.session_state.steps_since_last_perturbation = 0

    if 'num_frames_to_skip_selected' not in st.session_state:
        st.session_state.num_frames_to_skip_selected = 0

    if 'perturbation_settings_mode_selected' not in st.session_state:
        st.session_state.perturbation_settings_mode_selected = 'Exact'

    if 'results_path' not in st.session_state:
        st.session_state.results_folder_path = './'

    if 'carla_expert_stats' not in st.session_state:
        st.session_state.carla_expert_stats = pd.DataFrame(columns=['acceleration', 'speed', 'x', 'y', 'heading'])

    if 'baseline_traj_xy' not in st.session_state:
        st.session_state.baseline_traj_xy = None

    if 'carla_driving_metrics' not in st.session_state:
        st.session_state.carla_driving_metrics = None

    if 'total_routes' not in st.session_state:
        st.session_state.total_routes = 0

def reset_control_state():
    ## when switching between exact and interval based controls, delete all control information from session_state to avoid discrepencies

    if 'image_perturbation_settings' in st.session_state:
        del st.session_state['image_perturbation_settings']

    if 'lidar_perturbation_settings' in st.session_state:
        del st.session_state['lidar_perturbation_settings']

    if 'image_noise_type' in st.session_state:
        del st.session_state['image_noise_type']

    if 'image_noise_form' in st.session_state:
        del st.session_state['image_noise_form']

    if 'image_sp_amount' in st.session_state:
        del st.session_state['image_sp_amount']

    if 'image_gaussian_variance' in st.session_state:
        del st.session_state['image_gaussian_variance']

    if 'image_blur_form' in st.session_state:
        del st.session_state['image_blur_form']

    if 'image_blur_amount' in st.session_state:
        del st.session_state['image_blur_amount']

    if 'image_attack_type' in st.session_state:
        del st.session_state['image_attack_type']

    if 'image_attack_form' in st.session_state:
        del st.session_state['image_attack_form']

    if 'image_fgsm_epsilon' in st.session_state:
        del st.session_state['image_fgsm_epsilon']

    if 'image_pgd_epsilon' in st.session_state:
        del st.session_state['image_pgd_epsilon']

    if 'image_pgd_alpha' in st.session_state:
        del st.session_state['image_pgd_alpha']

    if 'image_pgd_iters' in st.session_state:
        del st.session_state['image_pgd_iters']

    if 'image_cw_epsilon' in st.session_state:
        del st.session_state['image_cw_epsilon']

    if 'image_cw_binary_search_steps' in st.session_state:
        del st.session_state['image_cw_binary_search_steps']

    if 'image_cw_iters' in st.session_state:
        del st.session_state['image_cw_iters']

    if 'image_occlusion_form' in st.session_state:
        del st.session_state['image_occlusion_form']

    if 'image_occlusion_x' in st.session_state:
        del st.session_state['image_occlusion_x']

    if 'image_occlusion_y' in st.session_state:
        del st.session_state['image_occlusion_y']

    if 'image_occlusion_width' in st.session_state:
        del st.session_state['image_occlusion_width']

    if 'image_hallucination_form' in st.session_state:
        del st.session_state['image_hallucination_form']

    if 'image_hallucination_x' in st.session_state:
        del st.session_state['image_hallucination_x']

    if 'image_hallucination_y' in st.session_state:
        del st.session_state['image_hallucination_y']

    if 'image_hallucination_size' in st.session_state:
        del st.session_state['image_hallucination_size']


    if 'lidar_noise_form' in st.session_state:
        del st.session_state['lidar_noise_form']

    if 'lidar_noise_amount' in st.session_state:
        del st.session_state['lidar_noise_amount']

    if 'lidar_attack_type' in st.session_state:
        del st.session_state['lidar_attack_type']

    if 'lidar_attack_form' in st.session_state:
        del st.session_state['lidar_attack_form']

    if 'lidar_fgsm_epsilon' in st.session_state:
        del st.session_state['lidar_fgsm_epsilon']

    if 'lidar_pgd_epsilon' in st.session_state:
        del st.session_state['lidar_pgd_epsilon']

    if 'lidar_pgd_alpha' in st.session_state:
        del st.session_state['lidar_pgd_alpha']

    if 'lidar_pgd_iters' in st.session_state:
        del st.session_state['lidar_pgd_iters']

    if 'lidar_cw_epsilon' in st.session_state:
        del st.session_state['lidar_cw_epsilon']

    if 'lidar_cw_box_constraints' in st.session_state:
        del st.session_state['lidar_cw_box_constraints']

    if 'lidar_cw_iters' in st.session_state:
        del st.session_state['lidar_cw_iters']

    st.info('All selected perturbation settings have been reset!')


def reset_control_state_if_min_frame_selected():
    if st.session_state.num_frames_to_perturb_selected == 1 and st.session_state.perturbation_settings_mode_selected == 'Interval':
        st.session_state.perturbation_settings_mode_selected = 'Exact'
        reset_control_state()

def reset_carla_state():
    if 'data' in st.session_state:
        #del st.session_state.data
        del st.session_state['data']

    if 'original_image' in st.session_state:
        #del st.session_state.original_image
        del st.session_state['original_image']

    if 'original_lidar' in st.session_state:
        #del st.session_state.original_lidar
        del st.session_state['original_lidar']

    if 'carla_args' in st.session_state:
        del st.session_state['carla_args']

    if 'carla_leaderboard_evaluator' in st.session_state:
        #del st.session_state.carla_leaderboard_evaluator
        del st.session_state['carla_leaderboard_evaluator']

    if 'carla_ego_velocity' in st.session_state:
        #del st.session_state.carla_ego_velocity
        del st.session_state['carla_ego_velocity']

    if 'carla_ego_acceleration' in st.session_state:
        #del st.session_state.carla_ego_acceleration
        del st.session_state['carla_ego_acceleration']

    if 'carla_ego_stats' in st.session_state:
        del st.session_state['carla_ego_stats']

    if 'results_folder_path' in st.session_state:
        del st.session_state['results_folder_path']

    if 'carla_expert_stats' in st.session_state:
        del st.session_state['carla_expert_stats']

    if 'baseline_traj_xy' in st.session_state:
        del st.session_state['baseline_traj_xy']

    if 'carla_driving_metrics' in st.session_state:
        del st.session_state['carla_driving_metrics']

    reset_control_state()