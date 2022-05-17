import streamlit as st

import components.inference.functional as inf_fn

def display_inference_button(col, num_frames_to_process):
	col.button(label='Perform inference on {} frame(s)'.format(num_frames_to_process),
				on_click=inf_fn.set_perform_inference_flag,
			)