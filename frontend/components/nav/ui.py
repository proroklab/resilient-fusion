import streamlit as st

import components.nav.functional as nav_fn

def display_prev_button(col, label='< Prev'):
	current_page = st.session_state.page_selected['page_name']
	#print('CURR: ', current_page)

	col.button(label=label, 
				on_click=nav_fn.get_prev_page,
				args=(current_page,)
			)

def display_next_button(col, label='Next >'):
	current_page = st.session_state.page_selected['page_name']

	col.button(label=label, 
				on_click=nav_fn.get_next_page, 
				args=(current_page,)
			)

def display_next_sample_button(col):
	current_page = st.session_state.page_selected['page_name']

	col.button(label='Next Sample', 
				on_click=nav_fn.get_next_sample,
				args=(current_page,)
			)

'''
def display_modify_sample_button(col):
	current_page = st.session_state.page_selected['page_name']

	col.button(label='Modify Sample', 
				on_click=nav_fn.get_next_page, 
				args=(current_page,)
			)
'''