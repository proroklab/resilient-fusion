import streamlit as st

import pandas as pd
import pydeck

def display_image(col, image, caption=None):
	col.image(image)
	if caption is not None:
		col.caption(caption)

def display_point_cloud(col, point_cloud, caption=None):
	point_cloud_df = pd.DataFrame(data=point_cloud, columns=["x", "y", "z"])
	
	point_cloud_layer = pydeck.Layer(
		"PointCloudLayer",
		data=point_cloud_df,
		get_position=["x", "y", "z"],
		get_normal=[0, 0, 15],
		auto_highlight=True,
		point_size=2,
	)
	point_cloud_deck = pydeck.Deck(point_cloud_layer, width='100%', height=500, map_provider=None)
	
	col.pydeck_chart(point_cloud_deck, use_container_width=True)
	if caption is not None:
		col.caption(caption)