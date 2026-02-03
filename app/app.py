import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load model
with open('pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataframe
with open('df.pkl', 'rb') as f:
    df = pickle.load(f)
st.title('Real State')
# st.write(df)
st.sidebar.title('Real State')
st.sidebar.radio('home',('home','price prediction','recomendation','anaylis'))
# bedRoom, bathroom, balcony, agePossession, built_up_area, servant room, store
# room, furnishing_type,luxury_cat,floor catt
property_type=st.selectbox('propertytype',df['property_type'].unique())
sector_type=st.selectbox('sectortype',df['sector'].unique())
# bedroom_type=float(st.selectbox('beadroomtype',df['bedRoom'].unique()))
bedroom_type=st.slider('badroomtype',0,10)
# bathroom_type=float(st.selectbox('bathroomtype',df['bathroom'].unique()))
bathroom_type=st.slider('bathroom type',0,10)

balcony_type=st.selectbox('balconytype',df['balcony'].unique())
age_type=st.selectbox('agetype',df['agePossession'].unique())
built_in=float(st.number_input("built up area", min_value=0, max_value=10000, value=25))
servant_type=st.selectbox('servant room',['Yes','No'])
store_type=st.selectbox('store room',['yes','No'])
furnish_type=st.selectbox('furnish type',df['furnishing_type'].unique())
luxury_type=st.selectbox('luxuryytype',df['luxury_cat'].unique())
floor_type=st.selectbox('floortype',df['floor catt'].unique())
if st.button("Predict Price"):
    if servant_type=='Yes':
        servant_type=1
    else:
        servant_type=0
    if store_type=='Yes':
        store_type=1
    else:
        store_type=0
    data=np.array([property_type,sector_type,bedroom_type,bathroom_type,balcony_type,age_type,built_in,servant_type,store_type,furnish_type,luxury_type,floor_type]).reshape(1,12)
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type','luxury_cat','floor catt']
    one_df = pd.DataFrame(data, columns=columns)
    st.title(np.expm1(model.predict(one_df)))