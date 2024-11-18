import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('Non-invasive semi-automatic inspection system for lead rubber bearings (LRB)')

st.info('This is app predict the level of damage of lead rubber bearings')

with st.expander('Geometric characteristics of LRB'):
  st.write('**Raw data**')
  #df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  #df

  st.write('**X**')

  st.write('**y**')


with st.expander('Mechanical Propierties of LRB Materials'):
  #st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
  st.write('**X**')

# Input features
with st.sidebar:
  st.header('Input LRB propierties (Geometrical and mechanical)')
  #Input propierties
  Di_mm = st.slider('LRB Diameter (mm)', 749.3, 952.5, 850.9,0.01) #1
  Ht_mm = st.slider('LRB High (mm)', 407.64, 420.34, 410.00,0.01) #2
  Dl_mm = st.slider('Lead Core Diameter (mm)', 114.3, 203.2, 165.1,0.01) #3
  W_kg = st.slider('LRB Weight (kg)', 808, 1456, 1068,1) #4
  e_pc_mm = st.slider('Thickness of Exterior Rubber Layer (mm)', 18, 20, 19,1) #5
  cc_und = st.slider('Number Rubber Layers', 10, 30, 28,1) #6
  e_cc_mm = st.slider('Thickness of Interior Rubber Layers (mm)', 5, 8, 10,1) #7
  cs_und = st.slider('Number Steel Layers', 10, 30, 27,1) #8
  e_cs_mm= st.slider('Thickness of Interior Steel Layers (mm)', 2.00, 4.00, 3.04,0.01) #9
  Fy_ac_mpa=st.slider('Yield Strees Steel (Mpa)', 165, 345, 250,1) #10
  E_cau_mpa= st.slider('Vertical Elastic Modulus (Mpa)', 1.00, 2.00, 1.44,0.01) #11
  G_cau_mpa = st.slider('Shear Modulus of Rubber (Mpa)', 0.300, 0.500, 0.392,0.001) #12
  Fycort_pb_mpa = st.slider('Yield Shear Stress of Lead (Mpa)', 5.00, 10, 8.33,0.01) #13

  #'Di (mm)','Ht (mm)','Dl (mm)','W (kg)','e_pc (mm)','#cc','e_cc (mm)'
  #'#cs','e_cs (mm)','Fy_ac (MPa)','E_cau (MPa)','G_cau (MPa)','Fycort_pb (MPa)'
  # Create a DataFrame for the input features
  """data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)"""

with st.expander('Input features'):
  st.write('**Input Signal 1**')
  S1=st.file_uploader("Choose a file in .txt format")
  st.write('**Input Signal 2**')
  S2=st.file_uploader("Choose a file in .txt format")



# Data preparation
# Encode X
encode = ['island', 'sex']
#df_penguins = pd.get_dummies(input_penguins, prefix=encode)

#X = df_penguins[1:]
#input_row = df_penguins[:1]

# Encode y
"""target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input penguin)**')
  input_row
  st.write('**Encoded y**')
  y
"""

# Model training and inference
## Train the ML model
#clf = RandomForestClassifier()
#clf.fit(X, y)

## Apply model to make predictions
"""prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                                 1: 'Chinstrap',
                                 2: 'Gentoo'})"""

"""# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))
"""