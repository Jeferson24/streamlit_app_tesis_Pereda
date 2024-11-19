import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tsfel
from sklearn.preprocessing import StandardScaler

import requests
from io import BytesIO
from tensorflow.keras.models import load_model

# URL del modelo en GitHub (asegúrate de usar el enlace raw)
url = "https://github.com/Jeferson24/streamlit_app_tesis_Pereda/raw/master/model_FCDN/N/mi_modelo.h5"

# Descargar el archivo del modelo
response = requests.get(url)

# Verifica si la descarga fue exitosa (código 200)
if response.status_code == 200:
    # Cargar el modelo directamente desde la respuesta (BytesIO)
    model_FCDNN = load_model(BytesIO(response.content))
    print("Modelo cargado exitosamente")
else:
    print("Error al descargar el modelo:", response.status_code)


###### Funciones######
from scipy.signal import butter, filtfilt
import glob
import zipfile
import time

#highpass filtro
def high_pass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

#lowpass filtro
def low_pass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

#bandpass filtro
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
#

DF_evaluar=pd.DataFrame()

###### Feature Extraction #######
def load_dataset(S1,S2):   #Columnas S1:   'tiempo(s)' | 'N-S' | 'E-W' | 'U-D'
  #Lectura para el Aislador A1
  df_S1= pd.read_csv((S1),sep=' ') #Sótano 1
  df_S2= pd.read_csv((S2),sep=' ') #Sótano 2
  df_S1 = df_S1.loc[:, ~df_S1.columns.str.contains('^Unnamed')] #Eliminar columna unnamed
  df_S2 = df_S2.loc[:, ~df_S2.columns.str.contains('^Unnamed')] #Eliminar columna unnamed

  #APLICACIÓN DEL BAND PASS FILTER A LA SEÑAL
  fs = 100  # Sample rate

  #Bandpass filter
  lowcut=0.25
  highcut=20

  # Crear la lista de tiempo con un paso de 0.01s
  time=np.arange(0,len(df_S1))/fs
  time=[round(tiempo, 2) for tiempo in time]

  #Constantes de conversión V a gal (0.01 m/s^2)
  P=float(1/8)
  U=float(1/8)
  S=float(1/2)

  #Aplicación de la conversión
  df_S1["tiempo(s)"] = time
  df_S1['N-S_filtered'] = butter_bandpass_filter(df_S1['N-S']*(P)*(U)*(S), lowcut, highcut, fs)
  df_S1['E-W_filtered'] = butter_bandpass_filter(df_S1['E-W']*(P)*(U)*(S), lowcut, highcut, fs)
  df_S1['U-D_filtered'] = butter_bandpass_filter(df_S1['U-D']*(P)*(U)*(S), lowcut, highcut, fs)

  df_S2["tiempo(s)"] = time
  df_S2['N-S_filtered'] = butter_bandpass_filter(df_S2['N-S']*(P)*(U)*(S), lowcut, highcut, fs)
  df_S2['E-W_filtered'] = butter_bandpass_filter(df_S2['E-W']*(P)*(U)*(S), lowcut, highcut, fs)
  df_S2['U-D_filtered'] = butter_bandpass_filter(df_S2['U-D']*(P)*(U)*(S), lowcut, highcut, fs)

  df_filtered_S1 = df_S1[['tiempo(s)', 'N-S_filtered', 'E-W_filtered', 'U-D_filtered']]
  df_filtered_S2 = df_S2[['tiempo(s)', 'N-S_filtered', 'E-W_filtered', 'U-D_filtered']]


  """
  plt.rcParams['axes.facecolor'] = 'white'
  plt.plot(df_S1['tiempo(s)'], df_S1['N-S_filtered'], label='Canal N-S',c='darkorange',linewidth=0.5)
  plt.ylabel('Aceleración (gal)', fontsize=16)
  plt.grid(True, linestyle='-', linewidth=0.5, color='gray')
  plt.yticks(np.arange(-0.001, 0.0013, 0.0005), fontsize=12)
  # Get current axis limits
  x_min, x_max = plt.xlim()
  y_min, y_max = plt.ylim()
  # Expand the limits
  plt.xlim(0 , 200)  # Expand x-axis by 10 units on both sides
  plt.ylim(y_min, y_max)  # Expand y-axis by 50 units on both sides
  plt.legend()
  plt.show()


  plt.plot(df_S1['tiempo(s)'], df_S1['E-W_filtered'], label='Canal E-W',c='steelblue',linewidth=0.5)
  plt.ylabel('Aceleración (gal)', fontsize=16)
  plt.grid(True, linestyle='-', linewidth=0.5, color='gray')
  plt.yticks(np.arange(-0.001, 0.0013, 0.0005), fontsize=12)
  # Get current axis limits
  x_min, x_max = plt.xlim()
  y_min, y_max = plt.ylim()
  # Expand the limits
  plt.xlim(0 , 200)  # Expand x-axis by 10 units on both sides
  plt.ylim(y_min, y_max)  # Expand y-axis by 50 units on both sides
  plt.legend()
  plt.show()
  """

  #SEÑAL S1
  acelerometer_Horizontal_1=df_filtered_S1['N-S_filtered'] * 0.7 + df_filtered_S1['E-W_filtered'] * 0.3
  acelerometer_Vertical_1=df_filtered_S1['U-D_filtered']
  #Añadiendo columnas a la señal S1
  df_filtered_S1["Horizontal"]=acelerometer_Horizontal_1
  df_filtered_S1["Vertical"]=acelerometer_Vertical_1

  #SEÑAL S2
  acelerometer_Horizontal_2=df_filtered_S2['N-S_filtered'] * 0.7 + df_filtered_S2['E-W_filtered'] * 0.3
  acelerometer_Vertical_2=df_filtered_S2['U-D_filtered']
  #Añadiendo columnas a la señal S2
  df_filtered_S2["Horizontal"]=acelerometer_Horizontal_2
  df_filtered_S2["Vertical"]=acelerometer_Vertical_2

  """
  plt.plot(df_S1['tiempo(s)'], df_filtered_S1['Horizontal'], label='Horizontal: N-S (70%) + E-W (30%)',c='maroon',linewidth=0.5)
  plt.xlabel('Tiempo (s)', fontsize=16)
  plt.ylabel('Aceleración (gal)', fontsize=16)
  plt.grid(True, linestyle='-', linewidth=0.5, color='gray')
  plt.xticks(np.arange(0, 600, 25), fontsize=12)
  plt.yticks(np.arange(-0.001, 0.0013, 0.0005), fontsize=12)
  # Get current axis limits
  x_min, x_max = plt.xlim()
  y_min, y_max = plt.ylim()
  # Expand the limits
  plt.xlim(0 , 200)  # Expand x-axis by 10 units on both sides
  plt.ylim(y_min, y_max)  # Expand y-axis by 50 units on both sides
  plt.legend()
  plt.show()
  """
  #EXTRACCIÓN DE FEATURE POR CADA SEÑAL

  #FEATURES S1
  acel_horiz_S1=df_filtered_S1["Horizontal"]
  acel_vert_S1=df_filtered_S1["Vertical"]
  cfg_file = tsfel.get_features_by_domain()

  #Feature Extraction de señal Horizontal S1
  X_data_S1 = tsfel.time_series_features_extractor(cfg_file, acel_horiz_S1, fs=100, window_size=1000,overlap=0.1)
  #X_data_S1.to_excel(excel_writer=(r'/content/drive/My Drive/TESIS/ARR3_MED_FILT/'+aislador+'/DATA_S1_H.xlsx'))
  #X_data_S1.to_csv('/content/drive/My Drive/TESIS/ARR3_MED_FILT/'+aislador+'/DATA_S1_H.txt', sep=' ', index=False)

  #FEATURES S2
  acel_horiz_S2=df_filtered_S2["Horizontal"]
  acel_vert_S2=df_filtered_S2["Vertical"]

  #Feature Extraction de señal Horizontal S2
  X_data_S2 = tsfel.time_series_features_extractor(cfg_file, acel_horiz_S2, fs=100, window_size=1000,overlap=0.1)
  X_data_S2.columns = ['1' + col[1:] for col in X_data_S2]
  #X_data_S2.to_excel(excel_writer=(r'/content/drive/My Drive/TESIS/ARR3_MED_FILT/'+aislador+'/DATA_S2_H.xlsx'))
  #X_data_S2.to_csv('/content/drive/My Drive/TESIS/ARR3_MED_FILT/'+aislador+'/DATA_S2_H.txt', sep=' ', index=False)

  #APLICACIÓN DE FUNCIÓN DE TRANSFERENCIA (S1/S2)

  df_combined = pd.concat([X_data_S1, X_data_S2], axis=1)

  #print(df_transferencia1)

  return df_combined

st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="expanded", menu_items=None)
st.title('Non-invasive semi-automatic inspection system for lead rubber bearings (LRB)')

st.info('This is app predict the level of damage of lead rubber bearings')

with st.expander('Geometric characteristics of LRB',expanded=True):
  st.write('**Raw data**')
  #df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  #df
  
  st.write('**X**')

  st.write('**y**')

with st.expander('Mechanical Propierties of LRB Materials',expanded=True):
  #st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
  st.write('**X**')

# Input features
with st.sidebar:
  st.header('Input LRB propierties (Geometrical and mechanical)')
  #Input propierties
  Di_mm = st.slider('LRB Diameter (mm)', 749.30, 952.50, 850.90,0.01) #1
  Ht_mm = st.slider('LRB High (mm)', 407.64, 420.34, 410.00,0.01) #2
  Dl_mm = st.slider('Lead Core Diameter (mm)', 114.30, 203.20, 165.10,0.01) #3
  W_kg = st.slider('LRB Weight (kg)', 808, 1456, 1068, 1) #4
  e_pc_mm = st.slider('Thickness of Exterior Rubber Layer (mm)', 18, 20, 19, 1) #5
  cc_und = st.slider('Number Rubber Layers', 10, 30, 28, 1) #6
  e_cc_mm = st.slider('Thickness of Interior Rubber Layers (mm)', 5, 8, 10, 1) #7
  cs_und = st.slider('Number Steel Layers', 10, 30, 27, 1) #8
  e_cs_mm= st.slider('Thickness of Interior Steel Layers (mm)', 2.00, 4.00, 3.04, 0.01) #9
  Fy_ac_mpa=st.slider('Yield Strees Steel (Mpa)', 165, 345, 250, 1) #10
  E_cau_mpa= st.slider('Vertical Elastic Modulus (Mpa)', 1.00, 2.00, 1.44, 0.01) #11
  G_cau_mpa = st.slider('Shear Modulus of Rubber (Mpa)', 0.300, 0.500, 0.392, 0.001) #12
  Fycort_pb_mpa = st.slider('Yield Shear Stress of Lead (Mpa)', 5.00, 10.00, 8.33, 0.01) #13
  st.write('**Input Signal 1**')
  S1=st.file_uploader("Choose file in .txt format of Signal 1", key="file_uploader_1")

  st.write('**Input Signal 2**')
  S2=st.file_uploader("Choose a file in .txt format of Signal 2",key="file_uploader_2")


  #'Di (mm)','Ht (mm)','Dl (mm)','W (kg)','e_pc (mm)','#cc','e_cc (mm)'
  #'#cs','e_cs (mm)','Fy_ac (MPa)','E_cau (MPa)','G_cau (MPa)','Fycort_pb (MPa)'
  # Create a DataFrame for the input features

#model_FDCNN=tf.keras.models.load_model("model_FCDNN/mi_modelo.h5")

with st.expander('Input features',expanded=True):
  st.write('**Input Signal 1**')
  S1=st.file_uploader("Choose file in .txt format of Signal 1", key="file_uploader_3")

  st.write('**Input Signal 2**')
  S2=st.file_uploader("Choose a file in .txt format of Signal 2",key="file_uploader_4")

DF_evaluar = load_dataset(S1,S2)
#DF_evaluar.to_excel(excel_writer=r'/content/drive/My Drive/TESIS/ARR3_MED_FILT/ARR3_DF_FINAL.xlsx')
#DF_evaluar.to_csv('/content/drive/My Drive/TESIS/ARR3_MED_FILT/ARR3_DF_FINAL.txt', sep=' ', index=False)

escalador=StandardScaler()

datos_x = escalador.fit_transform(DF_evaluar)

predictions=model_FDCNN.predict(datos_x)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

total=len(predicted_labels)
nivel_1=0
nivel_2=0
nivel_3=0

for i in range(total):
  if predicted_labels[i] == 0:
      nivel_1+=1
  elif predicted_labels[i]==1:
      nivel_2+=1
  elif predicted_labels[i]==2:
      nivel_3+=1

resultados=[nivel_1,nivel_2,nivel_3]
nivel_final=max(resultados)/total*100
segundo_nivel=sorted(resultados)[-2]/total*100

nivel_mayor=resultados.index(max(resultados))+1
nivel_segundo_mayor=resultados.index(segundo_nivel)+1


with st.expander('Result of Inspection',expanded=True):
  #st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
  st.write('Level of Deterioration '+str(nivel_mayor))

# Data preparation
# Encode X
#encode = ['island', 'sex']
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