import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tsfel
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

#https://drive.google.com/uc?export=download&id=1RKYmoTDteQ9IiScgZOHfIUQ16gLizxol

import requests
import tempfile
import os
import joblib

# Reemplaza con tu ID de archivo de Google Drive
file_id = '1RKYmoTDteQ9IiScgZOHfIUQ16gLizxol'  # El ID del archivo en Google Drive
url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Descargar el archivo del modelo
response = requests.get(url)

# Verifica si la descarga fue exitosa (código 200)
if response.status_code == 200:
    # Crear un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name  # Guardar la ruta del archivo temporal

    # Cargar el modelo desde el archivo temporal
    model_FCDNN = load_model(temp_file_path)
    print("Modelo cargado exitosamente")

    # Elimina el archivo temporal después de cargar el modelo si es necesario
    os.remove(temp_file_path)
else:
    print(f"Error al descargar el modelo: {response.status_code}")

# Reemplaza con tu ID de archivo de Google Drive
file_id2 = '1NKATMIPUo3ohX4VIW9WWF27a1GIXcjpI'  # El ID del archivo en Google Drive
url2 = f"https://drive.google.com/uc?export=download&id={file_id}"

# Descargar el archivo del modelo
response = requests.get(url2)

# Verifica si la descarga fue exitosa (código 200)
if response.status_code == 200:
    # Crear un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file2:
        temp_file2.write(response.content)
        temp_file_path2 = temp_file2.name  # Guardar la ruta del archivo temporal

    # Cargar el modelo desde el archivo temporal
    escalador= load_model(temp_file_path2)
    print("Escalador cargado exitosamente")

    # Elimina el archivo temporal después de cargar el modelo si es necesario
    os.remove(temp_file_path2)
else:
    print(f"Error al descargar el modelo: {response.status_code}")


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
def load_dataset(S1,S2,prop_GM):   #Columnas S1:   'tiempo(s)' | 'N-S' | 'E-W' | 'U-D'
  if S1 is not None and S2 is not None:
        # Leer los archivos directamente desde el objeto UploadedFile
        df_S1 = pd.read_csv(S1, sep=' ')  # Sótano 1
        df_S2 = pd.read_csv(S2, sep=' ')  # Sótano 2
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


       
        #plt.rcParams['axes.facecolor'] = 'white'
        #plt.plot(df_S1['tiempo(s)'], df_S1['N-S_filtered'], label='Canal N-S',c='darkorange',linewidth=0.5)
        #plt.ylabel('Aceleración (gal)', fontsize=16)
        #plt.grid(True, linestyle='-', linewidth=0.5, color='gray')
        #plt.yticks(np.arange(-0.001, 0.0013, 0.0005), fontsize=12)
        # Get current axis limits
        #x_min, x_max = plt.xlim()
        #y_min, y_max = plt.ylim()
        # Expand the limits
        #plt.xlim(0 , 200)  # Expand x-axis by 10 units on both sides
        #plt.ylim(y_min, y_max)  # Expand y-axis by 50 units on both sides
        #plt.legend()
        #plt.show()


        #plt.plot(df_S1['tiempo(s)'], df_S1['E-W_filtered'], label='Canal E-W',c='steelblue',linewidth=0.5)
        #plt.ylabel('Aceleración (gal)', fontsize=16)
        #plt.grid(True, linestyle='-', linewidth=0.5, color='gray')
        #plt.yticks(np.arange(-0.001, 0.0013, 0.0005), fontsize=12)
        # Get current axis limits
        #x_min, x_max = plt.xlim()
        #y_min, y_max = plt.ylim()
        # Expand the limits
        #plt.xlim(0 , 200)  # Expand x-axis by 10 units on both sides
        #plt.ylim(y_min, y_max)  # Expand y-axis by 50 units on both sides
        #plt.legend()
        #plt.show()
      

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

        
        #plt.plot(df_S1['tiempo(s)'], df_filtered_S1['Horizontal'], label='Horizontal: N-S (70%) + E-W (30%)',c='maroon',linewidth=0.5)
        #plt.xlabel('Tiempo (s)', fontsize=16)
        #plt.ylabel('Aceleración (gal)', fontsize=16)
        #plt.grid(True, linestyle='-', linewidth=0.5, color='gray')
        #plt.xticks(np.arange(0, 600, 25), fontsize=12)
        #plt.yticks(np.arange(-0.001, 0.0013, 0.0005), fontsize=12)
        # Get current axis limits
        #x_min, x_max = plt.xlim()
        #y_min, y_max = plt.ylim()
        # Expand the limits
        #plt.xlim(0 , 200)  # Expand x-axis by 10 units on both sides
        #plt.ylim(y_min, y_max)  # Expand y-axis by 50 units on both sides
        #plt.legend()
        #plt.show()
        
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

        prop_GM= pd.concat([prop_GM] * (len(X_data_S1)), ignore_index=True)

        df_combined = pd.concat([prop_GM,X_data_S1, X_data_S2], axis=1)

        def rename_columns(column_name):
            if column_name.startswith("0_"):
                return column_name.replace("0_", "S1_")
            elif column_name.startswith("1_"):
                return column_name.replace("1_", "S2_")
            else:
                return column_name

        df_combined.columns = [rename_columns(col) for col in df_combined.columns]

        df_new=df_combined[['Di (mm)','Ht (mm)','Dl (mm)','W (kg)','e_pc (mm)','#cc','e_cc (mm)','#cs','e_cs (mm)','Fy_ac (MPa)','E_cau (MPa)','G_cau (MPa)','Fycort_pb (MPa)','S1_Median frequency', 'S1_Positive turning points', 'S1_Zero crossing rate','S1_Fundamental frequency',
                  'S1_Spectral roll-on','S1_Neighbourhood peaks','S1_Spectral positive turning points','S1_Power bandwidth','S1_Maximum frequency','S1_Max power spectrum','S2_Median frequency', 'S2_Positive turning points', 'S2_Zero crossing rate','S2_Fundamental frequency',
                  'S2_Spectral roll-on','S2_Neighbourhood peaks','S2_Spectral positive turning points','S2_Power bandwidth','S2_Maximum frequency','S2_Max power spectrum']]
        df_new

        #print(df_transferencia1)
        
        escalador=StandardScaler()

        datos_x = escalador.fit_transform(df_new)

        predictions=model_FCDNN.predict(datos_x)

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
        nivel_segundo_mayor=resultados.index(sorted(resultados)[-2])+1
        with st.expander('Result of Inspection',expanded=True):
        #st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
          st.write('Level of Deterioration '+str(nivel_mayor))

        return df_combined, resultados  # Asegúrate de que esta variable esté definida en tu lógica
  else:
        st.error("Por favor, sube ambos archivos de señal.")
        return None

  

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

#with st.expander('Input features',expanded=True):
  #st.write('**Input Signal 1**')
  #S1=st.file_uploader("Choose file in .txt format of Signal 1", key="file_uploader_3")

  #st.write('**Input Signal 2**')
  #S2=st.file_uploader("Choose a file in .txt format of Signal 2",key="file_uploader_4")


# Crear un DataFrame a partir de las variables de entrada
input_data = {
    'Di (mm)': [Di_mm],
    'Ht (mm)': [Ht_mm],
    'Dl (mm)': [Dl_mm],
    'W (kg)': [W_kg],
    'e_pc (mm)': [e_pc_mm],
    '#cc': [cc_und],
    'e_cc (mm)': [e_cc_mm],
    '#cs': [cs_und],
    'e_cs (mm)': [e_cs_mm],
    'Fy_ac (MPa)': [Fy_ac_mpa],
    'E_cau (MPa)':[E_cau_mpa],
    'G_cau (MPa)':[G_cau_mpa],
    'Fycort_pb (MPa)':[Fycort_pb_mpa]
}

# Convertir el diccionario a un DataFrame
df_input = pd.DataFrame(input_data)


if S1!=None or S2!=None:
  DF_evaluar,resultados = load_dataset(S1,S2,df_input)
  DF_evaluar.to_csv('dataframe.csv', index=False)
  

  