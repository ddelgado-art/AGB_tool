import streamlit as st
import pandas as pd
import sqlite3
import tempfile
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import io

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Confort Térmico & AFN",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Dashboard de Análisis: Temperatura y Caudales (ASHRAE 55)")
st.markdown("""
Esta herramienta permite analizar simulaciones energéticas cargando el archivo SQL de resultados.
Visualiza temperatura operativa, renovaciones de aire y porcentajes de confort adaptativo.
""")

# -----------------------------------------------------------------------------
# FUNCIONES DE CARGA Y PROCESAMIENTO (CACHED)
# Usamos @st.cache_data para que no recalcule todo si solo cambiamos un filtro visual
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_data_from_sql(uploaded_file):
    """
    Recibe un archivo subido, lo guarda temporalmente, conecta SQLite 
    y extrae los DataFrames base.
    """
    # Streamlit sube archivos como bytes, SQLite necesita una ruta en disco.
    # Creamos un archivo temporal.
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".sql")
    tfile.write(uploaded_file.getvalue())
    temp_path = tfile.name
    tfile.close()

    try:
        conn = sqlite3.connect(temp_path)
        
        # 1. Leer índices de tiempo
        df_type_1 = pd.read_sql_query('select Time.TimeIndex from Time WHERE Time.IntervalType="1"', conn)
        
        # 2. Leer datos crudos
        # Optimizamos leyendo solo lo necesario en vez de "select *" si la tabla es gigante, 
        # pero mantenemos la lógica original por seguridad.
        Rdata = pd.read_sql_query('select ReportDataDictionaryIndex, TimeIndex, Value from ReportData', conn)
        
        # 3. Diccionario de variables
        Rdatadict = pd.read_sql_query('select ReportDataDictionaryIndex, KeyValue, Name from ReportDataDictionary', conn)
        
        # 4. Construcción de DateTime (Optimizado con pandas to_datetime directo)
        df_time = pd.read_sql_query("SELECT TimeIndex, '2002' || '/' || Month ||'/'|| Day || '-' || Hour || ':00:00' as DateTime FROM Time", conn)
        df_time['DateTime'] = df_time['DateTime'].str.replace('24:00:00', '0:00:00')
        # format explícito es más rápido que infer_datetime_format
        df_time['DateTime'] = pd.to_datetime(df_time['DateTime'], format='%Y/%m/%d-%H:%M:%S') 

        conn.close()
        
    finally:
        # Limpieza del archivo temporal
        os.remove(temp_path)

    return df_type_1, Rdata, Rdatadict, df_time

@st.cache_data(show_spinner=True)
def process_indicators(df_type_1, Rdata, Rdatadict, df_time):
    """
    Procesa los dataframes crudos para obtener una tabla unificada con ZOT y AFN.
    """
    # Filtrar diccionarios
    Rdatadict_ZOT = Rdatadict[Rdatadict.Name == "Zone Operative Temperature"]
    Rdatadict_AFN = Rdatadict[Rdatadict.Name == "AFN Zone Infiltration Air Change Rate"]

    # --- Procesar ZOT (Temperatura) ---
    value_ZOT = pd.merge(left=Rdata, right=Rdatadict_ZOT, on='ReportDataDictionaryIndex', how='inner')
    value_ZOT = pd.merge(left=df_type_1, right=value_ZOT, on='TimeIndex', how='inner')
    value_ZOT = value_ZOT[["TimeIndex", "KeyValue", "Value"]].rename(columns={'Value': 'ZOT'})

    # --- Procesar AFN (Caudales) ---
    value_AFN = pd.merge(left=Rdata, right=Rdatadict_AFN, on='ReportDataDictionaryIndex', how='inner')
    value_AFN = pd.merge(left=df_type_1, right=value_AFN, on='TimeIndex', how='inner')
    value_AFN = value_AFN[["TimeIndex", "KeyValue", "Value"]].rename(columns={'Value': 'AFN'})

    # --- Unir Resultados ---
    # Usamos merge outer por si en algún momento falta un dato de alguna variable
    df_res = pd.merge(value_ZOT, value_AFN, on=["TimeIndex", "KeyValue"], how='outer')
    df_res.rename(columns={'KeyValue': 'Zone'}, inplace=True)

    # --- Unir con Tiempo ---
    df_final = pd.merge(df_res, df_time, on="TimeIndex", how='inner')
    df_final = df_final[["TimeIndex", "DateTime", "Zone", "ZOT", "AFN"]]
    df_final = df_final.sort_values(by=["Zone", "TimeIndex"])
    
    return df_final

def calculate_stats_and_comfort(df_input, r1, r2, s1, s2):
    """
    Calcula estadísticas y rangos de confort basados en los inputs del usuario.
    Esta función NO se cachea porque depende de los inputs variables (R1, R2...).
    """
    # 1. Estadísticas básicas por zona
    stats = df_input.groupby("Zone").agg(
        ZOT_min=("ZOT", "min"),
        ZOT_mean=("ZOT", "mean"),
        ZOT_max=("ZOT", "max"),
        AFN_min=("AFN", "min"),
        AFN_mean=("AFN", "mean"),
        AFN_max=("AFN", "max")
    ).reset_index()

    # 2. Cálculo de Confort (Vectorizado con pd.cut para eficiencia)
    # Copia para no alterar el original
    df_calc = df_input.copy()

    # Definir etiquetas y bins para Rango 1 (80%)
    # Bins: [-Inf, R1, R2, Inf] -> Labels: [cold1, confort1, hot1]
    df_calc['Rango1'] = pd.cut(
        df_calc['ZOT'], 
        bins=[-float('inf'), r1, r2, float('inf')], 
        labels=['cold1', 'confort1', 'hot1']
    )

    # Definir etiquetas y bins para Rango 2 (90%)
    df_calc['Rango2'] = pd.cut(
        df_calc['ZOT'], 
        bins=[-float('inf'), s1, s2, float('inf')], 
        labels=['cold2', 'confort2', 'hot2']
    )

    # 3. Calcular Porcentajes
    comfort_list = []
    
    for zone in df_calc['Zone'].unique():
        df_zone = df_calc[df_calc['Zone'] == zone]
        
        # Conteo normalizado (porcentaje)
        c1 = df_zone['Rango1'].value_counts(normalize=True)
        c2 = df_zone['Rango2'].value_counts(normalize=True)
        
        # Crear diccionario de resultados
        row = {'Zone': zone}
        for label in ['cold1', 'confort1', 'hot1']:
            row[label] = c1.get(label, 0)
        for label in ['cold2', 'confort2', 'hot2']:
            row[label] = c2.get(label, 0)
            
        comfort_list.append(row)
    
    df_comfort = pd.DataFrame(comfort_list)
    
    # Unir estadísticas con confort
    df_res_final = pd.merge(stats, df_comfort, on="Zone")
    
    return df_res_final.round(2)

# -----------------------------------------------------------------------------
# SIDEBAR - CONTROLES
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Carga de Datos")
    uploaded_sql = st.file_uploader("Subir archivo SQL (.sql, .db)", type=["sql", "db"])
    
    st.header("2. Parámetros ASHRAE 55")
    st.markdown("Definir límites de temperatura (°C)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**80% Aceptabilidad**")
        R1 = st.number_input("Mínimo (R1)", value=18.0, step=0.5)
        R2 = st.number_input("Máximo (R2)", value=24.0, step=0.5)
    with col2:
        st.markdown("**90% Aceptabilidad**")
        S1 = st.number_input("Mínimo (S1)", value=19.0, step=0.5)
        S2 = st.number_input("Máximo (S2)", value=23.0, step=0.5)

# -----------------------------------------------------------------------------
# LÓGICA PRINCIPAL
# -----------------------------------------------------------------------------

if uploaded_sql is not None:
    # 1. Cargar y procesar datos base
    type1, r_data, r_dict, d_time = load_data_from_sql(uploaded_sql)
    df_main = process_indicators(type1, r_data, r_dict, d_time)
    
    # 2. Filtro de Zonas (En sidebar o main, mejor aquí para afectar todo)
    all_zones = df_main['Zone'].unique()
    selected_zones = st.multiselect("Seleccionar Zonas para visualizar:", all_zones, default=all_zones[:2])
    
    if not selected_zones:
        st.warning("Por favor selecciona al menos una zona.")
        st.stop()
        
    # Filtrar dataframe principal para gráficos temporales
    df_filtered = df_main[df_main['Zone'].isin(selected_zones)]

    # 3. Calcular Estadísticas Finales (res_final)
    # Se calcula sobre TODAS las zonas cargadas para la tabla resumen, 
    # o solo las filtradas. Generalmente la tabla resumen se prefiere completa.
    # Aquí calcularemos sobre todo y filtraremos visualmente si es necesario.
    df_res_final = calculate_stats_and_comfort(df_main, R1, R2, S1, S2)

    # -------------------------------------------------------------------------
    # VISUALIZACIÓN
    # -------------------------------------------------------------------------
    
    # --- GRÁFICO 1: LÍNEAS TEMPORALES (DOBLE EJE) ---
    st.subheader("📈 Comportamiento Temporal: Temperatura vs. Caudales")
    
    # Usamos Plotly Graph Objects para mayor control sobre el doble eje
    fig_time = make_subplots(specs=[[{"secondary_y": True}]])

    # Colores para diferenciar zonas si hay múltiples
    colors = px.colors.qualitative.Plotly

    for i, zone in enumerate(selected_zones):
        df_z = df_filtered[df_filtered['Zone'] == zone]
        color = colors[i % len(colors)]
        
        # Traza Temperatura (Eje Izquierdo)
        fig_time.add_trace(
            go.Scatter(x=df_z['DateTime'], y=df_z['ZOT'], name=f"{zone} - Temp (°C)",
                       line=dict(color=color, width=2)),
            secondary_y=False
        )
        
        # Traza Caudales (Eje Derecho) - Estilo punteado o más suave
        fig_time.add_trace(
            go.Scatter(x=df_z['DateTime'], y=df_z['AFN'], name=f"{zone} - AFN (ACH)",
                       line=dict(color=color, width=1, dash='dot'), opacity=0.7),
            secondary_y=True
        )

    # Configuración de Layout
    fig_time.update_layout(
        title_text="Evolución Horaria",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Títulos de ejes
    fig_time.update_yaxes(title_text="Temperatura Operativa (°C)", secondary_y=False, showgrid=True)
    fig_time.update_yaxes(title_text="Infiltración / Ventilación (ACH)", secondary_y=True, showgrid=False)
    
    st.plotly_chart(fig_time, use_container_width=True)

    # --- GRÁFICO 2: CONFORT ASHRAE 55 (BARRAS APILADAS) ---
    st.subheader("🌡️ Porcentajes de Confort Adaptativo (ASHRAE 55)")
    
    col_graph1, col_graph2 = st.columns(2)
    
    # Filtramos la tabla de resultados para mostrar solo las zonas seleccionadas en los gráficos
    df_res_viz = df_res_final[df_res_final['Zone'].isin(selected_zones)].copy()

    # Función auxiliar para graficar barras apiladas
    def plot_stacked_comfort(df, suffix, title):
        # Seleccionar columnas relevantes
        cols = ['Zone', f'cold{suffix}', f'confort{suffix}', f'hot{suffix}']
        df_plot = df[cols].copy()
        
        # Convertir a formato largo para plotly express
        df_melt = df_plot.melt(id_vars='Zone', var_name='Estado', value_name='Porcentaje')
        
        # Mapa de colores intuitivo
        color_map = {
            f'cold{suffix}': '#3366CC',    # Azul
            f'confort{suffix}': '#109618', # Verde
            f'hot{suffix}': '#DC3912'      # Rojo
        }
        
        fig = px.bar(
            df_melt, x="Zone", y="Porcentaje", color="Estado",
            title=title,
            color_discrete_map=color_map,
            text_auto='.1%' # Mostrar valor en la barra
        )
        fig.update_layout(barmode='stack', yaxis=dict(tickformat=".0%"))
        return fig

    with col_graph1:
        st.plotly_chart(plot_stacked_comfort(df_res_viz, "1", "Rango 1 (80% Aceptabilidad)"), use_container_width=True)
        
    with col_graph2:
        st.plotly_chart(plot_stacked_comfort(df_res_viz, "2", "Rango 2 (90% Aceptabilidad)"), use_container_width=True)

    # --- TABLA RESUMEN Y DESCARGA ---
    st.divider()
    st.subheader("📋 Tabla de Resultados Consolidados (res_final)")
    
    # Mostrar tabla interactiva completa
    st.dataframe(df_res_final, use_container_width=True)
    
    # Preparar Excel en memoria
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_res_final.to_excel(writer, sheet_name='Resultados', index=False)
        # Se podría agregar otra hoja con los datos crudos si se quisiera
        
    # Botón de descarga
    st.download_button(
        label="📥 Descargar Resultados en Excel",
        data=buffer.getvalue(),
        file_name="res_final_ashrae55.xlsx",
        mime="application/vnd.ms-excel"
    )

else:
    # Mensaje de bienvenida cuando no hay archivo
    st.info("👋 Por favor, sube un archivo SQL usando la barra lateral para comenzar.")