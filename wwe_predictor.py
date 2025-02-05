import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
st.set_page_config(
    page_title="Predicción WWE: WrestleMania 2025",
    layout="wide",
    initial_sidebar_state="expanded"
)
data = {
    "Año": list(range(1993, 2026)),
    "Ganador": [
        "Yokozuna", "Bret Hart & Lex Luger", "Shawn Michaels", "Shawn Michaels",
        "Stone Cold Steve Austin", "Stone Cold Steve Austin", "Vince McMahon",
        "The Rock", "Stone Cold Steve Austin", "Triple H", "Brock Lesnar", "Chris Benoit",
        "Batista", "Rey Mysterio", "The Undertaker", "John Cena", "Randy Orton", "Edge",
        "Alberto Del Rio", "Sheamus", "John Cena", "Batista", "Roman Reigns", "Triple H",
        "Randy Orton", "Shinsuke Nakamura", "Seth Rollins", "Drew McIntyre", "Edge",
        "Brock Lesnar", "Cody Rhodes", "Cody Rhodes", "Jey Uso"
    ],
    "Campeón WM": [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, None],
    "Número Entrada": [27, 23, 1, 18, 5, 24, 2, 24, 27, 22, 29, 1, 28, 2, 30, 30, 8, 29, 28, 22, 19, 28, 19, 30, 23, 14, 10, 16, 1, 30, 30, 1, 20],
    "Veces Campeón Antes": [0, 1, 0, 1, 1, 2, 0, 1, 3, 5, 3, 0, 0, 1, 4, 8, 6, 9, 0, 0, 11, 3, 0, 14, 13, 0, 2, 0, 11, 9, 10, 2, 0],
    "Popularidad Fanáticos": [8, 9, 10, 10, 9, 10, 7, 10, 10, 10, 10, 9, 10, 9, 10, 10, 8, 9, 7, 8, 10, 7, 8, 10, 9, 7, 9, 9, 10, 9, 10, 8, 8]
}

df = pd.DataFrame(data)
train_df = df.dropna()
X = train_df[["Número Entrada", "Veces Campeón Antes", "Popularidad Fanáticos"]]
y = train_df["Campeón WM"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
selected_wrestler = "Jey Uso"
selected_data = df[df["Ganador"] == selected_wrestler].iloc[0]
wrestler_data = pd.DataFrame([[selected_data["Número Entrada"], selected_data["Veces Campeón Antes"], selected_data["Popularidad Fanáticos"]]], 
                             columns=["Número Entrada", "Veces Campeón Antes", "Popularidad Fanáticos"])
probabilidad = model.predict_proba(wrestler_data)[0][1] * 100
popularidad_promedio_ganadores = df[df["Campeón WM"] == 1]["Popularidad Fanáticos"].mean()
numero_entrada_promedio_ganadores = df[df["Campeón WM"] == 1]["Número Entrada"].mean()
def calcular_tasa_exito(numero_entrada, popularidad, veces_campeon_antes):
    factor_numero_entrada = 0.4
    factor_popularidad = 0.4
    factor_veces_campeon = 0.2
    cercania_numero_entrada = 1 - abs(numero_entrada - numero_entrada_promedio_ganadores) / 30
    cercania_popularidad = 1 - abs(popularidad - popularidad_promedio_ganadores) / 10
    ajuste_veces_campeon = min(veces_campeon_antes / 10, 1) 
    tasa_exito = (cercania_numero_entrada * factor_numero_entrada +
                  cercania_popularidad * factor_popularidad +
                  ajuste_veces_campeon * factor_veces_campeon) * 100

    return tasa_exito
tasa_exito_historica = df[df["Ganador"] == selected_wrestler]["Campeón WM"].mean() * 100
if pd.isna(tasa_exito_historica):
    tasa_exito_historica = 0
tasa_exito_final = (tasa_exito_historica + calcular_tasa_exito(selected_data["Número Entrada"], selected_data["Popularidad Fanáticos"], selected_data["Veces Campeón Antes"])) / 2

correlacion_popularidad_victoria = df[["Popularidad Fanáticos", "Campeón WM"]].corr().iloc[0, 1]
st.markdown("""
    <h1 style='text-align: center; color: #FFD700;'>Predicción WWE: WrestleMania 2025</h1>
    <h3 style='text-align: center;'>¿Jey Uso ganará el Campeonato Mundial en Wrestlemania?</h3>
""", unsafe_allow_html=True)
st.markdown("<div style='border-radius: 15px; padding: 15px; background-color: #2D2D2D;'>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label=f"Probabilidad de {selected_wrestler}", value=f"{probabilidad:.2f}%", delta=f"{probabilidad - 50:.2f}%")

with col2:
    st.metric(label="Popularidad del Luchador", value=f"{selected_data['Popularidad Fanáticos']}/10", 
              delta=f"{selected_data['Popularidad Fanáticos'] - popularidad_promedio_ganadores:.2f} vs promedio")

with col3:
    st.metric(label="Número de Entrada Promedio (Ganadores)", value=f"{numero_entrada_promedio_ganadores:.1f}",
              delta=f"{selected_data['Número Entrada'] - numero_entrada_promedio_ganadores:.2f} vs seleccionado")

with col4:
    st.metric(label="Tasa de Éxito Mejorada", value=f"{tasa_exito_final:.2f}%")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='border-radius: 15px; padding: 15px; background-color: #2D2D2D;'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Análisis de Correlación entre Popularidad y Victoria")
    st.write(f"La correlación entre la popularidad y las victorias en WrestleMania es: **{correlacion_popularidad_victoria:.2f}**")
    fig_scatter = px.scatter(df, x="Popularidad Fanáticos", y="Campeón WM", trendline="lowess",
                             labels={"Popularidad Fanáticos": "Popularidad", "Campeón WM": "Victoria (1 = Sí, 0 = No)"},
                             template="plotly_dark", color="Ganador", hover_name="Ganador")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("Número de Entrada de Cada Luchador")
    fig_bar = px.bar(df, x="Ganador", y="Número Entrada", title="Número de Entrada de Cada Luchador", 
                     labels={"Ganador": "Luchador", "Número Entrada": "Número de Entrada"}, 
                     template="plotly_dark", color="Campeón WM", color_continuous_scale='blues')
    st.plotly_chart(fig_bar, use_container_width=True)
st.subheader(f"Comparación de Características de {selected_wrestler}")
fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=[selected_data["Número Entrada"], selected_data["Veces Campeón Antes"], selected_data["Popularidad Fanáticos"]],
    theta=['Número Entrada', 'Veces Campeón Antes', 'Popularidad Fanáticos'],
    fill='toself',
    name=selected_wrestler
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 30]
        )),
    showlegend=True,
    template="plotly_dark"
)

st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
    <style>
        .stApp { background-color: #1E1E1E; color: #f1f1f1; }
        h1, h3 { text-align: center; }
        .stMetric { font-size: 24px; color: #FFA500; }
        .stSidebar { background-color: #2D2D2D; }
    </style>
""", unsafe_allow_html=True)