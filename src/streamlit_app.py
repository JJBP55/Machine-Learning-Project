# src/streamlit_app.py
# App de predicción de dirección BTC/USDT - Proyecto final 4Geeks
# Fase 1: setup + carga de artefactos del modelo

# src/streamlit_app.py
# App de predicción de dirección BTC/USDT - Proyecto final 4Geeks
# Fase 2: predicción en vivo con datos de Binance

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

# ---------- Configuración de la página ----------
st.set_page_config(
    page_title="Predictor BTC/USDT",
    page_icon="₿",
    layout="wide"
)

# ---------- Rutas a los artefactos ----------
RUTA_MODELO = "modelos_ml/modelo_final.pkl"
RUTA_SCALER = "modelos_ml/scaler.pkl"
RUTA_FEATURES = "modelos_ml/features_finales.pkl"


# ---------- Carga de artefactos (cacheado) ----------
@st.cache_data
def cargar_modelo():
    modelo = joblib.load(RUTA_MODELO)
    return modelo

@st.cache_data
def cargar_scaler():
    scaler = joblib.load(RUTA_SCALER)
    return scaler

@st.cache_data
def cargar_features():
    features = joblib.load(RUTA_FEATURES)
    return features


# ---------- Función para traer velas de Binance ----------
@st.cache_data(ttl=60)  # cachea el resultado 1 minuto
def obtener_velas_binance(cantidad):
    # Endpoint público de Binance para velas de BTC/USDT en intervalo 1h
    url = "https://api.binance.us/api/v3/klines"
    parametros = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "limit": cantidad
    }
    respuesta = requests.get(url, params=parametros)
    datos_crudos = respuesta.json()

    # Nombres de columnas según la documentación de Binance
    columnas = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(datos_crudos, columns=columnas)

    # Convertimos los tipos que nos interesan
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    # Nos quedamos solo con las columnas necesarias
    df = df[["open_time", "open", "high", "low", "close", "volume"]]

    return df


# ---------- Función para calcular RSI ----------
def calcular_rsi(precios, periodo):
    delta = precios.diff()
    ganancia = delta.copy()
    perdida = delta.copy()

    # Ganancias: solo los positivos. Pérdidas: solo los negativos (en valor absoluto)
    ganancia[ganancia < 0] = 0
    perdida[perdida > 0] = 0
    perdida = perdida.abs()

    media_ganancia = ganancia.rolling(window=periodo).mean()
    media_perdida = perdida.rolling(window=periodo).mean()

    rs = media_ganancia / media_perdida
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------- Función para calcular todas las features ----------
def calcular_features(df):
    # Retorno de 1 hora
    df["return_1h"] = df["close"].pct_change()

    # Medias móviles simples y distancias porcentuales al cierre
    df["sma_6"] = df["close"].rolling(window=6).mean()
    df["sma_12"] = df["close"].rolling(window=12).mean()
    df["sma_24"] = df["close"].rolling(window=24).mean()
    df["sma_72"] = df["close"].rolling(window=72).mean()

    df["dist_sma_6"] = (df["close"] - df["sma_6"]) / df["sma_6"]
    df["dist_sma_12"] = (df["close"] - df["sma_12"]) / df["sma_12"]
    df["dist_sma_24"] = (df["close"] - df["sma_24"]) / df["sma_24"]
    df["dist_sma_72"] = (df["close"] - df["sma_72"]) / df["sma_72"]

    # Volatilidad (desvío estándar de los retornos)
    df["volatilidad_6h"] = df["return_1h"].rolling(window=6).std()
    df["volatilidad_24h"] = df["return_1h"].rolling(window=24).std()

    # RSI 14
    df["rsi_14"] = calcular_rsi(df["close"], 14)

    # Bollinger %B: posición del precio dentro de las bandas
    sma_20 = df["close"].rolling(window=20).mean()
    std_20 = df["close"].rolling(window=20).std()
    banda_superior = sma_20 + 2 * std_20
    banda_inferior = sma_20 - 2 * std_20
    df["bollinger_pct"] = (df["close"] - banda_inferior) / (banda_superior - banda_inferior)

    # MACD (diferencia entre EMA 12 y EMA 26)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26

    # Rango de la vela (high - low en términos porcentuales)
    df["rango_vela"] = (df["high"] - df["low"]) / df["open"]

    # Mecha inferior: distancia entre el mínimo y el cuerpo inferior
    cuerpo_inferior = df[["open", "close"]].min(axis=1)
    df["mecha_inferior"] = (cuerpo_inferior - df["low"]) / df["open"]

    # es_outlier: retorno mayor a 3 desvíos estándar (aproximación con la muestra)
    media_retorno = df["return_1h"].mean()
    std_retorno = df["return_1h"].std()
    df["es_outlier"] = 0
    for i in range(len(df)):
        retorno_actual = df["return_1h"].iloc[i]
        if pd.notna(retorno_actual):
            if abs(retorno_actual - media_retorno) > 3 * std_retorno:
                df.loc[df.index[i], "es_outlier"] = 1

    # Hora del día y día de la semana
    df["hora"] = df["open_time"].dt.hour
    df["dia_semana"] = df["open_time"].dt.dayofweek

    return df


# ---------- Header ----------
st.title("₿ Predictor de dirección BTC/USDT")
st.markdown("### Proyecto final - Data Science Bootcamp 4Geeks Academy")
st.markdown("Modelo de clasificación binaria que predice la probabilidad de que la próxima vela de 1 hora sea alcista o bajista.")
st.markdown("---")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Configuración")
st.sidebar.markdown("Usa los datos más recientes de Binance para predecir la próxima vela de 1h.")
boton_predecir = st.sidebar.button("🔮 Predecir próxima vela", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("**Info del modelo:**")
st.sidebar.markdown("- Tipo: Regresión Logística")
st.sidebar.markdown("- Accuracy en test: 54.17%")
st.sidebar.markdown("- AUC-ROC: 55.78%")
st.sidebar.markdown("- Horizonte: próxima vela 1h")

# ---------- Carga de artefactos ----------
modelo = cargar_modelo()
scaler = cargar_scaler()
features = cargar_features()

# ---------- Lógica de predicción ----------
if boton_predecir:
    with st.spinner("Obteniendo datos de Binance..."):
        # Pedimos 200 velas: necesitamos como mínimo 72 para SMA_72, pedimos de más por seguridad
        df_velas = obtener_velas_binance(200)

    # Descartamos la última fila porque puede ser una vela en formación (no cerrada aún)
    df_velas_cerradas = df_velas.iloc[:-1].copy()

    with st.spinner("Calculando features..."):
        df_con_features = calcular_features(df_velas_cerradas)

    # Nos quedamos con la última vela cerrada
    ultima_fila = df_con_features.iloc[-1]

    # Verificamos que no falten valores en las features
    features_ultima_vela = ultima_fila[features]

    if features_ultima_vela.isna().any():
        st.error("❌ Hay valores faltantes en las features. Probablemente no hay suficientes velas históricas para calcular todos los indicadores.")
    else:
        # Preparamos el array de entrada para el modelo
        X = features_ultima_vela.values.reshape(1, -1)
        X_escalado = scaler.transform(X)

        # Predicción y probabilidades
        prediccion = modelo.predict(X_escalado)[0]
        probabilidades = modelo.predict_proba(X_escalado)[0]
        prob_baja = probabilidades[0]
        prob_sube = probabilidades[1]

        # ---------- Mostrar resultado ----------
        st.subheader("🎯 Resultado de la predicción")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Precio de cierre (última vela)", value=f"${ultima_fila['close']:,.2f}")

        with col2:
            hora_str = ultima_fila["open_time"].strftime("%Y-%m-%d %H:%M UTC")
            st.metric(label="Vela analizada", value=hora_str)

        with col3:
            if prediccion == 1:
                st.metric(label="Predicción", value="📈 SUBE")
            else:
                st.metric(label="Predicción", value="📉 BAJA")

        st.markdown("---")
        st.subheader("📊 Probabilidades")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Probabilidad de que suba:** {prob_sube*100:.2f}%")
            st.progress(float(prob_sube))
        with col_b:
            st.markdown(f"**Probabilidad de que baje:** {prob_baja*100:.2f}%")
            st.progress(float(prob_baja))

        # Interpretación
        st.markdown("---")
        st.subheader("💡 Interpretación")

        if prob_sube > 0.55:
            st.success(f"Señal alcista con confianza moderada ({prob_sube*100:.2f}%). El modelo sugiere que la próxima vela cerrará por encima del precio actual.")
        elif prob_baja > 0.55:
            st.error(f"Señal bajista con confianza moderada ({prob_baja*100:.2f}%). El modelo sugiere que la próxima vela cerrará por debajo del precio actual.")
        else:
            st.warning(f"Señal débil. El modelo no tiene confianza suficiente para tomar una decisión clara (probabilidades muy cercanas al 50%).")

        # Detalles de las features usadas
        with st.expander("🔍 Ver features usadas en la predicción"):
            st.dataframe(features_ultima_vela.to_frame(name="valor"))

else:
    st.info("👈 Hacé clic en **Predecir próxima vela** en el sidebar para obtener una predicción con los datos más recientes de Binance.")