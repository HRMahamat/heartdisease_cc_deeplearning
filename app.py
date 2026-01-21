import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import time
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense

# ==========================================
# 1. CONFIGURATION ET STYLE CSS "ULTRA-GLOW"
# ==========================================
st.set_page_config(page_title="CardioPredict", layout="wide", page_icon="🫀")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Poppins:wght@300;400;600&display=swap');

    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }
    /* Force le fond noir profond sur toute l'application */
    [data-testid="stAppViewContainer"] {
        background-color: #0f0c29 !important;
    }

    /* Force le fond sombre de la sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b091f !important;
    }

    /* Harmonise les onglets (Tabs) pour qu'ils ne soient pas blancs */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: white !important;
    }

    /* Style de l'onglet actif */
    button[aria-selected="true"] {
        border-bottom-color: #00d4ff !important;
        color: #00d4ff !important;
    }
    /* Animation du Titre Cyberpunk */
    .title-text {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00d4ff, #f5576c, #ffffff, #00d4ff);
        background-size: 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        animation: gradientMove 8s linear infinite;
        margin-bottom: 10px;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }

    /* Conteneurs Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        margin-bottom: 20px;
    }

    /* --- INPUTS : SELECTBOX & NUMBER --- */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #00d4ff !important;
        font-family: 'Orbitron', sans-serif;
    }

    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div {
        background-color: rgba(15, 12, 41, 0.6) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    /* --- CORRECTION SLIDERS (LA BARRE REVIENT) --- */
    /* On cible uniquement la barre de progression pour lui donner du néon */
    div[data-baseweb="slider"] > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #f5576c) !important;
    }

    /* Le bouton du slider (le rond) */
    div[role="slider"] {
        background-color: #f5576c !important;
        border: 2px solid white !important;
        box-shadow: 0 0 10px #f5576c !important;
    }

    /* Le rail du slider (la partie vide) */
    div[data-baseweb="slider"] > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* --- BOUTON D'ANALYSE --- */
    div.stButton > button {
        background: linear-gradient(45deg, #00d4ff, #f5576c);
        border: none;
        color: white;
        padding: 20px;
        font-family: 'Orbitron', sans-serif;
        font-size: 24px;
        border-radius: 50px;
        width: 100%;
        letter-spacing: 2px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        transition: 0.5s;
        margin-top: 20px;
    }

    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 40px rgba(245, 87, 108, 0.8);
        letter-spacing: 5px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid #00d4ff;
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 2. MOTEUR D'IA (CLASSES & CHARGEMENT)
# ==========================================

class HeartClassifier(nn.Module):
    def __init__(self, input_size):
        super(HeartClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x): return self.net(x)


@st.cache_resource
def load_models():
    # PyTorch
    with open('Hamad_Rassem_Mahamat_best_model.pkl', 'rb') as f:
        d_pt = pickle.load(f)
    m_pt = HeartClassifier(d_pt['input_size'])
    m_pt.load_state_dict(d_pt['model_state'])
    m_pt.eval()

    # TensorFlow
    with open('Hamad_Rassem_Mahamat_new_best_model.pkl', 'rb') as f:
        d_tf = pickle.load(f)
    m_tf = Sequential.from_config(d_tf['config'])
    m_tf.set_weights(d_tf['weights'])

    return (m_pt, d_pt['preprocessor']), (m_tf, d_tf['prep'])


# ==========================================
# 3. FONCTIONS UTILITAIRES (GUAuge & CONSEILS)
# ==========================================

def display_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Niveau de Risque Cardiaque (%)", 'font': {'size': 24, 'color': "#00d4ff"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#f5576c"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.3)'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 50}}))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Orbitron"})
    st.plotly_chart(fig, use_container_width=True)


def get_recommendations(prob, age, chol, bp, mhr):
    recs = []
    if prob >= 0.5:
        recs.append(
            "🔴 **URGENT :** Prenez rendez-vous pour un bilan cardiologique complet (ECG d'effort, Échographie).")

    if chol > 200:
        recs.append(
            "🧪 **Cholestérol :** Votre taux est élevé. Privilégiez les Oméga-3 et réduisez les graisses saturées.")

    if bp > 140:
        recs.append("🧂 **Tension :** Limitez votre consommation de sel à moins de 5g/jour et surveillez votre repos.")

    if age > 60 and prob > 0.4:
        recs.append("👴 **Âge :** Un suivi semestriel est recommandé pour prévenir les complications silencieuses.")

    if mhr < 120 and age < 50:
        recs.append(
            "🏃 **Activité :** Votre fréquence cardiaque maximale semble basse. Un entraînement cardio régulier est conseillé.")

    if not recs:
        recs.append("🌟 **Excellent :** Continuez votre routine actuelle. Votre profil est très sain !")

    return recs


# ==========================================
# 4. STRUCTURE DE L'APPLICATION
# ==========================================

st.markdown('<h1 class="title-text">CARDIO PREDICT AI</h1>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 ACCUEIL", "🧬 DIAGNOSTIC EXPERT", "📉 ANALYSE TECHNIQUE"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Bienvenue dans le futur du diagnostic")
        st.markdown("""  
        Cette application utilise le Deep Learning pour prédire les risques de maladies cardiaques avec une précision de plus de 88 %.
        """)
        st.markdown("---")
        st.write("""
            **Technologies utilisées :**
            - Frameworks : Pytorch et Tensorflow
            - Prétraitement : Scikit-learn 
            - Interface : Streamlit avec CSS personnalisé
            """)
    with col2:
        st.image(
            "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJtZnd4ZzRnd3B6Znd4Znd4Znd4Znd4Znd4Znd4Znd4Znd4JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCZjdD1n/3o7TKVUn7iM8FMEU24/giphy.gif")

with tab2:
    data_models = load_models()

    if data_models[0] is None:
        st.error("⚠️ Modèles introuvables. Vérifiez les fichiers .pkl.")
    else:
        (pt_m, pt_p), (tf_m, tf_p) = data_models

        # Carte Glassmorphism unique pour tous les inputs
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 PROFIL")
            age = st.slider("Âge du patient", 18, 100, 50)
            cp = st.selectbox("Douleur Thoracique",
                              [("Asymptomatique", 4), ("Non-Angineuse", 3), ("Angine Atypique", 2),
                               ("Angine Typique", 1)], format_func=lambda x: x[0])[1]
            sex = st.selectbox("Genre", [("Homme", 1), ("Femme", 0)], format_func=lambda x: x[0])[1]

        with col2:
            st.markdown("##### ❤️ SIGNES VITAUX")
            chol = st.slider("Cholestérol (mg/dl)", 100, 600, 220)
            fbs = st.selectbox("Glycémie > 120mg/dl", [("Non", 0), ("Oui", 1)], format_func=lambda x: x[0])[1]
            bp = st.slider("Pression systolique (mmHg)", 80, 200, 130)

        with col3:
            st.markdown("##### 🔬 EXAMENS")
            exang = st.selectbox("Angine d'effort", [("Non", 0), ("Oui", 1)], format_func=lambda x: x[0])[1]
            mhr = st.slider("Fréquence cardiaque max", 60, 220, 150)
            ecg = st.selectbox("Résultat ECG", [("Normal", 0), ("ST-T Anormal", 1), ("Hypertrophie", 2)],
                               format_func=lambda x: x[0])[1]

        st.markdown("---")
        c_last1, c_last2 = st.columns(2)
        with c_last1:
            oldpeak = st.slider("Dépression ST (Oldpeak)", 0.0, 6.0, 1.0)
        with c_last2:
            slope = st.selectbox("Pente du segment ST", [("Montante", 1), ("Plate", 2), ("Descendante", 3)],
                                 format_func=lambda x: x[0])[1]

        st.markdown("</div>", unsafe_allow_html=True)

        # Zone d'action (Hors du style des inputs glassmorphism)
        st.write("")
        engine = st.radio("SÉLECTIONNER LE MOTEUR NEURONAL",
                          ["Pytorch (stabilité maximale)", "Tensorflow (vitesse batch)"], horizontal=True)

        if st.button("LANCER L'ANALYSE BIOMÉTRIQUE"):
            input_data = pd.DataFrame([[age, sex, cp, bp, chol, fbs, ecg, mhr, exang, oldpeak, slope]],
                                      columns=['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
                                               'fasting blood sugar', 'resting ecg', 'max heart rate',
                                               'exercise angina', 'oldpeak', 'ST slope'])

            with st.spinner("Synchronisation avec les réseaux de neurones..."):
                time.sleep(1.5)
                if "Pytorch" in engine:
                    proc = pt_p.transform(input_data)
                    with torch.no_grad():
                        prob = pt_m(torch.tensor(proc, dtype=torch.float32)).item()
                else:
                    proc = tf_p.transform(input_data)
                    prob = tf_m.predict(proc, verbose=0)[0][0]

            # RÉSULTATS
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                display_risk_gauge(prob)

            with res_col2:
                st.subheader("💡 RECOMMANDATIONS")
                advices = get_recommendations(prob, age, chol, bp, mhr)
                for a in advices:
                    st.markdown(f"- {a}")

                st.warning(
                    "⚠️ *Note : Cette IA est un outil d'aide à la décision et ne remplace pas l'avis d'un cardiologue!*")
            st.markdown("</div>", unsafe_allow_html=True)
            if prob < 0.3: st.balloons()

with tab3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.header("")
    c1, c2 = st.columns(2)
    with c1:
        st.info("### Modèle Pytorch")
        st.metric("Précision", "89.92%", "+2.1%")
        st.write("- **Architecture :** 3 couches denses ;")
        st.write("- **Régularisation :** Dropout 20% ;")
        st.write("- **Particularité :** Convergence extrêmement stable sans écart d'overfitting.")
    with c2:
        st.success("### Modèle Tensorflow")
        st.metric("Précision", "87.79%", "-0.2%")
        st.write("- **Architecture :** 3 couches denses ;")
        st.write("- **Régularisation :** Dropout + L2 (0.01) ;")
        st.write("- **Particularité :** Très efficace pour les prédictions en batch.")

    st.write("---")
    st.write("### Pourquoi deux modèles ?")
    st.write("""
            Nous avons implémenté les deux frameworks leaders du marché. 
            Bien que les résultats finaux soient proches, **Pytorch** a montré une meilleure gestion de l'écart 
            entraînement/validation sur ce dataset spécifique.
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer Sidebar
st.sidebar.markdown(f"""
    <br><div style='text-align: center; padding-top: 20px;'>
        <img src="https://img.icons8.com/color/512/heart-with-pulse.png" width="80">
        <h3 style='color:#00d4ff; font-family:Orbitron;'>Cardio Predict v1.0</h3>
        <p style='font-size: 0.8em;'>Hamad • Rassem • Mahamat</p>
        <p style='font-size: 0.7em; color: #888;'>Projet master 2 Deep learning</p>
    </div>

""", unsafe_allow_html=True)
