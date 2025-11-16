import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt  # <--- Garantir que está importado
import itertools
from collections import Counter
import warnings
import urllib3
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from datetime import datetime
from typing import List, Tuple, Dict, Any

# --- 0. CONFIGURAÇÕES E CONSTANTES ---

# Ignorar avisos para um app mais limpo
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuração da página do Streamlit
st.set_page_config(layout="wide", page_title="Análise Mega-Sena", page_icon="🎲")

# Definir constantes globais para nomes de colunas
COLUNAS_BOLAS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
COLUNAS_BASE = ['Concurso', 'Data'] + COLUNAS_BOLAS

# --- 1. FUNÇÕES MODULARIZADAS DE CARREGAMENTO E PROCESSAMENTO DE DADOS ---

@st.cache_data(ttl=3600)
def carregar_dados_caixa() -> pd.DataFrame | None:
    """
    Baixa os dados mais recentes da Caixa, carrega em um DataFrame
    e pré-processa as colunas.
    """
    folder = 'dados_mega_sena'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    url = "https://servicebus2.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Mega-Sena"
    caminho_arquivo = os.path.join(folder, 'mega_sena.xlsx')
    
    try:
        response = requests.get(url, verify=False, timeout=30)
        response.raise_for_status()
        with open(caminho_arquivo, 'wb') as arquivo:
            arquivo.write(response.content)
    except Exception as e:
        st.error(f"Erro ao baixar os dados: {e}. Usando dados locais, se existirem.")
        if not os.path.exists(caminho_arquivo):
            return None

    try:
        df = pd.read_excel(caminho_arquivo, header=1)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Excel: {e}")
        return None

    # Renomear colunas para robustez
    try:
        nomes_colunas = {
            df.columns[0]: 'Concurso',
            df.columns[1]: 'Data',
            df.columns[2]: 'B1',
            df.columns[3]: 'B2',
            df.columns[4]: 'B3',
            df.columns[5]: 'B4',
            df.columns[6]: 'B5',
            df.columns[7]: 'B6'
        }
        df = df.rename(columns=nomes_colunas)
        
        # Selecionar apenas as colunas que importam
        df = df[COLUNAS_BASE]
        
        # Limpeza e conversão de tipos
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        df['Concurso'] = pd.to_numeric(df['Concurso'], errors='coerce')
        for col in COLUNAS_BOLAS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Remover linhas onde a conversão falhou
        df = df.dropna(subset=COLUNAS_BASE)

        # Garantir tipos inteiros para as bolas e concurso
        for col in ['Concurso'] + COLUNAS_BOLAS:
             df[col] = df[col].astype(int)

    except Exception as e:
        st.error(f"Erro ao renomear ou processar colunas: {e}. Verifique o formato do arquivo da Caixa.")
        return None

    # Ordenar por data (mais recente primeiro)
    df = df.sort_values('Data', ascending=False).reset_index(drop=True)
    return df

def validar_dados(df: pd.DataFrame | None) -> bool:
    """Verifica se o DataFrame é válido e contém as colunas necessárias."""
    if df is None or df.empty:
        return False
    
    return all(col in df.columns for col in COLUNAS_BASE)

# Funções de análise modularizadas e cacheadas
@st.cache_data
def get_frequencia(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Calcula a frequência total de todos os números."""
    todos_numeros = df[COLUNAS_BOLAS].values.flatten()
    frequencia = Counter(todos_numeros)
    return sorted(frequencia.items(), key=lambda x: x[1], reverse=True)

@st.cache_data
def get_pares_impares(df: pd.DataFrame) -> pd.Series:
    """Calcula o percentual de sorteios por quantidade de números ímpares."""
    def calcular_impares(linha):
        return sum(1 for num in linha if num % 2 == 1)
    
    df['Qtd_Impares'] = df[COLUNAS_BOLAS].apply(calcular_impares, axis=1)
    return df['Qtd_Impares'].value_counts(normalize=True).sort_index() * 100

@st.cache_data
def get_frequencia_faixas(df: pd.DataFrame) -> pd.Series:
    """Calcula a frequência de números por faixa (1-10, 11-20, etc.)."""
    todos_numeros_flat = df[COLUNAS_BOLAS].values.flatten()
    faixas_bins = [0, 10, 20, 30, 40, 50, 60]
    labels_faixas = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60']
    freq_faixas = pd.cut(todos_numeros_flat, bins=faixas_bins, labels=labels_faixas).value_counts().sort_index()
    return freq_faixas

@st.cache_data
def get_atrasados(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Calcula há quantos concursos cada número não sai."""
    ultimo_sorteio_numero = df['Concurso'].iloc[0]
    
    ultima_aparicao = {}
    for _, linha in df.iterrows():
        numero_sorteio = linha['Concurso']
        numeros_sorteados = linha[COLUNAS_BOLAS].values
        for num in numeros_sorteados:
            if num not in ultima_aparicao:
                ultima_aparicao[num] = numero_sorteio
                
    atrasos_dict = {}
    todos_os_numeros = set(range(1, 61))
    for num in todos_os_numeros:
        if num in ultima_aparicao:
            atrasos_dict[num] = ultimo_sorteio_numero - ultima_aparicao[num]
        else:
            atrasos_dict[num] = ultimo_sorteio_numero 
    
    return sorted(atrasos_dict.items(), key=lambda x: x[1], reverse=True)

@st.cache_data
def get_quentes_frios(df: pd.DataFrame, window: int = 50) -> Tuple[List, List]:
    """Calcula os números mais (quentes) e menos (frios) frequentes na janela."""
    ultimos_sorteios = df.head(window)
    numeros_recentes = ultimos_sorteios[COLUNAS_BOLAS].values.flatten()
    freq_recentes = Counter(numeros_recentes)
    
    for num in range(1, 61):
        if num not in freq_recentes:
            freq_recentes[num] = 0
            
    freq_ordenada = freq_recentes.most_common()
    
    numeros_quentes = freq_ordenada[:20]
    numeros_frios = freq_ordenada[-20:][::-1]
    
    return numeros_quentes, numeros_frios

@st.cache_data
def get_combinacoes(df: pd.DataFrame) -> Tuple[List, List]:
    """Calcula as duplas e triplas mais frequentes."""
    todas_duplas = Counter()
    todas_triplas = Counter()
    
    for _, linha in df.iterrows():
        numeros = sorted(linha[COLUNAS_BOLAS].values)
        for dupla in itertools.combinations(numeros, 2):
            todas_duplas[dupla] += 1
        for tripla in itertools.combinations(numeros, 3):
            todas_triplas[tripla] += 1
            
    return todas_duplas.most_common(30), todas_triplas.most_common(30)

@st.cache_data
def get_vizinhos(df: pd.DataFrame, num_analisar: int) -> List[Tuple[int, int]]:
    """Analisa os 'vizinhos' de um número específico."""
    vizinhos = Counter()
    for _, linha in df.iterrows():
        numeros = linha[COLUNAS_BOLAS].values
        if num_analisar in numeros:
            for outro_num in numeros:
                if num_analisar != outro_num:
                    vizinhos[outro_num] += 1
    return vizinhos.most_common(10)


# --- 2. FUNÇÕES DE ANÁLISE E MODELAGEM (ML) ---

def safe_choice(arr: List, size: int) -> List:
    """ Retorna uma amostra de 'arr', usando 'replace=True' se 'arr' for menor que 'size'."""
    if not isinstance(arr, list): 
        arr = list(arr)
    if not arr:
        return []
        
    replace = len(arr) < size
    return np.random.choice(arr, size, replace=replace).tolist()

def criar_features_melhoradas(df: pd.DataFrame, lookback_window: int = 80) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Cria o dataset de features e alvos para o modelo de ML."""
    features_list = []
    targets_list = []
    
    df_ml = df.sort_values('Data', ascending=True).reset_index(drop=True)
    
    for i in range(lookback_window, len(df_ml)-1):
        numeros_atual = df_ml.loc[i, COLUNAS_BOLAS].values
        
        window_data = []
        recent_15 = []
        
        for j in range(i-lookback_window, i):
            window_numeros = df_ml.loc[j, COLUNAS_BOLAS].values
            window_data.extend(window_numeros)
            if j >= i-15:
                recent_15.extend(window_numeros)
        
        freq_window = pd.Series(window_data).value_counts()
        freq_recent = pd.Series(recent_15).value_counts()
        
        ultima_aparicao = {}
        df_window_recente = df_ml.iloc[max(0, i-300):i]
        
        for num in range(1, 61):
            idx_aparicao = (df_window_recente[COLUNAS_BOLAS] == num).any(axis=1)
            if idx_aparicao.any():
                ultimo_idx = df_window_recente[idx_aparicao].index[-1]
                ultima_aparicao[num] = i - ultimo_idx
            else:
                ultima_aparicao[num] = 300
        
        base_features = [
            sum(numeros_atual),
            sum(1 for x in numeros_atual if x % 2 == 0),
            sum(1 for x in numeros_atual if x <= 30),
            max(numeros_atual) - min(numeros_atual),
            np.std(numeros_atual),
            len(set([x % 10 for x in numeros_atual])),
            sum(numeros_atual) / 6,
            len([x for x in numeros_atual if x in range(1, 21)]),
            len([x for x in numeros_atual if x in range(41, 61)]),
        ]
        
        proximo_sorteio = df_ml.loc[i+1, COLUNAS_BOLAS].values
        
        for num in range(1, 61):
            num_features = base_features.copy()
            num_features.extend([
                num,
                freq_window.get(num, 0),
                freq_recent.get(num, 0),
                ultima_aparicao[num],
                freq_window.get(num, 0) - freq_recent.get(num, 0),
                num % 2,
                1 if num <= 30 else 0,
                num % 10,
                sum(1 for x in numeros_atual if abs(x - num) <= 3), 
                len([x for x in numeros_atual if x // 10 == num // 10]),
                1 if num in [x for x in range(1, 61) if x % 5 == 0] else 0,
            ])
            
            target = 1 if num in proximo_sorteio else 0
            
            features_list.append(num_features)
            targets_list.append(target)
    
    return np.array(features_list), np.array(targets_list), df_ml

def gerar_previsoes_proximo_sorteio(df_ml: pd.DataFrame, model: Any, scaler: StandardScaler, lookback_window: int = 80) -> List[Tuple[int, float]]:
    """Gera features do estado ATUAL para prever o PRÓXIMO sorteio."""
    i = len(df_ml)
    
    window_data = []
    recent_15 = []
    
    for j in range(i-lookback_window, i):
        numeros = df_ml.loc[j, COLUNAS_BOLAS].values
        window_data.extend(numeros)
        if j >= i-15:
            recent_15.extend(numeros)
    
    freq_window = pd.Series(window_data).value_counts()
    freq_recent = pd.Series(recent_15).value_counts()
    
    ultima_aparicao = {}
    df_window_recente = df_ml.iloc[max(0, i-300):i]
    for num in range(1, 61):
        idx_aparicao = (df_window_recente[COLUNAS_BOLAS] == num).any(axis=1)
        if idx_aparicao.any():
            ultimo_idx = df_window_recente[idx_aparicao].index[-1]
            ultima_aparicao[num] = i - ultimo_idx
        else:
            ultima_aparicao[num] = 300
    
    ultimo_sorteio = df_ml.loc[i-1, COLUNAS_BOLAS].values
    base_features = [
        sum(ultimo_sorteio),
        sum(1 for x in ultimo_sorteio if x % 2 == 0),
        sum(1 for x in ultimo_sorteio if x <= 30),
        max(ultimo_sorteio) - min(ultimo_sorteio),
        np.std(ultimo_sorteio),
        len(set([x % 10 for x in ultimo_sorteio])),
        sum(ultimo_sorteio) / 6,
        len([x for x in ultimo_sorteio if x in range(1, 21)]),
        len([x for x in ultimo_sorteio if x in range(41, 61)]),
    ]
    
    probabilidades = []
    for num in range(1, 61):
        features = base_features.copy()
        features.extend([
            num,
            freq_window.get(num, 0),
            freq_recent.get(num, 0),
            ultima_aparicao[num],
            freq_window.get(num, 0) - freq_recent.get(num, 0),
            num % 2,
            1 if num <= 30 else 0,
            num % 10,
            sum(1 for x in ultimo_sorteio if abs(x - num) <= 3),
            len([x for x in ultimo_sorteio if x // 10 == num // 10]),
            1 if num in [x for x in range(1, 61) if x % 5 == 0] else 0,
        ])
        
        features_scaled = scaler.transform([features])
        prob = model.predict_proba(features_scaled)[0][1]
        probabilidades.append((num, prob))
    
    return sorted(probabilidades, key=lambda x: x[1], reverse=True)


@st.cache_data
def gerar_combinacoes_sugeridas(previsoes: List[Tuple[int, float]], n_combinacoes: int = 8) -> List[List[int]]:
    """Gera combinações heurísticas baseadas nos números mais prováveis."""
    numeros = [num for num, prob in previsoes[:25]]
    
    pares = [x for x in numeros if x % 2 == 0]
    impares = [x for x in numeros if x % 2 == 1]
    baixos = [x for x in numeros if x <= 30]
    altos = [x for x in numeros if x > 30]

    combinacoes = []
    
    combinacoes.append(sorted(numeros[:6]))
    combinacoes.append(sorted(safe_choice(numeros, 6)))
    combinacoes.append(sorted(safe_choice(pares, 3) + safe_choice(impares, 3)))
    combinacoes.append(sorted(safe_choice(pares, 2) + safe_choice(impares, 4)))
    combinacoes.append(sorted(safe_choice(pares, 4) + safe_choice(impares, 2)))
    combinacoes.append(sorted(safe_choice(baixos, 3) + safe_choice(altos, 3)))
    combinacoes.append(sorted(safe_choice(baixos, 2) + safe_choice(altos, 4)))
    combinacoes.append(sorted(safe_choice(baixos, 4) + safe_choice(altos, 2)))
    
    combinacoes_unicas = list(set(tuple(c) for c in combinacoes if len(c) == 6))
    return [list(c) for c in combinacoes_unicas][:n_combinacoes]

@st.cache_resource(ttl=3600)
def treinar_modelo(df: pd.DataFrame) -> Tuple[Any, StandardScaler, pd.DataFrame]:
    """
    Cria features, treina o modelo (RandomForest) e o scaler.
    Retorna (modelo, scaler, df_ml)
    """
    if len(df) < 100:
        raise ValueError("Dados insuficientes para treinar o modelo. São necessários pelo menos 100 sorteios.")
        
    X, y, df_ml = criar_features_melhoradas(df)
    
    if len(X) == 0:
        raise ValueError("Não foi possível gerar features. Verifique o lookback_window e o tamanho dos dados.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    split_idx = int(0.7 * len(X))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=30,
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    
    modelo.fit(X_train, y_train)
    
    return modelo, scaler, df_ml


# --- 4. INTERFACE DO STREAMLIT ---

st.title("🎲 Análise de Dados da Mega-Sena")
st.markdown("Uma ferramenta interativa para análise de resultados históricos e tendências.")

# Carregar os dados
df = carregar_dados_caixa()

if not validar_dados(df):
    st.error("Não foi possível carregar ou validar os dados. Verifique a conexão ou o arquivo local.")
else:
    # --- BARRA LATERAL ---
    st.sidebar.title("Navegação")
    pagina = st.sidebar.radio("Escolha uma análise:", 
                              ("Visão Geral", 
                               "Frequência dos Números", 
                               "Pares, Ímpares e Faixas", 
                               "Análise de Combinações", 
                               "Quentes, Frios e Atrasados", 
                               "Modelo Preditivo (ML)"))

    st.sidebar.markdown("---")
    st.sidebar.info(
        "🔍 **Sobre este app:** Análise estatística dos resultados históricos da Mega-Sena. "
        "Desenvolvido para fins educacionais."
    )
    
    # --- PÁGINA 1: VISÃO GERAL ---
    if pagina == "Visão Geral":
        st.header("📊 Visão Geral dos Sorteios")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Sorteios Analisados", f"{len(df):,}")
        col2.metric("Sorteio Mais Recente", f"{df['Concurso'].iloc[0]} ({df['Data'].iloc[0].strftime('%d/%m/%Y')})")
        col3.metric("Sorteio Mais Antigo", f"{df['Concurso'].iloc[-1]} ({df['Data'].iloc[-1].strftime('%d/%m/%Y')})")
        
        st.subheader("Últimos 30 Sorteios (Mais Recentes)")
        st.dataframe(df.head(30)[COLUNAS_BASE], use_container_width=True, hide_index=True)
        
        st.subheader("Estatísticas Adicionais")
        col1, col2 = st.columns(2)
        
        with col1:
            sequencias = 0
            for _, linha in df.iterrows():
                numeros = sorted(linha[COLUNAS_BOLAS].values)
                for i in range(len(numeros)-1):
                    if numeros[i+1] - numeros[i] == 1:
                        sequencias += 1
                        break
            st.metric("Sorteios com Nº Consecutivos", f"{sequencias:,} ({sequencias/len(df)*100:.1f}%)")

        with col2:
            def is_prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5)+1):
                    if n % i == 0: return False
                return True
            
            todos_numeros_flat = df[COLUNAS_BOLAS].values.flatten()
            primos = sum(1 for num in todos_numeros_flat if is_prime(num))
            st.metric("Total de Números Primos Sorteados", f"{primos:,} ({primos/len(todos_numeros_flat)*100:.1f}%)")

        st.subheader("Sorteios por Ano")
        df['Ano'] = df['Data'].dt.year
        freq_por_ano = df.groupby('Ano').size().sort_index()
        freq_por_ano.name = "Total de Sorteios"
        st.line_chart(freq_por_ano)

    # --- PÁGINA 2: FREQUÊNCIA DOS NÚMEROS ---
    elif pagina == "Frequência dos Números":
        st.header("📈 Frequência dos Números Sorteados (Total)")
        st.markdown("Frequência de cada número (1 a 60) em todos os sorteios.")

        frequencia_ordenada = get_frequencia(df)
        freq_df = pd.DataFrame(frequencia_ordenada, columns=['Numero', 'Frequencia']).set_index('Numero')
        
        st.bar_chart(freq_df.sort_index())
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔥 Top 10 Mais Frequentes")
            st.dataframe(freq_df.sort_values('Frequencia', ascending=False).head(10), use_container_width=True)
        with col2:
            st.subheader("❄️ Top 10 Menos Frequentes")
            st.dataframe(freq_df.sort_values('Frequencia', ascending=True).head(10), use_container_width=True)

    # --- PÁGINA 3: PARES, ÍMPARES E FAIXAS ---
    elif pagina == "Pares, Ímpares e Faixas":
        st.header("⚖️ Análise de Pares, Ímpares e Faixas")
        
        # Carregar dados para esta página
        percentuais_impares = get_pares_impares(df)
        freq_faixas = get_frequencia_faixas(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # --- MUDANÇA: Gráfico de Pizza (Matplotlib) ---
            st.subheader("Distribuição de Ímpares (Pizza)")
            st.markdown("Percentual de sorteios por quantidade de números ímpares.")
            
            fig, ax = plt.subplots()
            ax.pie(
                percentuais_impares, 
                labels=[f'{i} ímpares' for i in percentuais_impares.index], 
                autopct='%1.1f%%', 
                startangle=90
            )
            ax.axis('equal')  # Assegura que o gráfico seja um círculo
            st.pyplot(fig)
            # --- Fim da Mudança ---

            st.subheader("Distribuição de Ímpares (Barras)")
            st.markdown("Mesmos dados em gráfico de barras para comparação.")
            
            df_impares = percentuais_impares.reset_index()
            df_impares.columns = ['Qtd Ímpares', 'Percentual']
            df_impares['Legenda'] = df_impares['Qtd Ímpares'].apply(lambda x: f"{x} Ímpares")
            st.bar_chart(df_impares.set_index('Legenda')['Percentual'])

        with col2:
            st.subheader("Frequência por Faixa de Números")
            st.markdown("Frequência total de números sorteados em cada faixa de 10.")
            
            st.bar_chart(freq_faixas)

    # --- PÁGINA 4: ANÁLISE DE COMBINAÇÕES ---
    elif pagina == "Análise de Combinações":
        st.header("🤝 Combinações e Vizinhos")
        
        tipo_analise = st.selectbox("Escolha o tipo de combinação:", ["Duplas (Top 30)", "Triplas (Top 30)", "Números Vizinhos"])
        
        top_duplas, top_triplas = get_combinacoes(df)

        if tipo_analise == "Duplas (Top 30)":
            st.subheader("🏆 Top 30 Duplas Mais Frequentes")
            df_duplas = pd.DataFrame(top_duplas, columns=['Dupla', 'Frequência'])
            st.dataframe(df_duplas, use_container_width=True, hide_index=True)

        elif tipo_analise == "Triplas (Top 30)":
            st.subheader("🏆 Top 30 Triplas Mais Frequentes")
            df_triplas = pd.DataFrame(top_triplas, columns=['Tripla', 'Frequência'])
            st.dataframe(df_triplas, use_container_width=True, hide_index=True)

        elif tipo_analise == "Números Vizinhos":
            st.subheader("🔍 Análise de Números Vizinhos")
            st.markdown("Veja quais números mais saem juntos com um número específico.")
            num_analisar = st.number_input("Digite um número de 1 a 60:", min_value=1, max_value=60, value=5)
            
            vizinhos_data = get_vizinhos(df, num_analisar)
            
            if vizinhos_data:
                st.subheader(f"Parceiros mais comuns do número {num_analisar}:")
                df_vizinhos = pd.DataFrame(vizinhos_data, columns=['Número Vizinho', 'Frequência']).set_index('Número Vizinho')
                
                st.bar_chart(df_vizinhos)
                st.dataframe(df_vizinhos, use_container_width=True)
            else:
                st.warning(f"O número {num_analisar} não foi encontrado em nenhum sorteio.")

    # --- PÁGINA 5: QUENTES, FRIOS E ATRASADOS ---
    elif pagina == "Quentes, Frios e Atrasados":
        st.header("🔥❄️⏰ Números Quentes, Frios e Atrasados")
        
        window = st.slider("Janela de análise para Quentes/Frios (nº de sorteios):", min_value=10, max_value=100, value=50, step=5)
        
        atrasados_ordenados = get_atrasados(df)
        numeros_quentes, numeros_frios = get_quentes_frios(df, window)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader(f"🔥 Quentes (Últimos {window})")
            st.markdown(f"Mais sorteados nos últimos {window} concursos.")
            df_quentes = pd.DataFrame(numeros_quentes, columns=['Número', 'Frequência'])
            st.dataframe(df_quentes, use_container_width=True, hide_index=True)
            
        with col2:
            st.subheader(f"❄️ Frios (Últimos {window})")
            st.markdown(f"Menos sorteados (ou ausentes) nos últimos {window} concursos.")
            df_frios = pd.DataFrame(numeros_frios, columns=['Número', 'Frequência'])
            st.dataframe(df_frios, use_container_width=True, hide_index=True)

        with col3:
            st.subheader("⏰ Atrasados (Geral)")
            st.markdown("Não saem há mais tempo (em nº de concursos).")
            df_atrasados = pd.DataFrame(atrasados_ordenados[:20], columns=['Número', 'Concursos Atrás'])
            st.dataframe(df_atrasados, use_container_width=True, hide_index=True)

    # --- PÁGINA 6: MODELO PREDITIVO ---
    elif pagina == "Modelo Preditivo (ML)":
        st.header("🤖 Modelo Preditivo (Machine Learning)")
        st.markdown(
            "Esta seção usa um modelo de *Random Forest* para prever a probabilidade "
            "de cada número sair no próximo sorteio, com base em padrões históricos."
        )
        
        st.warning(
            "🚨 **LEMBRETE CRÍTICO:** Este modelo é uma ferramenta estatística para fins "
            "educacionais e de análise. **Não é uma garantia de resultado.** A loteria é "
            "um jogo de azar. Jogue com responsabilidade."
        )
        
        aceito = st.checkbox("Entendo que isso é apenas uma análise estatística e não garante resultados")
        
        if aceito:
            # --- MUDANÇA: Tempo de ML atualizado para 5-10 min ---
            if st.button("Treinar Modelo e Gerar Previsões (pode levar 5-10 min)"):
                with st.spinner("Treinando modelo e analisando dados... Por favor, aguarde."):
                    try:
                        # Lógica de ML modularizada
                        
                        # 1. Treinar (ou carregar do cache de recursos)
                        modelo, scaler, df_ml = treinar_modelo(df)
                        
                        # 2. Gerar previsões para o próximo sorteio
                        previsoes = gerar_previsoes_proximo_sorteio(df_ml, modelo, scaler)
                        
                        # 3. Gerar combinações sugeridas
                        combinacoes = gerar_combinacoes_sugeridas(previsoes)
                        
                        st.success("Modelo treinado e previsões geradas!")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.subheader("🎯 Top 20 Números Preditos")
                            df_previsoes = pd.DataFrame(previsoes[:20], columns=['Número', 'Probabilidade'])
                            df_previsoes['Probabilidade'] = df_previsoes['Probabilidade'].map(lambda x: f"{x*100:.2f}%")
                            st.dataframe(df_previsoes, use_container_width=True, hide_index=True)

                        with col2:
                            st.subheader("💡 Combinações Sugeridas")
                            st.markdown("Combinações geradas pelo modelo, diversificando as escolhas com maior probabilidade.")
                            
                            for i, comb in enumerate(combinacoes, 1):
                                soma = sum(comb)
                                pares = sum(1 for x in comb if x % 2 == 0)
                                impares = 6 - pares
                                baixos = sum(1 for x in comb if x <= 30)
                                altos = 6 - baixos
                                
                                comb_str = ', '.join(map(str, sorted(comb)))
                                
                                with st.expander(f"**Combinação {i}:** {comb_str}"):
                                    c1, c2, c3 = st.columns(3)
                                    c1.metric("Soma", soma)
                                    c2.metric("Pares/Ímpares", f"{pares}P / {impares}I")
                                    c3.metric("Baixos/Altos", f"{baixos}B / {altos}A")
                    
                    except Exception as e:
                        st.error(f"❌ Erro during training: {str(e)}")
                        st.exception(e) 
                        st.info("💡 Tente recarregar a página (F5) ou verificar os dados de origem.")
        else:
            st.info("📝 Marque a caixa acima para habilitar o modelo preditivo.")