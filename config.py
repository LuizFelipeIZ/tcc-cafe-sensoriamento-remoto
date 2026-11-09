"""
Configurações comuns a todas as etapas do projeto.

Aqui ficam, num lugar só: os caminhos das pastas, os nomes das colunas,
as listas de variáveis de cada grupo (controles, clima e NDVI), a grade de
hiperparâmetros do Random Forest e as funções usadas em várias etapas
(métricas, teste de Wilcoxon por observação e por safra, ajuste de Holm).

Mudar algo aqui muda o projeto inteiro, por isso cada item está comentado.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

# ---------------------------------------------------------------------------
# Pastas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
DADOS_BRUTOS = RAIZ / "dados" / "brutos"            # saídas da etapa 1 (coleta)
DADOS_PROC = RAIZ / "dados" / "processados"         # geradas pelas etapas 2 e 3
LIMITES = RAIZ / "dados" / "limites"                # limites municipais (mapas)
RESULTADOS = RAIZ / "resultados"                    # tabelas e JSON de resultados
FIGURAS = RESULTADOS / "figuras"                    # figuras do TCC

for _p in (DADOS_PROC, RESULTADOS, FIGURAS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colunas e constantes
# ---------------------------------------------------------------------------
ALVO = "Rendimento_kg_ha"   # rendimento médio do IBGE (kg/ha)
MUN = "Municipio"
ANO = "Ano"                 # ano da safra (colheita em maio a agosto do ano t)
KG_POR_SACA = 60            # 1 saca de café beneficiado = 60 kg
SEMENTE = 42                # semente fixa: os resultados se repetem a cada execução

MUNICIPIOS = [
    "Arceburgo", "Cabo Verde", "Guaranesia", "Guaxupe", "Itamogi",
    "Jacui", "Juruaia", "Monte Belo", "Muzambinho", "Nova Resende",
]

# Safras em que o IBGE repete o mesmo rendimento por 2 a 4 anos seguidos
# na maioria dos municípios. As análises são repetidas sem elas.
SAFRAS_REPETIDAS = list(range(1991, 2003))

# ---------------------------------------------------------------------------
# Variáveis preditoras (12 no modelo principal)
# A ORDEM importa para a reprodutibilidade exata do Random Forest.
# ---------------------------------------------------------------------------
CONTROLES = [
    "anomalia_bienal_t_1",   # A(t) = Y(t-1) - M(t): fase do ciclo bienal
    "tendencia_temporal",    # T(t) = t - 1988: ganho tecnológico
]
CLIMA = [
    "Precip_Florada_t1",                 # chuva set-nov do ano t-1
    "Tmin_Abs_Florada_t1",               # mínima absoluta set-nov do ano t-1
    "Precip_Enchimento_Graos",           # chuva jan-abr e dez do ano t
    "Tmin_Abs_Enchimento_Graos",         # mínima absoluta jan-abr e dez do ano t
    "Precip_Inverno_Risco_Geada_t1",     # chuva jun-ago do ano t-1
    "Tmin_Abs_Inverno_Risco_Geada_t1",   # mínima absoluta jun-ago do ano t-1
]
NDVI = [
    "NDVI_Vigor_Maximo",     # percentil 90 do NDVI, jul t-1 a jun t
    "NDVI_Estresse_Maximo",  # percentil 10 do NDVI, jul t-1 a jun t
    "NDVI_Florada",          # média do NDVI set-nov t-1
    "NDVI_Amplitude",        # vigor - estresse (perda de folhas no ano)
]
VARIAVEIS = CONTROLES + CLIMA + NDVI

# Agrupamento das variáveis climáticas por fase fenológica (Figura 3b)
FASES = {
    "Florada (t-1)": ["Precip_Florada_t1", "Tmin_Abs_Florada_t1"],
    "Inverno (t-1)": ["Precip_Inverno_Risco_Geada_t1", "Tmin_Abs_Inverno_Risco_Geada_t1"],
    "Enchimento": ["Precip_Enchimento_Graos", "Tmin_Abs_Enchimento_Graos"],
}

# Nomes legíveis para tabelas e figuras
NOMES = {
    "anomalia_bienal_t_1": "Anomalia bienal (t−1)",
    "tendencia_temporal": "Tendência temporal",
    "Precip_Florada_t1": "Chuva na florada (t−1)",
    "Tmin_Abs_Florada_t1": "Temp. mínima na florada (t−1)",
    "Precip_Enchimento_Graos": "Chuva no enchimento",
    "Tmin_Abs_Enchimento_Graos": "Temp. mínima no enchimento",
    "Precip_Inverno_Risco_Geada_t1": "Chuva no inverno (t−1)",
    "Tmin_Abs_Inverno_Risco_Geada_t1": "Temp. mínima no inverno (t−1)",
    "NDVI_Vigor_Maximo": "NDVI vigor máximo",
    "NDVI_Estresse_Maximo": "NDVI estresse máximo",
    "NDVI_Florada": "NDVI na florada",
    "NDVI_Amplitude": "NDVI amplitude",
}

# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
# Grade da busca de hiperparâmetros (ciclo interno da validação aninhada)
GRADE = {
    "n_estimators": [100, 150],
    "max_depth": [3, 5],
    "min_samples_split": [5, 10],
}


def busca_em_grade():
    """Random Forest com busca em grade em 3 partes (ciclo interno)."""
    return GridSearchCV(
        RandomForestRegressor(random_state=SEMENTE), GRADE, cv=3,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )


def kfold_externo():
    """Ciclo externo da validação: 5 partes, embaralhadas com semente fixa."""
    return KFold(n_splits=5, shuffle=True, random_state=SEMENTE)


def validacao_aninhada(dados, variaveis):
    """K-Fold aninhado: retorna as estimativas fora da amostra (kg/ha) e o
    desvio padrão do MAE entre as 5 partes (sacas/ha).

    Em cada parte do ciclo externo, a busca em grade roda SÓ com as 4 partes
    de treino; a parte de teste nunca é vista no ajuste.
    """
    X, y = dados[variaveis], dados[ALVO]
    pred = np.zeros(len(y))
    mae_partes = []
    for treino, teste in kfold_externo().split(X):
        modelo = busca_em_grade().fit(X.iloc[treino], y.iloc[treino])
        pred[teste] = modelo.predict(X.iloc[teste])
        mae_partes.append(np.mean(np.abs(y.iloc[teste] - pred[teste])) / KG_POR_SACA)
    return pred, float(np.std(mae_partes, ddof=1))


# ---------------------------------------------------------------------------
# Métricas e testes
# ---------------------------------------------------------------------------
def metricas(y_kg, pred_kg):
    """MAE e RMSE em sacas/ha e R² fora da amostra."""
    return {
        "MAE": round(mean_absolute_error(y_kg, pred_kg) / KG_POR_SACA, 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_kg, pred_kg))) / KG_POR_SACA, 2),
        "R2": round(float(r2_score(y_kg, pred_kg)), 3),
    }


def wilcoxon_obs_e_safra(erro_a, erro_b, anos):
    """Compara dois conjuntos de erros absolutos pelo Wilcoxon pareado.

    Retorna [p por observação, p por safra, nº de safras em que B errou
    menos que A, nº de safras]. O teste por safra usa o erro médio de cada
    safra, porque os municípios de uma mesma safra não são independentes
    (compartilham o clima).
    """
    erro_a, erro_b = np.asarray(erro_a), np.asarray(erro_b)
    por_safra = pd.DataFrame({"ano": np.asarray(anos), "a": erro_a, "b": erro_b}).groupby("ano").mean()
    return [
        round(float(wilcoxon(erro_a, erro_b).pvalue), 4),
        round(float(wilcoxon(por_safra.a, por_safra.b).pvalue), 4),
        int((por_safra.b < por_safra.a).sum()),
        len(por_safra),
    ]


def holm(p_valores):
    """Ajuste de Holm (1979) para comparações múltiplas."""
    p = np.asarray(p_valores, dtype=float)
    ordem = np.argsort(p)
    m = len(p)
    ajustado = np.empty(m)
    acumulado = 0.0
    for i, k in enumerate(ordem):
        acumulado = max(acumulado, (m - i) * p[k])
        ajustado[k] = min(1.0, acumulado)
    return ajustado


def carregar_base():
    """Base de modelagem gerada na etapa 3 (N = 350)."""
    caminho = DADOS_PROC / "base_modelagem.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            "base_modelagem.csv não encontrada. Rode antes as etapas 2 e 3:\n"
            "  python etapa2_preprocessamento/2a_auditoria_e_matriz.py\n"
            "  python etapa2_preprocessamento/2b_calibracao_ndvi.py\n"
            "  python etapa3_base/3_montar_base.py"
        )
    return pd.read_csv(caminho)


def salvar_json(obj, nome):
    import json
    caminho = RESULTADOS / nome
    caminho.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return caminho
