"""
ETAPA 5i: robustez da comparação entre os cenários 1 a 4
=========================================================

Três verificações pedidas na preparação da defesa:

1. Safra inteira fora do treino para os quatro cenários (não só para o
   completo). No K-Fold, a tendência T(t) e os vizinhos da mesma safra já
   informam o ano, e o clima, que varia sobretudo entre safras, fica redundante.
   Com a safra inteira fora, o modelo não conhece o nível daquela safra, e o
   clima teria a chance de fazer diferença. Comparam-se os cenários 2, 3 e 4
   com o 1 (Wilcoxon por observação e por safra, Holm sobre os seis testes) e
   cada cenário com a persistência histórica.

2. Intervalo de confiança de 95% da diferença de MAE entre cada cenário e o
   Cenário 1, por reamostragem das 35 safras (bootstrap com 10.000 repetições),
   no K-Fold aninhado e com a safra inteira fora. A reamostragem é por safra
   porque os municípios de uma mesma safra não são independentes.

3. Sensibilidade ao sorteio de variáveis: no padrão do scikit-learn, todas as
   variáveis são candidatas em cada divisão (max_features = 1,0). Repete-se o
   K-Fold aninhado com max_features = "sqrt", o sorteio de variáveis do
   Random Forest de Breiman (2001).

Tudo na série completa (N = 350) e sem as safras de 1991 a 2002 (N = 230).

    python etapa5_avaliacao/5i_robustez.py   (~15 min com 2 núcleos)

Saídas: resultados/5i_robustez.json, resultados/5i_predicoes_safra_fora.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CLIMA, CONTROLES, GRADE, KG_POR_SACA, MUN, NDVI, RESULTADOS,  # noqa: E402
                    SAFRAS_REPETIDAS, SEMENTE, busca_em_grade, carregar_base, holm, kfold_externo,
                    metricas, salvar_json, wilcoxon_obs_e_safra)

CENARIOS = {"C1": CONTROLES, "C2": CONTROLES + CLIMA, "C3": CONTROLES + NDVI, "C4": CONTROLES + CLIMA + NDVI}
B = 10_000


def safra_fora(dados, variaveis):
    """Cada safra (10 municípios) estimada pelo modelo ajustado com as outras."""
    X, y = dados[variaveis], dados[ALVO]
    pred = np.zeros(len(y))
    for treino, teste in LeaveOneGroupOut().split(X, y, dados[ANO]):
        pred[teste] = busca_em_grade().fit(X.iloc[treino], y.iloc[treino]).predict(X.iloc[teste])
    return pred


def aninhada_sqrt(dados, variaveis):
    """K-Fold aninhado igual ao da etapa 4a, com sorteio de variáveis (sqrt)."""
    X, y = dados[variaveis], dados[ALVO]
    pred = np.zeros(len(y))
    for treino, teste in kfold_externo().split(X):
        gs = GridSearchCV(RandomForestRegressor(random_state=SEMENTE, max_features="sqrt"), GRADE, cv=3,
                          scoring="neg_mean_absolute_error", n_jobs=-1)
        pred[teste] = gs.fit(X.iloc[treino], y.iloc[treino]).predict(X.iloc[teste])
    return pred


def ic_diferenca(erro_k, erro_1, anos, rng):
    """IC 95% da diferença de MAE (cenário k − Cenário 1), em sacas/ha, por bootstrap de safras."""
    df = pd.DataFrame({"ano": np.asarray(anos), "d": (np.asarray(erro_k) - np.asarray(erro_1)) / KG_POR_SACA,
                       "n": 1})
    por_safra = df.groupby("ano").agg(soma=("d", "sum"), n=("n", "sum"))
    soma, n = por_safra.soma.values, por_safra.n.values
    idx = rng.integers(0, len(soma), size=(B, len(soma)))
    dif = soma[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {"diferenca": round(float(df.d.mean()), 2),
            "IC95": [round(float(np.percentile(dif, 2.5)), 2), round(float(np.percentile(dif, 97.5)), 2)]}


base = carregar_base()
conjuntos = {"serie_completa": base,
             "sem_1991_2002": base[~base[ANO].isin(SAFRAS_REPETIDAS)].reset_index(drop=True)}
rng = np.random.default_rng(SEMENTE)
kfold = pd.read_csv(RESULTADOS / "cenarios_predicoes.csv")       # estimativas do K-Fold (etapa 5f), sacas/ha
saida, tabela_pred = {}, []
for nome, d in conjuntos.items():
    y = d[ALVO]
    erro_m = np.abs(y - d.media_movel_3anos)
    r = {"N": len(d), "persistencia": metricas(y, d.media_movel_3anos)}

    # 1. safra inteira fora
    sf = {}
    for c, v in CENARIOS.items():
        sf[c] = safra_fora(d, v)
        r[f"{c}_safra_fora"] = metricas(y, sf[c])
        r[f"{c}_safra_fora_vs_persistencia"] = wilcoxon_obs_e_safra(erro_m, np.abs(y - sf[c]), d[ANO])
        print(f"[{nome}] {c} safra fora: MAE {r[f'{c}_safra_fora']['MAE']:.2f}", flush=True)
    brutos = []
    for c in ["C2", "C3", "C4"]:
        w = wilcoxon_obs_e_safra(np.abs(y - sf["C1"]), np.abs(y - sf[c]), d[ANO])
        r[f"{c}_vs_C1_safra_fora"] = w
        brutos += w[:2]
    r["holm_safra_fora_vs_C1"] = [round(float(p), 4) for p in holm(brutos)]

    # 2. IC 95% da diferença de MAE contra o Cenário 1
    k = d[[MUN, ANO]].merge(kfold, on=[MUN, ANO], how="left")
    assert len(k) == len(d) and k[["C1", "C2", "C3", "C4"]].notna().all().all()
    ys = y.values / KG_POR_SACA
    for c in ["C1", "C2", "C3", "C4"]:
        r[f"{c}_kfold_MAE_conferencia"] = round(float(np.mean(np.abs(ys - k[c].values))), 2)
    for c in ["C2", "C3", "C4"]:
        r[f"{c}_menos_C1_kfold"] = ic_diferenca(np.abs(ys - k[c].values) * KG_POR_SACA,
                                                np.abs(ys - k["C1"].values) * KG_POR_SACA, d[ANO], rng)
        r[f"{c}_menos_C1_safra_fora"] = ic_diferenca(np.abs(y - sf[c]), np.abs(y - sf["C1"]), d[ANO], rng)

    # 3. sorteio de variáveis (max_features = sqrt)
    sq = {}
    for c, v in CENARIOS.items():
        sq[c] = aninhada_sqrt(d, v)
        r[f"{c}_kfold_sqrt"] = metricas(y, sq[c])
    for c in ["C2", "C3", "C4"]:
        r[f"{c}_vs_C1_kfold_sqrt"] = wilcoxon_obs_e_safra(np.abs(y - sq["C1"]), np.abs(y - sq[c]), d[ANO])
    print(f"[{nome}] sqrt: " + ", ".join(f"{c} {r[f'{c}_kfold_sqrt']['MAE']:.2f}" for c in CENARIOS), flush=True)

    saida[nome] = r
    if nome == "serie_completa":
        tabela_pred = pd.DataFrame({MUN: d[MUN], ANO: d[ANO], "real": ys,
                                    **{f"{c}_safra_fora": sf[c] / KG_POR_SACA for c in CENARIOS}}).round(2)

salvar_json(saida, "5i_robustez.json")
tabela_pred.to_csv(RESULTADOS / "5i_predicoes_safra_fora.csv", index=False)
print("ok")
