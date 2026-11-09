"""
ETAPA 4b: previsores intra-safra de seca e de geada (Tabela 4 do TCC)
====================================================================

Pergunta
--------
Dá para estimar a safra ANTES da colheita, num evento de seca ou de geada,
melhor do que repetindo o histórico?

Janela expansiva (walk-forward)
-------------------------------
Diferente do K-Fold da etapa 4a, aqui cada safra de teste é estimada por um
modelo treinado SÓ com as safras anteriores a ela. É a situação real de quem
quer prever a safra em curso: não há nenhuma informação do futuro no treino.

Os dois previsores
------------------
* SECA, testado nas safras de 2014 a 2017 (N = 40):
    controles + déficit hídrico na florada (set-nov de t-1) + déficit na
    expansão dos grãos (dez de t-1 a mar de t) + temperatura máxima média da
    expansão. Tudo isso é conhecido em março, antes da colheita (maio-agosto).
* GEADA, testado nas safras de 1994, 1995, 2000, 2001, 2021, 2022 e 2023 (N = 70):
    controles + temperatura mínima, dias abaixo de 3 °C e chuva do inverno
    ANTERIOR (jun-ago de t-1), porque a geada de julho queima as gemas que
    dariam a florada e a safra seguinte.

Hiperparâmetros fixos (200 árvores, profundidade 5, mínimo de 5 amostras por
divisão): o primeiro treino é pequeno (N = 50) para uma busca em grade.

Comparações: persistência histórica M(t) e o mesmo Random Forest só com os
controles. Incerteza: intervalo P10-P90 das previsões das 200 árvores; se o
intervalo estivesse bem calibrado, cobriria cerca de 80% dos valores reais.

Como rodar
----------
    python etapa4_modelagem/4b_previsores_intrasafra.py   (~30 s)

Saídas em resultados/: 4b_previsores.json, tabela4_previsores.csv,
previsor_seca.csv, previsor_geada.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CONTROLES, DADOS_BRUTOS, DADOS_PROC, KG_POR_SACA, MUN,  # noqa: E402
                    RESULTADOS, SEMENTE, salvar_json)

PARAMS = dict(n_estimators=200, max_depth=5, min_samples_split=5, random_state=SEMENTE)

PREVISORES = {
    "seca": {
        "variaveis": CONTROLES + ["DEF_Florada", "DEF_Expansao_Graos", "Tmax_Expansao_Graos"],
        "safras_teste": [2014, 2015, 2016, 2017],
    },
    "geada": {
        "variaveis": CONTROLES + ["Tmin_Abs_Inverno_Risco_Geada_t1", "Dias_Abaixo_3C_t1",
                                  "Precip_Inverno_Risco_Geada_t1"],
        "safras_teste": [1994, 1995, 2000, 2001, 2021, 2022, 2023],
    },
}


def preparar():
    df = pd.read_csv(DADOS_PROC / "matriz_mestra.csv")
    bh = pd.read_csv(DADOS_BRUTOS / "balanco_hidrico_openmeteo_1988_2023.csv")
    df = df.merge(bh, on=[MUN, ANO], how="left").sort_values([MUN, ANO]).reset_index(drop=True)
    g = df.groupby(MUN)[ALVO]
    df["media_movel_3anos"] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["anomalia_bienal_t_1"] = g.shift(1) - df.media_movel_3anos
    df["tendencia_temporal"] = df[ANO] - df[ANO].min()
    for col in ["Tmin_Abs_Inverno_Risco_Geada", "Dias_Abaixo_3C", "Precip_Inverno_Risco_Geada"]:
        df[col + "_t1"] = df.groupby(MUN)[col].shift(1)
    return df


def janela_expansiva(df, variaveis, safras):
    d = df.dropna(subset=variaveis + [ALVO, "media_movel_3anos"]).reset_index(drop=True)
    linhas = []
    for safra in safras:
        treino, teste = d[d[ANO] < safra], d[d[ANO] == safra]
        rf = RandomForestRegressor(**PARAMS).fit(treino[variaveis], treino[ALVO])
        so_controles = RandomForestRegressor(**PARAMS).fit(treino[CONTROLES], treino[ALVO])
        arvores = np.array([a.predict(teste[variaveis].values) for a in rf.estimators_])
        for i, (_, lin) in enumerate(teste.iterrows()):
            linhas.append({
                MUN: lin[MUN], ANO: safra, "N_treino": len(treino),
                "real": lin[ALVO] / KG_POR_SACA,
                "rf": rf.predict(teste[variaveis].iloc[[i]])[0] / KG_POR_SACA,
                "so_controles": so_controles.predict(teste[CONTROLES].iloc[[i]])[0] / KG_POR_SACA,
                "baseline_M": lin.media_movel_3anos / KG_POR_SACA,
                "p10": np.percentile(arvores[:, i], 10) / KG_POR_SACA,
                "p90": np.percentile(arvores[:, i], 90) / KG_POR_SACA,
            })
    r = pd.DataFrame(linhas)
    for c in ["rf", "so_controles", "baseline_M"]:
        r["erro_" + c] = (r[c] - r.real).abs()
    r["dentro_p10_p90"] = r.real.between(r.p10, r.p90)
    return r


def resumir(r):
    return {
        "N": len(r),
        "MAE_rf": round(r.erro_rf.mean(), 2),
        "MAE_so_controles": round(r.erro_so_controles.mean(), 2),
        "MAE_baseline_M": round(r.erro_baseline_M.mean(), 2),
        "p_wilcoxon_rf_vs_baseline": round(float(wilcoxon(r.erro_rf, r.erro_baseline_M).pvalue), 3),
        # erro percentual médio (MAPE), comparável ao de Kouadio et al. (2021)
        "erro_percentual_rf": round(float((r.erro_rf / r.real).mean() * 100), 1),
        "cobertura_p10_p90_pct": round(float(r.dentro_p10_p90.mean() * 100), 1),
        "MAE_por_safra": r.groupby(ANO)[["erro_rf", "erro_so_controles", "erro_baseline_M"]].mean().round(2)
                          .to_dict("index"),
    }


df = preparar()
saida, tabela = {}, []
for nome, cfg in PREVISORES.items():
    r = janela_expansiva(df, cfg["variaveis"], cfg["safras_teste"])
    r.to_csv(RESULTADOS / f"previsor_{nome}.csv", index=False)
    saida[nome] = resumir(r)
    tabela.append([nome, ", ".join(map(str, cfg["safras_teste"])), *[saida[nome][k] for k in
                  ["N", "MAE_rf", "MAE_so_controles", "MAE_baseline_M", "p_wilcoxon_rf_vs_baseline"]]])
    if nome == "geada":
        # Sensibilidade: sem 2000 e 2001, safras com valores repetidos no IBGE.
        # Exclusão definida DEPOIS de ver os resultados; lida só como sensibilidade.
        s = r[~r[ANO].isin([2000, 2001])]
        saida["geada_sem_2000_2001"] = resumir(s)
        tabela.append(["geada (sem 2000 e 2001)", "1994, 1995, 2021, 2022, 2023",
                       *[saida["geada_sem_2000_2001"][k] for k in
                         ["N", "MAE_rf", "MAE_so_controles", "MAE_baseline_M", "p_wilcoxon_rf_vs_baseline"]]])

salvar_json(saida, "4b_previsores.json")
pd.DataFrame(tabela, columns=["previsor", "safras", "N", "MAE_RF", "MAE_so_controles", "MAE_baseline",
                              "p_Wilcoxon_RF_vs_baseline"]).to_csv(RESULTADOS / "tabela4_previsores.csv", index=False)

for nome, s in saida.items():
    print(f"{nome:22s} N={s['N']:3d}  RF {s['MAE_rf']:.2f}  só controles {s['MAE_so_controles']:.2f}  "
          f"baseline {s['MAE_baseline_M']:.2f}  p={s['p_wilcoxon_rf_vs_baseline']}  "
          f"cobertura P10-P90 {s['cobertura_p10_p90_pct']}%  erro {s['erro_percentual_rf']}%")
