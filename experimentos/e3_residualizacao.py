"""
EXPERIMENTO 3: o clima explica o que sobra depois da tendência e da bienalidade?
===============================================================================

Residualização em duas etapas (partialling-out):
  1. Regressão linear do rendimento na tendência e na anomalia bienal,
     ajustada SÓ nas partes de treino. O resíduo é a parte do rendimento
     que a tecnologia e o ciclo bienal não explicam.
  2. Random Forest com clima e/ou NDVI para estimar esse resíduo.
Também calcula a importância (MDI e SHAP) dentro do resíduo, para ver se a
ordem das variáveis climáticas se mantém sem a competição dos controles.

Resultado: clima e NDVI explicam cerca de 26% do resíduo fora da amostra
(38% sem 1991-2002), com a florada (t-1) no topo, como no modelo principal.
Mas a soma linear + RF erra 5,05 sacas/ha, mais que o RF só com os
controles (4,54): o Random Forest aproveita a tendência e a bienalidade de
forma não linear melhor do que a reta. Por isso o modelo final usa as 12
variáveis juntas, e a atribuição ao clima é feita pelo SHAP (etapa 5).

    python experimentos/e3_residualizacao.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CLIMA, KG_POR_SACA, NDVI, SAFRAS_REPETIDAS, busca_em_grade,  # noqa: E402
                    carregar_base, kfold_externo, salvar_json, wilcoxon_obs_e_safra)

CT = ["tendencia_temporal", "anomalia_bienal_t_1"]
base = carregar_base()
saida = {}
for nome_conj, d in [("serie_completa", base),
                     ("sem_1991_2002", base[~base[ANO].isin(SAFRAS_REPETIDAS)].reset_index(drop=True))]:
    y = d[ALVO].values
    r = {"N": len(d)}
    for nome, V in [("clima_e_ndvi", CLIMA + NDVI), ("so_clima", CLIMA), ("so_ndvi", NDVI)]:
        p_lin, p_res, res_teste = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
        for tr, te in kfold_externo().split(d):
            lin = LinearRegression().fit(d.iloc[tr][CT], y[tr])                 # etapa 1, só no treino
            p_lin[te] = lin.predict(d.iloc[te][CT])
            res_teste[te] = y[te] - p_lin[te]
            p_res[te] = busca_em_grade().fit(d.iloc[tr][V], y[tr] - lin.predict(d.iloc[tr][CT])).predict(d.iloc[te][V])
        e_lin, e_comb = np.abs(y - p_lin), np.abs(y - (p_lin + p_res))
        r[nome] = {
            "R2_do_residuo_fora_da_amostra": round(float(r2_score(res_teste, p_res)), 3),
            "MAE_so_linear": round(e_lin.mean() / KG_POR_SACA, 2),
            "MAE_linear_mais_RF": round(e_comb.mean() / KG_POR_SACA, 2),
            "wilcoxon_comb_vs_linear": wilcoxon_obs_e_safra(e_lin, e_comb, d[ANO]),
        }
    resid = y - LinearRegression().fit(d[CT], y).predict(d[CT])
    m = busca_em_grade().fit(d[CLIMA + NDVI], resid).best_estimator_
    imp = pd.Series(m.feature_importances_ * 100, index=CLIMA + NDVI)
    sv = np.abs(shap.TreeExplainer(m).shap_values(d[CLIMA + NDVI])).mean(0)
    r["MDI_no_residuo_pct"] = imp.round(1).sort_values(ascending=False).to_dict()
    r["SHAP_no_residuo_pct"] = pd.Series(sv / sv.sum() * 100, index=CLIMA + NDVI).round(1).sort_values(
        ascending=False).to_dict()
    saida[nome_conj] = r

salvar_json(saida, "e3_residualizacao.json")
for nome_conj, r in saida.items():
    print(f"\n== {nome_conj} (N = {r['N']})")
    for nome in ["clima_e_ndvi", "so_clima", "so_ndvi"]:
        x = r[nome]
        print(f"  {nome:13s} R² resíduo {x['R2_do_residuo_fora_da_amostra']:+.3f}  "
              f"MAE linear {x['MAE_so_linear']:.2f} -> linear+RF {x['MAE_linear_mais_RF']:.2f}")
    print("  3 maiores no resíduo (MDI):", list(r["MDI_no_residuo_pct"].items())[:3])
