"""
EXPERIMENTO 7: um modelo para cada município
============================================

Sugestão do autor: treinar só com as outras safras do município X e estimar
o próprio município X. Assim o treino não mistura municípios, e nenhum
vizinho da mesma safra entra no treino (o efeito do ano não "vaza").

Para cada município (35 safras) e cada cenário (1 a 4):
  * K-Fold aninhado em 5 partes sobre as SAFRAS do município (ciclo interno
    em 3 partes, mesma grade do TCC);
  * deixa uma safra de fora (LOO), para usar 34 safras no treino.
As estimativas dos 10 municípios são juntadas (N = 350) e comparadas com a
persistência M(t) e com o Cenário 1 pelo Wilcoxon por safra, como no TCC.

Saída: resultados/e7_modelo_por_municipio.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CLIMA, CONTROLES, MUN, NDVI, RESULTADOS, SEMENTE,  # noqa: E402
                    busca_em_grade, carregar_base, holm, metricas, wilcoxon_obs_e_safra)

CENARIOS = {"C1": CONTROLES, "C2": CONTROLES + CLIMA, "C3": CONTROLES + NDVI, "C4": CONTROLES + CLIMA + NDVI}
T0 = time.time()
base = carregar_base()
y = base[ALVO].values
erro_M = np.abs(y - base.media_movel_3anos.values)
res = {"N": len(base), "persistencia_M": metricas(y, base.media_movel_3anos)}

for nome_val, divisor in [("kfold5_no_municipio", KFold(5, shuffle=True, random_state=SEMENTE)),
                          ("loo_no_municipio", LeaveOneOut())]:
    rv, erros = {}, {}
    for c, var in CENARIOS.items():
        pred = np.zeros(len(base))
        por_mun = {}
        for mun, g in base.groupby(MUN):
            X, yy = g[var], g[ALVO]
            for tr, te in divisor.split(X):
                pred[g.index[te]] = busca_em_grade().fit(X.iloc[tr], yy.iloc[tr]).predict(X.iloc[te])
            por_mun[mun] = metricas(yy, pred[g.index])["MAE"]
        erros[c] = np.abs(y - pred)
        rv[c] = {**metricas(y, pred), "wilcoxon_vs_M": wilcoxon_obs_e_safra(erro_M, erros[c], base[ANO]),
                 "MAE_por_municipio": por_mun}
        print(f"[{(time.time() - T0) / 60:5.1f} min] {nome_val} {c}: MAE {rv[c]['MAE']:.2f}", flush=True)
    brutos = []
    for c in ["C2", "C3", "C4"]:
        rv[c]["wilcoxon_vs_C1"] = wilcoxon_obs_e_safra(erros["C1"], erros[c], base[ANO])
        brutos.append(rv[c]["wilcoxon_vs_C1"][1])
    for c, p in zip(["C2", "C3", "C4"], holm(brutos)):
        rv[c]["holm_safra_vs_C1"] = round(float(p), 4)
    res[nome_val] = rv
    (RESULTADOS / "e7_modelo_por_municipio.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("fim")
