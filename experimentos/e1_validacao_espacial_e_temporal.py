"""
EXPERIMENTO 1: o modelo generaliza para municípios e safras que não viu?
=======================================================================

O K-Fold da etapa 4a sorteia observações; outros municípios da mesma safra
ficam no treino. Aqui o mesmo modelo de 12 variáveis é testado de duas
formas mais duras:

* GroupKFold por MUNICÍPIO: em cada parte, 2 municípios inteiros saem do
  treino (nenhum ano deles é visto). Mede a generalização no espaço.
* Holdout TEMPORAL: treina com 1989-2013 e estima 2014-2023 de uma vez.
  Mede a previsão de um bloco de safras futuras.

Resultado (MAE em sacas/ha): por município, o modelo completo erra 4,52,
perto do K-Fold (4,80), então ele generaliza bem para um município que não
viu. No bloco 2014-2023, erra 4,36 contra 4,85 da baseline, mas com R²
próximo de zero: acerta o nível médio do período, não a variação entre
safras. Esses testes não entraram como resultado principal do TCC; a
previsão de safras futuras foi avaliada pela janela expansiva (etapa 4b).

    python experimentos/e1_validacao_espacial_e_temporal.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CLIMA, CONTROLES, MUN, NDVI, busca_em_grade, carregar_base,  # noqa: E402
                    metricas, salvar_json)

base = carregar_base()
y = base[ALVO]
saida = {"baseline_M_serie": metricas(y, base.media_movel_3anos)}
for nome, V in [("completo", CONTROLES + CLIMA + NDVI), ("sem_ndvi", CONTROLES + CLIMA), ("controles", CONTROLES)]:
    p = np.zeros(len(y))
    for treino, teste in GroupKFold(5).split(base, y, base[MUN]):
        p[teste] = busca_em_grade().fit(base.iloc[treino][V], y.iloc[treino]).predict(base.iloc[teste][V])
    saida[f"groupkfold_municipio_{nome}"] = metricas(y, p)

    tr, te = base[ANO] < 2014, base[ANO] >= 2014
    p = busca_em_grade().fit(base.loc[tr, V], y[tr]).predict(base.loc[te, V])
    saida[f"holdout_2014_2023_{nome}"] = metricas(y[te], p)
saida["holdout_2014_2023_baseline_M"] = metricas(y[base[ANO] >= 2014], base.media_movel_3anos[base[ANO] >= 2014])

salvar_json(saida, "e1_validacao_espacial_e_temporal.json")
for k, v in saida.items():
    print(f"{k:38s} MAE {v['MAE']:.2f}  RMSE {v['RMSE']:.2f}  R² {v['R2']:.2f}")
