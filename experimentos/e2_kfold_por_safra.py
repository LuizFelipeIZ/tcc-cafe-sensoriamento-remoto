"""
EXPERIMENTO 2: o que o K-Fold mede de verdade ("completar a série")
==================================================================

No K-Fold sorteado, a safra que está sendo estimada quase sempre tem outros
municípios no treino. Como os 10 municípios sobem e descem juntos, o modelo
pode aprender "como foi aquela safra" pelos vizinhos, via tendência
temporal (que é a própria safra). Este experimento mostra isso:

1. RF só com a tendência temporal, no mesmo K-Fold.
2. Uma regra sem modelo: estimar cada observação pela média dos OUTROS
   municípios da mesma safra que estão no treino.
3. K-Fold agrupado por SAFRA: a safra inteira sai do treino. Aí o modelo
   perde a informação dos vizinhos e o erro sobe.

Resultado: só a tendência erra 4,33 sacas/ha e a simples média dos
vizinhos, 4,34, quase o mesmo que o modelo completo (4,80). Com a safra
inteira fora do treino, o RF passa a errar 6,68, mais que a baseline (6,26).
Por isso o TCC diz que o K-Fold mede a capacidade de completar a série, e
não de prever safras futuras.

    python experimentos/e2_kfold_por_safra.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CLIMA, CONTROLES, KG_POR_SACA, NDVI, busca_em_grade, carregar_base,  # noqa: E402
                    kfold_externo, salvar_json, wilcoxon_obs_e_safra)

base = carregar_base()
y = base[ALVO].values


def validar(V, divisoes):
    p = np.zeros(len(y))
    for tr, te in divisoes:
        p[te] = busca_em_grade().fit(base.iloc[tr][V], y[tr]).predict(base.iloc[te][V])
    return p


saida = {}
p = validar(["tendencia_temporal"], kfold_externo().split(base))
saida["kfold_so_tendencia_MAE"] = round(np.abs(y - p).mean() / KG_POR_SACA, 2)

media_vizinhos = np.full(len(y), np.nan)
for tr, te in kfold_externo().split(base):
    por_safra = base.iloc[tr].groupby(ANO)[ALVO].mean()
    media_vizinhos[te] = base.iloc[te][ANO].map(por_safra).values
ok = ~np.isnan(media_vizinhos)
saida["kfold_media_dos_vizinhos_mesma_safra_MAE"] = round(np.abs(y[ok] - media_vizinhos[ok]).mean() / KG_POR_SACA, 2)

def por_safra():
    return GroupKFold(5).split(base, y, base[ANO])


preds = {n: validar(V, por_safra()) for n, V in
         [("completo", CONTROLES + CLIMA + NDVI), ("controles", CONTROLES), ("sem_ndvi", CONTROLES + CLIMA)]}
erro_base = np.abs(y - base.media_movel_3anos.values)
for n, p in preds.items():
    saida[f"safra_inteira_fora_{n}_MAE"] = round(np.abs(y - p).mean() / KG_POR_SACA, 2)
saida["safra_inteira_fora_baseline_M_MAE"] = round(erro_base.mean() / KG_POR_SACA, 2)
saida["wilcoxon_completo_vs_baseline"] = wilcoxon_obs_e_safra(erro_base, np.abs(y - preds["completo"]), base[ANO])

salvar_json(saida, "e2_kfold_por_safra.json")
for k, v in saida.items():
    print(f"{k:45s} {v}")
