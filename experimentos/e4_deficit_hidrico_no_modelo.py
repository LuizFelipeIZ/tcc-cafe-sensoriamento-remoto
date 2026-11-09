"""
EXPERIMENTO 4: vale pôr o déficit hídrico no modelo principal?
=============================================================

O balanço hídrico (etapa 1d) entra no previsor de seca. Aqui ele é testado
também no modelo principal (K-Fold aninhado): o modelo completo e o modelo
só com controles, com e sem as três variáveis do balanço (déficit na
florada, déficit na expansão dos grãos e temperatura máxima da expansão).

Resultado: o MAE do modelo completo foi de 4,80 para 4,71 sacas/ha, sem
diferença significativa (p = 0,12 por observação e 0,24 por safra), e o
modelo só com controles piorou (4,54 para 4,63). O balanço ficou fora do
modelo principal porque não traz ganho claro e é calculado num ponto na
sede do município, e não na área de café.

    python experimentos/e4_deficit_hidrico_no_modelo.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CLIMA, CONTROLES, DADOS_BRUTOS, KG_POR_SACA, MUN, NDVI,  # noqa: E402
                    busca_em_grade, carregar_base, salvar_json, validacao_aninhada, wilcoxon_obs_e_safra)

DEF = ["DEF_Florada", "DEF_Expansao_Graos", "Tmax_Expansao_Graos"]
base = carregar_base().merge(pd.read_csv(DADOS_BRUTOS / "balanco_hidrico_openmeteo_1988_2023.csv"),
                             on=[MUN, ANO], how="left")
y = base[ALVO].values
print("valores faltantes do balanço:", int(base[DEF].isna().sum().sum()))

P = {n: validacao_aninhada(base, V)[0] for n, V in [
    ("completo", CONTROLES + CLIMA + NDVI), ("completo_mais_DEF", CONTROLES + CLIMA + NDVI + DEF),
    ("controles", CONTROLES), ("controles_mais_DEF", CONTROLES + DEF)]}
saida = {f"MAE_{n}": round(np.abs(y - p).mean() / KG_POR_SACA, 2) for n, p in P.items()}
saida["wilcoxon_completo_vs_completo_mais_DEF"] = wilcoxon_obs_e_safra(
    np.abs(y - P["completo"]), np.abs(y - P["completo_mais_DEF"]), base[ANO])
saida["wilcoxon_controles_vs_controles_mais_DEF"] = wilcoxon_obs_e_safra(
    np.abs(y - P["controles"]), np.abs(y - P["controles_mais_DEF"]), base[ANO])

V = CONTROLES + CLIMA + NDVI + DEF
m = busca_em_grade().fit(base[V], y).best_estimator_
imp = pd.Series(m.feature_importances_ * 100, index=V)
saida["MDI_das_variaveis_do_balanco_pct"] = imp[DEF].round(1).to_dict()

salvar_json(saida, "e4_deficit_hidrico.json")
for k, v in saida.items():
    print(f"{k:45s} {v}")
