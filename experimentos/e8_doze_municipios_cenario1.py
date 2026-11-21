"""
EXPERIMENTO 8: Cenário 1 com 12 municípios (reserva para a banca)
=================================================================
Acrescenta São Sebastião do Paraíso e Monte Santo de Minas, que têm a série
completa na PAM (Tabela 1613 do SIDRA, consultada em 21/11/2026), aos dez do
TCC e repete o Cenário 1 (tendência e anomalia bienal), que só usa o IBGE.
Mesmas regras do TCC: café total até 2012 e arábica de 2013 a 2023, safras
1989 a 2023, K-Fold aninhado e safra inteira fora, Wilcoxon por safra.

Saída: resultados/e8_doze_municipios_cenario1.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CONTROLES, DADOS_BRUTOS, MUN, RESULTADOS, busca_em_grade,  # noqa: E402
                    metricas, validacao_aninhada, wilcoxon_obs_e_safra)

pam = pd.concat([pd.read_csv(DADOS_BRUTOS / "producao_ibge_pam_1988_2023.csv"),
                 pd.read_csv(DADOS_BRUTOS / "producao_ibge_pam_ssp_montesanto_1988_2023.csv")])
pam = pam.rename(columns={"Rendimento_kg_ha": ALVO}).sort_values([MUN, ANO]) if ALVO not in pam else pam
g = pam.groupby(MUN)[ALVO]
pam["produtividade_t_1"] = g.shift(1)
pam["media_movel_3anos"] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
pam["anomalia_bienal_t_1"] = pam.produtividade_t_1 - pam.media_movel_3anos
pam["tendencia_temporal"] = pam[ANO] - 1988
pam["repete"] = g.diff().eq(0)


def safra_fora(d):
    p = np.zeros(len(d))
    for tr, te in LeaveOneGroupOut().split(d, groups=d[ANO]):
        p[te] = busca_em_grade().fit(d[CONTROLES].iloc[tr], d[ALVO].iloc[tr]).predict(d[CONTROLES].iloc[te])
    return p


res = {}
for nome, muns in [("10_municipios", None), ("12_municipios", "todos"), ("so_os_2_novos", "novos")]:
    d = pam[pam[ANO] >= 1989]
    if muns is None:
        d = d[~d[MUN].isin(["Monte Santo de Minas", "Sao Sebastiao do Paraiso"])]
    d = d.reset_index(drop=True)
    y = d[ALVO].values
    eM = np.abs(y - d.media_movel_3anos.values)
    pk = validacao_aninhada(d, CONTROLES)[0]
    ps = safra_fora(d)
    r = {"N": len(d), "repetidos": int(d.repete.sum()), "media_real_sc": round(y.mean() / 60, 2),
         "persistencia": metricas(y, d.media_movel_3anos),
         "C1_kfold": {**metricas(y, pk), "wilcoxon_vs_M": wilcoxon_obs_e_safra(eM, np.abs(y - pk), d[ANO])},
         "C1_safra_fora": {**metricas(y, ps), "wilcoxon_vs_M": wilcoxon_obs_e_safra(eM, np.abs(y - ps), d[ANO])}}
    if muns == "todos":
        for m in ["Monte Santo de Minas", "Sao Sebastiao do Paraiso"]:
            s = (d[MUN] == m).values
            r[m] = {"MAE_C1_kfold": metricas(y[s], pk[s])["MAE"], "MAE_persistencia": metricas(y[s], d.media_movel_3anos[s])["MAE"]}
    res[nome] = r
    print(nome, json.dumps(r, ensure_ascii=False, default=float)[:600], flush=True)
    if muns == "todos":
        break
(RESULTADOS / "e8_doze_municipios_cenario1.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
