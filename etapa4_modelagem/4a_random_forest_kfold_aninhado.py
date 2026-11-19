"""
ETAPA 4a: Random Forest com validação cruzada aninhada (base dos cenários 1 a 4)
================================================================================

Pergunta
--------
Com clima e NDVI, o Random Forest estima o rendimento municipal com erro
menor que a persistência histórica? E de onde vem esse ganho?

Como o modelo é avaliado
------------------------
Validação cruzada ANINHADA:
  * ciclo externo: as 350 observações são sorteadas em 5 partes; 4 treinam e
    1 é estimada, até cada parte ter sido estimada uma vez;
  * ciclo interno: dentro das 4 partes de treino, uma busca em grade em 3
    partes escolhe os hiperparâmetros (100 ou 150 árvores, profundidade 3 ou
    5, 5 ou 10 amostras por divisão).
Assim, toda estimativa é "fora da amostra". Atenção: como outros municípios
da mesma safra ficam no treino, este K-Fold mede a capacidade de COMPLETAR a
série (estimar um município numa safra cujos vizinhos já se conhece), e não
de prever safras futuras. A previsão do futuro é testada na etapa 4b.

Comparações
-----------
  * baselines: persistência histórica M(t) e o rendimento de dois anos antes
    Y(t-2), na mesma fase do ciclo bienal;
  * modelos reduzidos: sem NDVI, sem clima e só com os controles;
  * teste de Wilcoxon pareado nos erros absolutos, por observação e por safra
    (média dos 10 municípios, porque eles compartilham o clima), com ajuste
    de Holm nas comparações dos modelos reduzidos;
  * tudo repetido sem as safras de 1991 a 2002 (valores repetidos do IBGE);
  * o modelo completo validado também com UMA SAFRA INTEIRA fora do treino
    (LeaveOneGroupOut por safra), sem vizinhos da mesma safra.

Como rodar
----------
    python etapa4_modelagem/4a_random_forest_kfold_aninhado.py   (~2 min)

Saídas em resultados/: 4a_kfold_aninhado.json, 4a_desempenho.csv,
4a_modelos_reduzidos.csv, predicoes_kfold.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CLIMA, CONTROLES, MUN, NDVI, RESULTADOS, SAFRAS_REPETIDAS,  # noqa: E402
                    busca_em_grade, carregar_base, holm, metricas, salvar_json, validacao_aninhada,
                    wilcoxon_obs_e_safra)


def validacao_safra_fora(dados, variaveis):
    """Ciclo externo com UMA SAFRA INTEIRA fora do treino (os 10 municípios dela).

    É o teste mais duro do K-Fold: o modelo não vê nenhum vizinho da safra que
    estima. A busca em grade (ciclo interno) roda só com as outras safras.
    """
    X, y = dados[variaveis], dados[ALVO]
    pred = np.zeros(len(y))
    for treino, teste in LeaveOneGroupOut().split(X, y, dados[ANO]):
        pred[teste] = busca_em_grade().fit(X.iloc[treino], y.iloc[treino]).predict(X.iloc[teste])
    return pred

MODELOS = {
    "completo": CONTROLES + CLIMA + NDVI,
    "sem_ndvi": CONTROLES + CLIMA,
    "sem_clima": CONTROLES + NDVI,
    "controles": CONTROLES,
}

base = carregar_base()
conjuntos = {
    "serie_completa": base,
    "sem_1991_2002": base[~base[ANO].isin(SAFRAS_REPETIDAS)].reset_index(drop=True),
}

saida = {}
for nome_conj, d in conjuntos.items():
    y = d[ALVO]
    r = {"N": len(d), "baseline_M": metricas(y, d.media_movel_3anos)}
    pred = {}
    for nome, variaveis in MODELOS.items():
        p, dp = validacao_aninhada(d, variaveis)
        pred[nome] = p
        r[nome] = {**metricas(y, p), "dp_MAE_entre_partes": round(dp, 2)}
        print(f"[{nome_conj}] {nome:10s} MAE {r[nome]['MAE']:.2f} sacas/ha")

    erro_rf = np.abs(y - pred["completo"])
    # RF x persistência histórica
    r["wilcoxon_RF_vs_M"] = wilcoxon_obs_e_safra(np.abs(y - d.media_movel_3anos), erro_rf, d[ANO])
    # Safra inteira fora do treino (sugestão do orientador): sem vizinhos da mesma safra
    p_sf = validacao_safra_fora(d, MODELOS["completo"])
    r["completo_safra_fora"] = metricas(y, p_sf)
    r["wilcoxon_safra_fora_vs_M"] = wilcoxon_obs_e_safra(np.abs(y - d.media_movel_3anos), np.abs(y - p_sf), d[ANO])
    print(f"[{nome_conj}] safra fora MAE {r['completo_safra_fora']['MAE']:.2f} sacas/ha")
    # Modelos reduzidos x completo, com Holm sobre os 6 p-valores
    brutos = []
    for nome in ["sem_ndvi", "sem_clima", "controles"]:
        r[f"wilcoxon_{nome}_vs_completo"] = wilcoxon_obs_e_safra(np.abs(y - pred[nome]), erro_rf, d[ANO])
        brutos += r[f"wilcoxon_{nome}_vs_completo"][:2]
    r["holm_obs_safra"] = [round(float(v), 4) for v in holm(brutos)]

    # Baseline Y(t-2): só onde existe a safra t-2
    ok = d.y_t2.notna().values
    dd = d[ok]
    r["baseline_Y_t2"] = {
        "N": int(ok.sum()),
        "Y_t2": metricas(dd[ALVO], dd.y_t2),
        "M": metricas(dd[ALVO], dd.media_movel_3anos),
        "RF": metricas(dd[ALVO], pred["completo"][ok]),
        "wilcoxon_RF_vs_Y_t2": wilcoxon_obs_e_safra(np.abs(dd[ALVO] - dd.y_t2).values, erro_rf[ok].values, dd[ANO]),
    }
    if nome_conj == "serie_completa":
        pd.DataFrame({MUN: d[MUN], ANO: d[ANO], ALVO: y, "media_movel_3anos": d.media_movel_3anos,
                      "pred_kfold": pred["completo"]}).to_csv(RESULTADOS / "predicoes_kfold.csv", index=False)
    saida[nome_conj] = r

# Correlação de Spearman NDVI x clima (a falta de ganho do NDVI não é redundância)
rho = base[NDVI + CLIMA].corr(method="spearman").loc[NDVI, CLIMA]
saida["spearman_max_abs_ndvi_clima"] = round(float(rho.abs().max().max()), 2)
salvar_json(saida, "4a_kfold_aninhado.json")

# Tabelas resumidas (no TCC, as tabelas finais saem da etapa 5f) -----------------------
t1 = []
for nome_conj, r in saida.items():
    if not isinstance(r, dict) or "N" not in r:
        continue
    t1.append([nome_conj, "Persistência histórica M(t)", r["baseline_M"]["MAE"], r["baseline_M"]["RMSE"],
               r["baseline_M"]["R2"], "", ""])
    t1.append([nome_conj, "Rendimento de dois anos antes Y(t-2)", r["baseline_Y_t2"]["Y_t2"]["MAE"],
               r["baseline_Y_t2"]["Y_t2"]["RMSE"], r["baseline_Y_t2"]["Y_t2"]["R2"], "", ""])
    c = r["completo"]
    t1.append([nome_conj, "Random Forest (clima + NDVI)", c["MAE"], c["RMSE"], c["R2"],
               c["dp_MAE_entre_partes"], f"p obs={r['wilcoxon_RF_vs_M'][0]}, p safra={r['wilcoxon_RF_vs_M'][1]}"])
    sf = r["completo_safra_fora"]
    t1.append([nome_conj, "Random Forest, safra inteira fora", sf["MAE"], sf["RMSE"], sf["R2"], "",
               f"p obs={r['wilcoxon_safra_fora_vs_M'][0]}, p safra={r['wilcoxon_safra_fora_vs_M'][1]}"])
pd.DataFrame(t1, columns=["conjunto", "modelo", "MAE", "RMSE", "R2", "dp_MAE", "Wilcoxon_vs_M"]).to_csv(
    RESULTADOS / "4a_desempenho.csv", index=False)

t2 = []
for nome_conj in conjuntos:
    r = saida[nome_conj]
    t2.append([nome_conj, "completo", r["completo"]["MAE"], "", ""])
    for i, nome in enumerate(["sem_ndvi", "sem_clima", "controles"]):
        w = r[f"wilcoxon_{nome}_vs_completo"]
        hb = r["holm_obs_safra"][2 * i:2 * i + 2]
        t2.append([nome_conj, nome, r[nome]["MAE"], f"{w[0]} [{hb[0]}]", f"{w[1]} [{hb[1]}]"])
pd.DataFrame(t2, columns=["conjunto", "modelo", "MAE", "p_obs [Holm]", "p_safra [Holm]"]).to_csv(
    RESULTADOS / "4a_modelos_reduzidos.csv", index=False)

# Resumo ---------------------------------------------------------------------------
for nome_conj in conjuntos:
    r = saida[nome_conj]
    ganho = (1 - r["completo"]["MAE"] / r["baseline_M"]["MAE"]) * 100
    print(f"\n== {nome_conj} (N = {r['N']})")
    print(f"  baseline M(t): MAE {r['baseline_M']['MAE']:.2f} | RF: MAE {r['completo']['MAE']:.2f} "
          f"RMSE {r['completo']['RMSE']:.2f} R² {r['completo']['R2']:.2f} | redução {ganho:.0f}%")
    w = r["wilcoxon_RF_vs_M"]
    print(f"  Wilcoxon RF x M: p obs {w[0]}, p safra {w[1]} (RF melhor em {w[2]} de {w[3]} safras)")
    print(f"  RF x Y(t-2): p safra {r['baseline_Y_t2']['wilcoxon_RF_vs_Y_t2'][1]}")
    for nome in ["sem_ndvi", "sem_clima", "controles"]:
        print(f"  {nome:10s} MAE {r[nome]['MAE']:.2f}  p {r[f'wilcoxon_{nome}_vs_completo'][:2]}")
print(f"\nSpearman |NDVI x clima| máximo: {saida['spearman_max_abs_ndvi_clima']}")
