"""
ETAPA 5f: os seis cenários do TCC (Tabelas 1 e 3; dispersão real x estimada)
============================================================================

Pergunta
--------
O clima e o NDVI melhoram a estimativa da produtividade feita pelo Random
Forest, além do que a tendência temporal e a bienalidade já explicam? E o
modelo consegue antecipar a safra em anos de seca e de geada?

Cada cenário é um Random Forest com um conjunto de variáveis:

  Cenário 1  Base ............ tendência temporal + anomalia bienal (2 variáveis)
  Cenário 2  Base + clima .... + chuva e mínima no inverno, florada e enchimento (8)
  Cenário 3  Base + NDVI ..... + vigor, estresse, NDVI na florada e amplitude (6)
  Cenário 4  Completo ........ base + clima + NDVI (12)
  Cenário 5  Previsor de seca  (etapa 4b, janela expansiva)
  Cenário 6  Previsor de geada (etapa 4b, janela expansiva)

Os cenários 1 a 4 são avaliados pela validação cruzada aninhada (config.py);
a ordem das colunas é a mesma da etapa 4a, então as estimativas são idênticas
às de lá. Comparações:
  * cada cenário contra a persistência histórica M(t) (Wilcoxon pareado);
  * cenários 2, 3 e 4 contra o cenário 1, que testa a hipótese (Wilcoxon,
    com ajuste de Holm sobre as 6 comparações: 3 cenários x observação/safra).

Saídas em resultados/:
  5f_cenarios.json, tabela1_cenarios.csv, cenarios_predicoes.csv,
  tabela3_safras_de_evento.csv, figuras/real_estimada_dispersao.png

    python etapa5_avaliacao/5f_cenarios.py   (~2 min)
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CLIMA, CONTROLES, FIGURAS, KG_POR_SACA, MUN, NDVI, RESULTADOS,  # noqa: E402
                    SAFRAS_REPETIDAS, carregar_base, holm, metricas, salvar_json, validacao_aninhada,
                    wilcoxon_obs_e_safra)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8})

CENARIOS = {   # mesma ordem de colunas da etapa 4a (reprodutibilidade exata)
    "C1": ("Base", CONTROLES),
    "C2": ("Base + clima", CONTROLES + CLIMA),
    "C3": ("Base + NDVI", CONTROLES + NDVI),
    "C4": ("Completo", CONTROLES + CLIMA + NDVI),
}

base = carregar_base()
conjuntos = {"serie_completa": base, "sem_1991_2002": base[~base[ANO].isin(SAFRAS_REPETIDAS)].copy()}
k4a = json.load(open(RESULTADOS / "4a_kfold_aninhado.json"))

saida, predicoes = {}, None
for nome, d in conjuntos.items():
    d = d.reset_index(drop=True)
    y = d[ALVO]
    erro_M = np.abs(y - d.media_movel_3anos)
    r = {"N": len(d), "persistencia_M": metricas(y, d.media_movel_3anos)}
    pred, erro = {}, {}
    for c, (rotulo, variaveis) in CENARIOS.items():
        p, dp = validacao_aninhada(d, variaveis)
        pred[c], erro[c] = p, np.abs(y - p)
        r[c] = {"rotulo": rotulo, "n_variaveis": len(variaveis), **metricas(y, p),
                "dp_MAE_entre_partes": dp,
                "wilcoxon_vs_M": wilcoxon_obs_e_safra(erro_M, erro[c], d[ANO])}
        print(f"[{nome}] {c} {rotulo}: MAE {r[c]['MAE']:.2f}")
    # Hipótese: os cenários 2, 3 e 4 erram menos que o cenário 1?
    brutos = []
    for c in ["C2", "C3", "C4"]:
        w = wilcoxon_obs_e_safra(erro["C1"], erro[c], d[ANO])
        r[c]["wilcoxon_vs_C1"] = w
        brutos += w[:2]
    ajust = holm(brutos)
    for i, c in enumerate(["C2", "C3", "C4"]):
        r[c]["holm_vs_C1"] = [round(float(ajust[2 * i]), 4), round(float(ajust[2 * i + 1]), 4)]
    r["C4_safra_inteira_fora"] = k4a[nome]["completo_safra_fora"]
    r["C4_safra_inteira_fora"]["wilcoxon_vs_M"] = k4a[nome]["wilcoxon_safra_fora_vs_M"]
    saida[nome] = r
    if nome == "serie_completa":
        predicoes = pd.DataFrame({MUN: d[MUN], ANO: d[ANO], "real": y / KG_POR_SACA,
                                  "persistencia_M": d.media_movel_3anos / KG_POR_SACA,
                                  **{c: pred[c] / KG_POR_SACA for c in CENARIOS}})

predicoes.round(3).to_csv(RESULTADOS / "cenarios_predicoes.csv", index=False)

# Tabela 1: desempenho dos cenários 1 a 4 e das referências ---------------------------
linhas = []
for nome, r in saida.items():
    linhas.append([nome, "Persistência histórica M(t)", r["persistencia_M"]["MAE"], r["persistencia_M"]["RMSE"],
                   r["persistencia_M"]["R2"], "", ""])
    for c in CENARIOS:
        x = r[c]
        vs1 = f"{x['wilcoxon_vs_C1'][1]} [{x['holm_vs_C1'][1]}]" if c != "C1" else ""
        linhas.append([nome, f"{c} {x['rotulo']}", x["MAE"], x["RMSE"], x["R2"], x["wilcoxon_vs_M"][1], vs1])
    sf = r["C4_safra_inteira_fora"]
    linhas.append([nome, "C4 Completo, safra inteira fora", sf["MAE"], sf["RMSE"], sf["R2"],
                   sf["wilcoxon_vs_M"][1], ""])
pd.DataFrame(linhas, columns=["conjunto", "cenario", "MAE", "RMSE", "R2", "p_safra_vs_M", "p_safra_vs_C1_[Holm]"]
             ).to_csv(RESULTADOS / "tabela1_cenarios.csv", index=False)

# Cenários 5 e 6: previsores da etapa 4b (janela expansiva) ------------------------------
prev = {"C5": pd.read_csv(RESULTADOS / "previsor_seca.csv"), "C6": pd.read_csv(RESULTADOS / "previsor_geada.csv")}
for c, rotulo in [("C5", "Previsor de seca"), ("C6", "Previsor de geada")]:
    t = prev[c]
    saida[c] = {"rotulo": rotulo, "N": len(t), "safras": sorted(int(a) for a in t[ANO].unique()),
                "MAE": round(float(t.erro_rf.mean()), 2), "MAE_persistencia": round(float(t.erro_baseline_M.mean()), 2),
                "RMSE": round(float(np.sqrt(((t.rf - t.real) ** 2).mean())), 2)}

# Tabela 3: produtividade real e estimada nas safras de evento ---------------------------
EVENTOS = [(1995, "Geada 1994, 1ª safra", (1992, 1994)), (1996, "Geada 1994, 2ª safra", (1992, 1994)),
           (2014, "Seca 2014, enchimento", (2011, 2013)), (2015, "Seca 2014, florada", (2011, 2013)),
           (2022, "Geada 2021, 1ª safra", (2019, 2021)), (2023, "Geada 2021, 2ª safra", (2019, 2021))]
P = predicoes
media = P.groupby(ANO).mean(numeric_only=True)
ev = []
for safra, rotulo, (a0, a1) in EVENTOS:
    ref = media.loc[a0:a1, "real"].mean()
    linha = {"Safra": safra, "Evento": rotulo, "Real": media.loc[safra, "real"],
             "Queda": media.loc[safra, "real"] - ref, "Persistência": media.loc[safra, "persistencia_M"],
             **{c: media.loc[safra, c] for c in CENARIOS}}
    for c in ["C5", "C6"]:
        t = prev[c]
        linha[c] = t.loc[t[ANO] == safra, "rf"].mean() if safra in set(t[ANO]) else np.nan
    # erro médio absoluto por município, para saber qual cenário ficou mais perto em cada safra
    q = P[P[ANO] == safra]
    linha["MAE_por_cenario"] = {c: round(float((q[c] - q.real).abs().mean()), 2) for c in CENARIOS}
    ev.append(linha)
tab3 = pd.DataFrame(ev)
tab3.drop(columns="MAE_por_cenario").round(1).to_csv(RESULTADOS / "tabela3_safras_de_evento.csv", index=False)
saida["safras_de_evento"] = json.loads(tab3.round(2).to_json(orient="records", force_ascii=False))

# Quanto da variação de cada variável climática ocorre ENTRE safras (e não entre municípios)?
# Se for quase toda, o clima funciona também como um marcador do ano, papel que a tendência
# temporal pode cumprir quando o clima é retirado (Seção 3.2 do TCC).
saida["clima_variacao_entre_safras_pct"] = {
    v: round(float(100 * base.groupby(ANO)[v].transform("mean").var() / base[v].var()), 1) for v in CLIMA}
salvar_json(saida, "5f_cenarios.json")

# Dispersão real x estimada, um painel por cenário (fora do TCC; ver a etapa 5g) -------------------------
fig, axs = plt.subplots(2, 3, figsize=(6.3, 4.45), sharex=True, sharey=True)
lim = (0, 50)
pers_kfold = float(np.mean(np.abs(P.real - P.persistencia_M)))
paineis = [(c, f"Cenário {c[1]}: {CENARIOS[c][0]}", P.real, P[c], pers_kfold) for c in CENARIOS]
paineis += [("C5", "Cenário 5: Previsor de seca", prev["C5"].real, prev["C5"].rf, prev["C5"].erro_baseline_M.mean()),
            ("C6", "Cenário 6: Previsor de geada", prev["C6"].real, prev["C6"].rf, prev["C6"].erro_baseline_M.mean())]
for a, (c, titulo, real, est, pers) in zip(axs.flat, paineis):
    a.scatter(real, est, s=5, alpha=0.55, color="#0B3D2A" if c in CENARIOS else "#B5651D", edgecolor="none")
    a.plot(lim, lim, color="#C0392B", lw=0.8, ls="--")
    mae = float(np.mean(np.abs(real - est)))
    a.text(0.05, 0.95, f"MAE = {mae:.2f}\nPersistência = {pers:.2f}\nN = {len(real)}".replace(".", ","),
           transform=a.transAxes, va="top", fontsize=7)
    saida.setdefault("R2_por_cenario", {})[c] = round(float(1 - ((real - est) ** 2).sum() /
                                                             ((real - real.mean()) ** 2).sum()), 3)
    a.set_title(titulo, fontsize=8, fontweight="bold")
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_aspect("equal")
    a.set_xticks([0, 10, 20, 30, 40, 50])
    a.set_yticks([0, 10, 20, 30, 40, 50])
    a.spines[["top", "right"]].set_visible(False)
for a in axs[1]:
    a.set_xlabel("Produtividade real (sacas/ha)")
for a in axs[:, 0]:
    a.set_ylabel("Estimada (sacas/ha)")
plt.tight_layout(w_pad=0.6, h_pad=0.8)
plt.savefig(FIGURAS / "real_estimada_dispersao.png", dpi=300)
plt.close()
salvar_json(saida, "5f_cenarios.json")
print(json.dumps({k: (v["MAE"] if isinstance(v, dict) and "MAE" in v else None) for k, v in saida.items()}))
