"""
ETAPA 5c: onde e quanto o modelo erra (diagnósticos do K-Fold)
=============================================================

Usa as estimativas fora da amostra salvas pela etapa 4a
(resultados/predicoes_kfold.csv) e responde:

1. Viés por faixa de rendimento: o Random Forest "comprime" os extremos?
   (a média das árvores reduz a variância, mas puxa as estimativas para o
   centro: superestima safras muito baixas e subestima as muito altas)
2. Produtividade média e erro por município, com o ganho sobre a persistência
   (Figura 7 do TCC).
3. Observado x estimado, lado a lado com a baseline (figura de dispersão).
4. Por que tirar 1991-2002: numa regressão linear do rendimento na tendência
   e na anomalia bienal, o sinal da anomalia muda. Numa cultura bienal ele
   deveria ser NEGATIVO (safra alta puxa a seguinte para baixo); com os
   valores repetidos do IBGE ele fica positivo, porque a repetição apaga a
   alternância.

Como rodar
----------
    python etapa5_avaliacao/5c_erros_e_diagnosticos.py
(precisa de geopandas para o mapa; sem ele, o mapa é pulado)

Saídas: resultados/5c_diagnosticos.json, resultados/erro_por_municipio.csv,
        resultados/figuras/figura_obs_vs_estimado.png,
        resultados/figuras/figura7_mapa_municipios.png
"""
import sys
import unicodedata
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, FIGURAS, KG_POR_SACA, LIMITES, MUN, RESULTADOS,  # noqa: E402
                    SAFRAS_REPETIDAS, carregar_base, salvar_json)

plt.rcParams.update({"font.family": "serif", "font.size": 9})
ESCURO, CINZA, VERMELHO = "#0B3D2A", "#9A9A9A", "#C0392B"

caminho = RESULTADOS / "predicoes_kfold.csv"
if not caminho.exists():
    sys.exit("Rode antes: python etapa4_modelagem/4a_random_forest_kfold_aninhado.py")
P = pd.read_csv(caminho)
obs = P[ALVO] / KG_POR_SACA
rf = P.pred_kfold / KG_POR_SACA
bl = P.media_movel_3anos / KG_POR_SACA
saida = {}

# 1. Viés por faixa ------------------------------------------------------------------
faixas = {}
for lo, hi in [(0, 12), (12, 20), (20, 30), (30, 35), (35, 60)]:
    k = (obs >= lo) & (obs < hi)
    faixas[f"{lo}-{hi} sacas/ha"] = {"n": int(k.sum()), "vies_medio": round(float((rf[k] - obs[k]).mean()), 2)}
saida["vies_por_faixa"] = faixas

# 2. Erro por município ----------------------------------------------------------------
P["erro_rf"] = np.abs(obs - rf)
P["erro_baseline"] = np.abs(obs - bl)
mun = P.groupby(MUN)[["erro_rf", "erro_baseline"]].mean()
mun["reducao_pct"] = (1 - mun.erro_rf / mun.erro_baseline) * 100
mun = mun.round(2).sort_values("erro_rf", ascending=False)
mun.to_csv(RESULTADOS / "erro_por_municipio.csv")
saida["municipios_em_que_o_RF_errou_menos"] = int((mun.reducao_pct > 0).sum())
saida["erro_por_municipio"] = mun.to_dict("index")

# 3. Observado x estimado ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.6), sharex=True, sharey=True)
lim = (5, 48)
for a, est, titulo, cor in [(ax[0], bl, "Persistência histórica M(t)", CINZA),
                            (ax[1], rf, "Random Forest (clima e NDVI)", ESCURO)]:
    a.scatter(obs, est, s=9, alpha=0.6, color=cor, edgecolor="none")
    a.plot(lim, lim, color=VERMELHO, lw=1, ls="--")
    texto = f"MAE = {np.mean(np.abs(obs - est)):.2f}\nRMSE = {np.sqrt(np.mean((obs - est) ** 2)):.2f}"
    a.text(0.04, 0.96, texto.replace(".", ","), transform=a.transAxes, va="top", fontsize=8.5)
    a.set_title(titulo, fontsize=9.5, fontweight="bold")
    a.set_xlabel("Rendimento observado (sacas/ha)")
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_aspect("equal")
    a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("Rendimento estimado (sacas/ha)")
plt.tight_layout()
plt.savefig(FIGURAS / "figura_obs_vs_estimado.png", dpi=300)
plt.close()

# 4. Sinal da anomalia bienal com e sem 1991-2002 ------------------------------------------
base = carregar_base()
controles = ["tendencia_temporal", "anomalia_bienal_t_1"]
coef = {}
for nome, d in [("serie_completa", base), ("sem_1991_2002", base[~base[ANO].isin(SAFRAS_REPETIDAS)])]:
    lr = LinearRegression().fit(d[controles], d[ALVO])
    coef[nome] = round(float(lr.coef_[1]), 2)
saida["coeficiente_linear_da_anomalia_bienal"] = coef

# Mapa do erro por município (opcional: precisa de geopandas) ------------------------------
try:
    import geopandas as gpd

    def sem_acento(s):
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

    g = gpd.read_file(LIMITES / "municipios_recorte_ssp.gpkg").to_crs(31983)
    g["chave"] = g.NM_MUN.map(sem_acento)
    m = mun.copy()
    m.index = m.index.map(sem_acento)
    G = g[g.chave.isin(m.index)].join(m, on="chave")
    F = g[~g.chave.isin(m.index)]
    from matplotlib.ticker import FuncFormatter
    virgula = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "\u2212"))
    # deslocamento dos rótulos (m) onde os nomes se sobrepõem às divisas
    desloca = {"Guaranesia": (-1500, 3500), "Guaxupe": (0, -3000), "Arceburgo": (800, -2500),
               "Juruaia": (0, 1500), "Nova Resende": (1500, 0)}
    # (a) produtividade média real; (b) MAE do Random Forest (Cenário 4), com a redução frente à persistência
    prod = obs.groupby(P[MUN]).mean()
    prod.index = prod.index.map(sem_acento)
    G["produtividade"] = G.chave.map(prod)
    saida["produtividade_media_por_municipio"] = prod.round(1).to_dict()
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for a, col, titulo, cmap in [(axs[0], "produtividade", "(a) Produtividade média (sacas/ha)", "YlGn"),
                                 (axs[1], "erro_rf", "(b) MAE do Random Forest (sacas/ha)", "YlOrRd")]:
        F.plot(ax=a, color="#F2F2F2", edgecolor="#BBBBBB", lw=0.4)
        G.plot(ax=a, column=col, cmap=cmap, edgecolor="k", lw=0.5, legend=True, legend_kwds={"shrink": 0.6})
        for _, r in G.iterrows():
            p = r.geometry.representative_point()
            dx, dy = desloca.get(r.chave, (0, 0))
            rot = f"{r.NM_MUN}\n{r[col]:.1f}"
            if col == "erro_rf":   # entre parênteses: redução do erro frente à persistência histórica
                rot += f" ({r.reducao_pct:.0f}%)"
            a.annotate(rot.replace(".", ",").replace("-", "\u2212"), (p.x + dx, p.y + dy),
                       ha="center", va="center", fontsize=6.6)
        x0, y0, x1, y1 = G.total_bounds
        a.set_xlim(x0 - 4000, x1 + 4000)
        a.set_ylim(y0 - 4000, y1 + 4000)
        a.set_title(titulo, fontsize=9.5, fontweight="bold")
        a.set_axis_off()
        fig.axes[-1].yaxis.set_major_formatter(virgula)
    plt.tight_layout()
    plt.savefig(FIGURAS / "figura7_mapa_municipios.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close()
except ImportError:
    print("[aviso] geopandas não instalado: mapa por município não gerado")
salvar_json(saida, "5c_diagnosticos.json")

print("Viés por faixa:", faixas)
print(f"\nErro por município (RF errou menos em {saida['municipios_em_que_o_RF_errou_menos']} de 10):")
print(mun.to_string())
print("\nCoeficiente linear da anomalia bienal:", coef)
