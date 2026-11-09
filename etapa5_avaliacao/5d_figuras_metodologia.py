"""
ETAPA 5d: figuras de apoio do texto (fluxograma, área de estudo e séries)
========================================================================

Gera as figuras que não dependem do modelo:
  * Figura 1: fluxograma das etapas do trabalho
  * Figura 2: localização dos dez municípios estudados
  * série do NDVI harmonizado (média dos 10 municípios, escala do Landsat 8),
    com as safras de evento destacadas
  * série do rendimento médio regional (IBGE), 1988-2023

Como rodar
----------
    python etapa5_avaliacao/5d_figuras_metodologia.py
Saída: resultados/figuras/
"""
import sys
import unicodedata
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ALVO, ANO, DADOS_PROC, FIGURAS, KG_POR_SACA, LIMITES, MUNICIPIOS  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9})
ESCURO, CINZA_CLARO, VERDE_CLARO, AZUL_CLARO = "#0B3D2A", "#EEEEEE", "#DBF3EC", "#DCE9F3"

# ---------------------------------------------------------------------------
# Figura 1: fluxograma
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8.4)
ax.axis("off")
X0, W = 1.75, 1.6


def cx(i):
    return X0 + i * (W + 0.05)


def caixa(c0, c1, y, texto, cor=CINZA_CLARO, negrito=False, h=1.0):
    x = cx(c0)
    w = cx(c1) + W - x
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=cor, ec=ESCURO, lw=1.3))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=9.6,
            fontweight="bold" if negrito else "normal", linespacing=1.25)
    return (x + w / 2, y, y + h)


def seta(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=ESCURO, lw=1.3, mutation_scale=11))


Y = [7.2, 5.65, 4.1, 2.55, 1.0]
for y, rotulo in zip(Y, ["Dados", "Pré-\nprocessamento", "Variáveis\n(N = 350)", "Modelagem", "Avaliação e\ninterpretação"]):
    ax.text(0.05, y + 0.5, rotulo, va="center", fontsize=10.5, fontweight="bold", color=ESCURO)
d = [caixa(i, i, Y[0], t) for i, t in enumerate([
    "IBGE/SIDRA\nTabela 1613\n1988-2023", "ERA5-Land\nTmin e chuva\ndiárias", "Landsat 5, 7 e 8\nreflectância\n(30 m)",
    "MapBiomas\nColeção 9\nclasse 46", "Open-Meteo\nERA5, T e chuva\n(sede)"])]
p0 = caixa(0, 0, Y[1], "Auditoria da série\nkg/ha → sacas/ha\nrepetições 1991-2002")
p1 = caixa(1, 1, Y[1], "Janelas fenológicas\nno ano agrícola\n(área cafeeira)")
p2 = caixa(2, 3, Y[1], "Máscaras de café e de nuvem\ncompósitos por safra (jul t−1 a jun t)\ncalibração entre sensores (RMA)")
p4 = caixa(4, 4, Y[1], "Balanço hídrico\nThornthwaite e\nMather")
for a, b in [(d[0], p0), (d[1], p1), (d[4], p4)]:
    seta(a[0], a[1], b[0], b[2] + 0.02)
seta(d[2][0], d[2][1], d[2][0], p2[2] + 0.02)
seta(d[3][0], d[3][1], d[3][0], p2[2] + 0.02)
seta(d[3][0] - 0.3, d[3][1], p1[0] + 0.4, p1[2] + 0.02)
v0 = caixa(0, 0, Y[2], "Tendência e\nanomalia bienal", VERDE_CLARO)
v1 = caixa(1, 1, Y[2], "Clima: florada e\ninverno (t−1),\nenchimento", VERDE_CLARO)
v2 = caixa(2, 3, Y[2], "NDVI: vigor, estresse,\nflorada e amplitude", AZUL_CLARO)
v4 = caixa(4, 4, Y[2], "Déficit hídrico\n(previsor de seca)", VERDE_CLARO)
for a, b in [(p0, v0), (p1, v1), (p2, v2), (p4, v4)]:
    seta(a[0], a[1], b[0], b[2] + 0.02)
m0 = caixa(0, 2, Y[3], "Random Forest\nbusca em grade + K-Fold aninhado\n× baselines M(t) e Y(t−2)", VERDE_CLARO, True)
m1 = caixa(3, 4, Y[3], "Previsão intra-safra\njanela expansiva (seca e geada)\n× baseline e só controles", VERDE_CLARO, True)
seta(v0[0], v0[1], v0[0], m0[2] + 0.02)
seta(v1[0], v1[1], v1[0], m0[2] + 0.02)
seta(v2[0] - 0.6, v2[1], m0[0] + 0.9, m0[2] + 0.02)
seta(v4[0], v4[1], v4[0], m1[2] + 0.02)
seta(v1[0] + 0.5, v1[1], m1[0] - 0.4, m1[2] + 0.02)
e0 = caixa(0, 0, Y[4], "MAE, RMSE, R²\nWilcoxon por\nobservação e safra")
e1 = caixa(1, 2, Y[4], "Modelos com e sem clima\ne com e sem NDVI (Holm);\nsem 1991-2002")
e2 = caixa(3, 4, Y[4], "Atribuição do dano climático\n(MDI e SHAP por fase);\nerros por município; P10-P90")
seta(m0[0] - 1.2, m0[1], e0[0], e0[2] + 0.02)
seta(m0[0], m0[1], e1[0], e1[2] + 0.02)
seta(m0[0] + 1.2, m0[1], e2[0] - 0.6, e2[2] + 0.02)
seta(m1[0], m1[1], e2[0] + 0.3, e2[2] + 0.02)
plt.tight_layout()
plt.savefig(FIGURAS / "figura1_fluxograma.png", dpi=250, bbox_inches="tight", pad_inches=0.05)
plt.close()

# ---------------------------------------------------------------------------
# Figura 2: área de estudo (precisa de geopandas)
# ---------------------------------------------------------------------------
try:
    import geopandas as gpd

    def sem_acento(s):
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

    g = gpd.read_file(LIMITES / "municipios_recorte_ssp.gpkg").to_crs(31983)
    g["estudado"] = g.NM_MUN.map(sem_acento).isin(MUNICIPIOS)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    g[~g.estudado].plot(ax=ax, color="#EEEEEE", edgecolor="#999999", lw=0.5)
    g[g.estudado].plot(ax=ax, color="#13CBA0", edgecolor=ESCURO, lw=0.8)
    for _, r in g[g.estudado | (g.NM_MUN == "São Sebastião do Paraíso")].iterrows():
        p = r.geometry.representative_point()
        ax.annotate(r.NM_MUN, (p.x, p.y), ha="center", va="center", fontsize=7.5 if r.estudado else 7,
                    color="k" if r.estudado else "#555555")
    ax.legend(handles=[Patch(color="#13CBA0", ec=ESCURO, label="Municípios estudados"),
                       Patch(color="#EEEEEE", ec="#999999", label="Demais municípios do recorte")],
              loc="lower left", bbox_to_anchor=(0, 0.08), fontsize=7.5, frameon=False)
    ax.set_axis_off()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs, ys = x0 + 0.05 * (x1 - x0), y0 + 0.05 * (y1 - y0)
    ax.plot([xs, xs + 20000], [ys, ys], "k-", lw=2)
    ax.text(xs + 10000, ys + 0.015 * (y1 - y0), "20 km", ha="center", fontsize=7.5)
    ax.annotate("N", xy=(x1 - 0.06 * (x1 - x0), y1 - 0.06 * (y1 - y0)),
                xytext=(x1 - 0.06 * (x1 - x0), y1 - 0.16 * (y1 - y0)),
                arrowprops=dict(arrowstyle="-|>", color="k"), ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURAS / "figura2_area_estudo.png", dpi=300)
    plt.close()
except ImportError:
    print("[aviso] geopandas não instalado: Figura 2 não gerada")

# ---------------------------------------------------------------------------
# Séries: NDVI harmonizado e rendimento regional
# ---------------------------------------------------------------------------
H = pd.read_csv(DADOS_PROC / "ndvi_harmonizado_rma.csv").groupby(ANO)[
    ["NDVI_Vigor_Maximo", "NDVI_Estresse_Maximo", "NDVI_Amplitude"]].mean()
fig, a = plt.subplots(figsize=(6.6, 3.3))
for col, cor, rot in [("NDVI_Vigor_Maximo", "#9A9A9A", "Vigor máximo"), ("NDVI_Estresse_Maximo", ESCURO, "Estresse máximo"),
                      ("NDVI_Amplitude", "#2E86AB", "Amplitude")]:
    a.plot(H.index, H[col], "o-", ms=3, color=cor, label=rot)
for x in (1999.5, 2013.5):
    a.axvline(x, color="#BBBBBB", ls=":", lw=1)
for x, t in [(1994, "Landsat 5"), (2006.5, "Landsat 7"), (2018.5, "Landsat 8")]:
    a.text(x, 0.93, t, ha="center", fontsize=7.5, color="#666666")
for ev in (1995, 2001, 2015, 2022):
    a.axvspan(ev - 0.5, ev + 0.5, color="#F2D7D5", zorder=0)
a.set_ylim(0.1, 0.98)
a.set_xlim(1988.5, 2023.5)
a.set_ylabel("NDVI (escala do Landsat 8)")
a.legend(frameon=False, fontsize=7.5, loc="center left", ncol=3, bbox_to_anchor=(0, 0.42))
a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURAS / "serie_ndvi_harmonizado.png", dpi=300)
plt.close()

M = pd.read_csv(DADOS_PROC / "matriz_mestra.csv").groupby(ANO)[ALVO].mean() / KG_POR_SACA
fig, a = plt.subplots(figsize=(6.6, 3.0))
a.plot(M.index, M.values, "o-", ms=3, color=ESCURO)
a.axvspan(1990.5, 2002.5, color="#EEEEEE", zorder=0, label="Valores repetidos no IBGE")
for ev, t in [(1995, "geada 1994"), (2001, "geada 2000"), (2015, "seca 2014"), (2022, "geada 2021")]:
    a.annotate(t, (ev, M[ev]), xytext=(0, -18), textcoords="offset points", ha="center", fontsize=7, color="#C0392B")
a.set_ylabel("Rendimento médio (sacas/ha)")
a.legend(frameon=False, fontsize=7.5, loc="upper left")
a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURAS / "serie_rendimento_regional.png", dpi=300)
plt.close()
print("Figuras salvas em resultados/figuras/")
