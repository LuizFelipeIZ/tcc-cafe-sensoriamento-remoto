"""
ETAPA 5d: figuras de apoio do texto (fluxograma, área de estudo e séries)
========================================================================

Gera as figuras que não dependem do modelo:
  * Figura 1: fluxograma do Material e Métodos (dados, máscara, processamento,
    base, cenários e avaliação)
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
# Figura 1: fluxograma, na ordem do Material e Métodos (seção de cada etapa)
# ---------------------------------------------------------------------------
def figura1_fluxograma():
    """Seis linhas: dados, máscara de café, processamento, base, cenários e avaliação."""
    ESCURO, CINZA, AZUL, VERDE, BRANCO = "#1F4E3D", "#EEEEEE", "#E3EBF3", "#DCEEE8", "#FFFFFF"
    FS, FS_ROT = 9, 9.5
    W_FIG, H_FIG = 7.6, 4.9
    fig, ax = plt.subplots(figsize=(W_FIG, H_FIG))
    ax.set_xlim(0, W_FIG)
    ax.set_ylim(0, H_FIG)
    ax.axis("off")

    X0, X1 = 1.25, W_FIG - 0.02          # área das caixas
    GAP = 0.12
    COL = (X1 - X0 - 2 * GAP) / 3        # largura de uma coluna (3 colunas)
    H = 0.60                             # altura das caixas
    Y = [4.28, 3.50, 2.62, 1.86, 0.98, 0.08]   # base de cada linha
    HS = [H, 0.52, 0.66, 0.46, 0.72, 0.70]


    def xcol(i):
        return X0 + i * (COL + GAP)


    def caixa(x, y, w, h, texto, cor, negrito=False, fs=FS):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=cor, ec=ESCURO, lw=0.9))
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs, linespacing=1.12,
                fontweight="bold" if negrito else "normal")


    def seta(x, y0, y1):
        ax.annotate("", (x, y1), (x, y0), arrowprops=dict(arrowstyle="-|>", color=ESCURO, lw=0.9, mutation_scale=8))


    rotulos = ["Dados\n(2.2)", "Máscara de\ncafé (2.4)", "Processa-\nmento (2.4)", "Base de\ndados", "Cenários\n(2.5)",
               "Avaliação\n(2.6)"]
    for y, h, r in zip(Y, HS, rotulos):
        ax.text(0.02, y + h / 2, r, va="center", fontsize=FS_ROT, fontweight="bold", color=ESCURO, linespacing=1.05)

    # Dados
    dados = ["IBGE/SIDRA, Tabela 1613\nprodutividade (kg/ha)\n1988-2023",
             "ERA5-Land (≈ 11 km):\nchuva e Tmín diárias;\nOpen-Meteo (sede)",
             "Landsat 5, 7 e 8\nreflectância de superfície\n(30 m)"]
    for i, t in enumerate(dados):
        caixa(xcol(i), Y[0], COL, HS[0], t, CINZA)
    # Máscara (colunas 2 e 3)
    caixa(xcol(1), Y[1], 2 * COL + GAP, HS[1],
          "MapBiomas, Coleção 9 (30 m): pixels da classe café\nNDVI: máscara do ano da safra  ·  clima: união de cinco anos",
          AZUL)
    # Processamento
    proc = ["kg/ha → sacas/ha; valores\nrepetidos; tendência T(t)\ne anomalia bienal A(t)",
            "chuva e Tmín por fase:\ninverno, florada e enchimento;\nbalanço hídrico",
            "compósitos P90, P10 e\nNDVI na florada; ajuste\nentre sensores"]
    for i, t in enumerate(proc):
        caixa(xcol(i), Y[2], COL, HS[2], t, BRANCO)
    # Variáveis e modelo (largura total)
    caixa(xcol(0), Y[3], X1 - X0, HS[3],
          "N = 350 (10 municípios × 35 safras): produtividade (sacas/ha) e variáveis\n"
          "independentes: 2 controles, 6 de clima, 4 de NDVI e as dos previsores (Quadro 1)", VERDE)
    # Cenários (duas caixas)
    W2 = (X1 - X0 - GAP) / 2
    cen = [("Cenários 1 a 4: validação cruzada aninhada", "1 Base  ·  2 Base + clima\n3 Base + NDVI  ·  4 Completo"),
           ("Cenários 5 e 6: janela expansiva", "5 Previsor de seca\n6 Previsor de geada")]
    for i, (tit, t) in enumerate(cen):
        x = X0 + i * (W2 + GAP)
        caixa(x, Y[4], W2, HS[4], "", VERDE)
        ax.text(x + W2 / 2, Y[4] + HS[4] - 0.13, tit, ha="center", va="center", fontsize=FS, fontweight="bold", color=ESCURO)
        ax.text(x + W2 / 2, Y[4] + 0.25, t, ha="center", va="center", fontsize=FS - 0.5, linespacing=1.15)
    # Avaliação (3 caixas)
    aval = [("Desempenho", "MAE, RMSE e R²;\n× persistência M(t) e Y(t−2);\nWilcoxon e Holm"),
            ("Importância", "MDI por variável\ne por fase do clima"),
            ("Safras de evento", "produtividade real ×\nestimada após seca\ne geada")]
    for i, (tit, t) in enumerate(aval):
        x = xcol(i)
        caixa(x, Y[5], COL, HS[5], "", CINZA)
        ax.text(x + COL / 2, Y[5] + HS[5] - 0.11, tit, ha="center", va="center", fontsize=FS, fontweight="bold",
                color=ESCURO)
        ax.text(x + COL / 2, Y[5] + (HS[5] - 0.2) / 2, t, ha="center", va="center", fontsize=FS - 0.5, linespacing=1.1)

    # Setas verticais, sem cruzamentos
    for i in (1, 2):
        seta(xcol(i) + COL / 2, Y[0], Y[1] + HS[1] + 0.01)
        seta(xcol(i) + COL / 2, Y[1], Y[2] + HS[2] + 0.01)
    seta(xcol(0) + COL / 2, Y[0], Y[2] + HS[2] + 0.01)
    for i in range(3):
        seta(xcol(i) + COL / 2, Y[2], Y[3] + HS[3] + 0.01)
    for i in range(2):
        seta(X0 + i * (W2 + GAP) + W2 / 2, Y[3], Y[4] + HS[4] + 0.01)
    for i in range(3):
        seta(xcol(i) + COL / 2, Y[4], Y[5] + HS[5] + 0.01)
    plt.savefig(FIGURAS / "figura1_fluxograma.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close()


with plt.rc_context({"font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"]}):
    figura1_fluxograma()


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
