"""
ETAPA 5g: produtividade real e estimada ao longo da série, por cenário
======================================================================

Uma figura por cenário (Seção 3 do TCC), com quatro municípios: Guaxupé,
Muzambinho, Nova Resende e Cabo Verde. Cada painel mostra a produtividade real
do IBGE, a estimada pelo cenário e a persistência histórica M(t), com as safras
seguintes às geadas de 1994 e 2021 e à seca de 2014 sombreadas. No canto, o
erro médio absoluto (MAE) do cenário no município: na série toda e nas safras
de evento.

Cenários 1 a 4: estimativas da validação cruzada aninhada (etapa 5f).
Cenários 5 e 6: estimativas da janela expansiva (etapa 4b), só nas safras de
teste; o MAE é calculado nessas safras.

    python etapa5_avaliacao/5g_series_por_cenario.py   (~10 s)

Saídas: resultados/figuras/figura3_cenario1.png ... figura6_cenario4.png,
        figura9_cenario5.png e figura10_cenario6.png (Figuras 3 a 6, 9 e 10 do TCC)
        resultados/5g_mae_municipios.json
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ANO, FIGURAS, MUN, RESULTADOS, salvar_json  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})

MUNICIPIOS = [("Guaxupe", "Guaxupé"), ("Muzambinho", "Muzambinho"), ("Nova Resende", "Nova Resende"),
              ("Cabo Verde", "Cabo Verde")]
EVENTOS = [((1995, 1996), "geada 1994"), ((2014, 2015), "seca 2014"), ((2022, 2023), "geada 2021")]
SAFRAS_EVENTO = [a for (anos, _) in EVENTOS for a in anos]
NOMES = {"C1": "Cenário 1: base", "C2": "Cenário 2: base + clima", "C3": "Cenário 3: base + NDVI",
         "C4": "Cenário 4: completo", "C5": "Cenário 5: previsor de seca", "C6": "Cenário 6: previsor de geada"}

ARQUIVO = {"C1": "figura3_cenario1.png", "C2": "figura4_cenario2.png", "C3": "figura5_cenario3.png",
           "C4": "figura6_cenario4.png", "C5": "figura9_cenario5.png", "C6": "figura10_cenario6.png"}  # numeração do TCC

# cores: tinta escura para o dado real, azul para a estimativa, cinza para a referência
REAL, ESTIMADA, PERSIST, FAIXA, TEXTO = "#1f1e1c", "#2a78d6", "#8d8b85", "#ece8df", "#52514e"

P = pd.read_csv(RESULTADOS / "cenarios_predicoes.csv")
prev = {"C5": pd.read_csv(RESULTADOS / "previsor_seca.csv"), "C6": pd.read_csv(RESULTADOS / "previsor_geada.csv")}


def estimativas(c):
    """Tabela Município, Ano, real, estimada, persistência do cenário c."""
    if c in prev:
        t = prev[c]
        return t[[MUN, ANO, "real"]].assign(estimada=t.rf, persistencia=t.baseline_M)
    return P[[MUN, ANO, "real"]].assign(estimada=P[c], persistencia=P.persistencia_M)


def trechos(anos):
    """Divide anos em sequências contínuas, para não ligar safras de teste distantes."""
    grupos, atual = [], [anos[0]]
    for a in anos[1:]:
        if a == atual[-1] + 1:
            atual.append(a)
        else:
            grupos.append(atual)
            atual = [a]
    return grupos + [atual]


saida = {}
for c, titulo in NOMES.items():
    E = estimativas(c)
    previsor = c in prev
    fig, axs = plt.subplots(1, 4, figsize=(6.3, 1.78), sharey=True)
    saida[c] = {}
    for k, (ax, (mun, rotulo)) in enumerate(zip(axs, MUNICIPIOS)):
        serie = P[P[MUN] == mun].sort_values(ANO)
        e = E[E[MUN] == mun].sort_values(ANO)
        for (a0, a1), nome in EVENTOS:
            ax.axvspan(a0 - 0.5, a1 + 0.5, color=FAIXA, lw=0, zorder=0)
        ax.plot(serie[ANO], serie.persistencia_M, color=PERSIST, lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.plot(serie[ANO], serie.real, color=REAL, lw=1.0, marker="o", ms=2.2, zorder=3)
        if previsor:
            for g in trechos(sorted(e[ANO])):
                q = e[e[ANO].isin(g)]
                ax.plot(q[ANO], q.estimada, color=ESTIMADA, lw=1.4, marker="o", ms=3.4, zorder=4,
                        markeredgecolor="white", markeredgewidth=0.5)
        else:
            ax.plot(e[ANO], e.estimada, color=ESTIMADA, lw=1.4, zorder=4)
        erro = (e.estimada - e.real).abs()
        mae = float(erro.mean())
        mae_pers = float((e.persistencia - e.real).abs().mean())
        if previsor:
            nota = f"MAE {mae:.2f}  ·  persistência {mae_pers:.2f}".replace(".", ",")
            saida[c][rotulo] = {"MAE": round(mae, 2), "MAE_persistencia": round(mae_pers, 2), "N": int(len(e))}
        else:
            mae_ev = float(erro[e[ANO].isin(SAFRAS_EVENTO)].mean())
            nota = f"MAE {mae:.2f}  ·  nos eventos {mae_ev:.2f}".replace(".", ",")
            saida[c][rotulo] = {"MAE": round(mae, 2), "MAE_eventos": round(mae_ev, 2),
                                "MAE_persistencia": round(mae_pers, 2)}
        ax.text(0.5, 1.02, nota, transform=ax.transAxes, fontsize=6.5, va="bottom", ha="center", color=TEXTO)
        ax.set_title(rotulo, fontsize=8, fontweight="bold", pad=11)
        ax.set_xlim(1988, 2024)
        ax.set_ylim(0, 51)
        ax.set_xticks([1990, 2000, 2010, 2020])
        ax.set_yticks([0, 10, 20, 30, 40, 50])
        ax.tick_params(length=2, pad=1.5, labelsize=7)
        ax.grid(axis="y", color="#e4e2dc", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("Produtividade (sacas/ha)", fontsize=7.5)
    rotulo_est = "Estimada pelo cenário" + (" (safras de teste)" if previsor else "")
    legenda = [Line2D([], [], color=REAL, lw=1.0, marker="o", ms=2.2, label="Real (IBGE)"),
               Line2D([], [], color=ESTIMADA, lw=1.4, marker="o" if previsor else None, ms=3.4, label=rotulo_est),
               Line2D([], [], color=PERSIST, lw=0.8, ls=(0, (3, 2)), label="Persistência histórica M(t)"),
               Patch(color=FAIXA, label="Safras após geada ou seca")]
    fig.legend(handles=legenda, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01),
               handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.5)
    fig.savefig(FIGURAS / ARQUIVO[c], dpi=300)
    plt.close(fig)
    print(c, {k: v["MAE"] for k, v in saida[c].items()})

salvar_json(saida, "5g_mae_municipios.json")
