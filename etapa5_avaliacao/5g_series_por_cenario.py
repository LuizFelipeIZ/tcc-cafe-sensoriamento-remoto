"""
ETAPA 5g: produtividade real e estimada ao longo da série, por cenário
======================================================================

Uma figura por cenário (Seção 3 do TCC), com quatro municípios: Guaxupé,
Muzambinho, Nova Resende e Cabo Verde, um painel cada. Cada painel mostra a
produtividade real do IBGE, a estimada pelo cenário e a persistência histórica
M(t), com as safras seguintes às geadas de 1994 e 2021 e à seca de 2014
sombreadas. A faixa entre a real e a estimada é o erro: laranja quando o
cenário superestima, azul quando subestima. Acima do painel, o erro médio
absoluto (MAE) do cenário no município: na série toda e nas safras de evento.

Cenários 1 a 4: estimativas da validação cruzada aninhada (etapa 5f).
Cenários 5 e 6: estimativas da janela expansiva (etapa 4b), só nas safras de
teste; o MAE é calculado nessas safras.

    python etapa5_avaliacao/5g_series_por_cenario.py   (~10 s)

Saídas: resultados/figuras/figura3_cenario1.png ... figura6_cenario4.png,
        figura9_cenario5.png e figura10_cenario6.png (Figuras 3 a 6, 9 e 10 do TCC,
        1 x 4 painéis) e resultados/figuras/apresentacao/slide_figura*.png (2 x 2
        painéis, para os slides)
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

# cores: tinta escura para o dado real, azul para a estimativa, cinza para a referência;
# a faixa entre real e estimada mostra o erro: laranja quando o cenário superestima, azul quando subestima
REAL, ESTIMADA, PERSIST, FAIXA, TEXTO = "#1f1e1c", "#2a78d6", "#8d8b85", "#efece5", "#5f5e5a"
SOBRE, SOB = "#f7c3ad", "#b7d3f6"
SOBRE_LINHA, SOB_LINHA = "#eb6834", "#2a78d6"

P = pd.read_csv(RESULTADOS / "cenarios_predicoes.csv")
prev = {"C5": pd.read_csv(RESULTADOS / "previsor_seca.csv"), "C6": pd.read_csv(RESULTADOS / "previsor_geada.csv")}


def estimativas(c):
    """Tabela Município, Ano, real, estimada, persistência do cenário c."""
    if c in prev:
        t = prev[c]
        return t[[MUN, ANO, "real"]].assign(estimada=t.rf, persistencia=t.baseline_M)
    return P[[MUN, ANO, "real"]].assign(estimada=P[c], persistencia=P.persistencia_M)


def desenhar(c, formato):
    """formato "texto": 1 x 4 painéis, para a largura da página do TCC;
    formato "slide": 2 x 2 painéis maiores, com os eventos rotulados, para a apresentação."""
    E = estimativas(c)
    previsor = c in prev
    texto = formato == "texto"
    if texto:
        fig, axs = plt.subplots(1, 4, figsize=(6.3, 2.05), sharey=True)
    else:
        fig, axs = plt.subplots(2, 2, figsize=(6.3, 3.6), sharex=True, sharey=True)
    maes = {}
    for ax, (mun, rotulo) in zip(axs.flat, MUNICIPIOS):
        serie = P[P[MUN] == mun].sort_values(ANO)
        e = E[E[MUN] == mun].sort_values(ANO)
        for (a0, a1), nome in EVENTOS:
            ax.axvspan(a0 - 0.5, a1 + 0.5, color=FAIXA, lw=0, zorder=0)
            if not texto:
                ax.text(a1 + 0.5 if a1 == 2023 else (a0 + a1) / 2, 53.2, nome, ha="right" if a1 == 2023 else "center",
                        va="center", fontsize=6.3, color=TEXTO, style="italic", zorder=1)
        ax.plot(serie[ANO], serie.persistencia_M, color=PERSIST, lw=0.7 if texto else 0.8, ls=(0, (3, 2)), zorder=2)
        if previsor:
            for _, r in e.iterrows():
                cor = SOBRE_LINHA if r.estimada > r.real else SOB_LINHA
                ax.plot([r[ANO], r[ANO]], [r.real, r.estimada], color=cor, lw=1.8 if texto else 2.4, alpha=0.55,
                        solid_capstyle="butt", zorder=3)
        else:
            x, yr, ye = e[ANO].to_numpy(), e.real.to_numpy(), e.estimada.to_numpy()
            ax.fill_between(x, yr, ye, where=ye >= yr, interpolate=True, color=SOBRE, lw=0, zorder=1.5)
            ax.fill_between(x, yr, ye, where=ye < yr, interpolate=True, color=SOB, lw=0, zorder=1.5)
        ax.plot(serie[ANO], serie.real, color=REAL, lw=0.9 if texto else 1.1, marker="o", ms=1.9 if texto else 2.6,
                zorder=4)
        if previsor:
            ax.plot(e[ANO], e.estimada, ls="none", marker="o", ms=3.6 if texto else 5, color=ESTIMADA,
                    markeredgecolor="white", markeredgewidth=0.5 if texto else 0.7, zorder=5)
        else:
            ax.plot(e[ANO], e.estimada, color=ESTIMADA, lw=1.3 if texto else 1.6, zorder=5)
        erro = (e.estimada - e.real).abs()
        mae = float(erro.mean())
        mae_pers = float((e.persistencia - e.real).abs().mean())
        if previsor:
            nota = f"MAE {mae:.2f}  ·  persistência {mae_pers:.2f}".replace(".", ",")
            maes[rotulo] = {"MAE": round(mae, 2), "MAE_persistencia": round(mae_pers, 2), "N": int(len(e))}
        else:
            mae_ev = float(erro[e[ANO].isin(SAFRAS_EVENTO)].mean())
            nota = f"MAE {mae:.2f}  ·  após eventos {mae_ev:.2f}".replace(".", ",")
            maes[rotulo] = {"MAE": round(mae, 2), "MAE_eventos": round(mae_ev, 2),
                            "MAE_persistencia": round(mae_pers, 2)}
        if texto:
            ax.set_title(rotulo, fontsize=8, fontweight="bold", pad=11)
            ax.text(0.5, 1.02, nota, transform=ax.transAxes, fontsize=6.3, va="bottom", ha="center", color=TEXTO)
            ax.set_ylim(0, 51)
            ax.set_xticks([1990, 2000, 2010, 2020])
        else:
            ax.set_title(rotulo, loc="left", fontsize=8.5, fontweight="bold", pad=4)
            ax.set_title(nota, loc="right", fontsize=7, color=TEXTO, pad=4)
            ax.set_ylim(0, 55)
            ax.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020])
        ax.set_xlim(1988, 2024)
        ax.set_yticks([0, 10, 20, 30, 40, 50])
        ax.tick_params(length=2, pad=1.5, labelsize=6.5 if texto else 7)
        ax.grid(axis="y", color="#e6e4de", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in (axs[:1] if texto else axs[:, 0]):
        ax.set_ylabel("Produtividade (sacas/ha)", fontsize=7 if texto else 7.5)
    rotulo_est = "Estimada" + (" (safras de teste)" if previsor else "")
    legenda = [Line2D([], [], color=REAL, lw=1.1, marker="o", ms=2.6, label="Real (IBGE)"),
               Line2D([], [], color=ESTIMADA, lw=0 if previsor else 1.6, marker="o" if previsor else None, ms=5,
                      markeredgecolor="white", label=rotulo_est),
               Line2D([], [], color=PERSIST, lw=0.8, ls=(0, (3, 2)), label="Persistência M(t)"),
               Patch(color=SOBRE, label="Superestimou"), Patch(color=SOB, label="Subestimou"),
               Patch(color=FAIXA, label="Safras após geada ou seca")]
    fig.legend(handles=legenda, loc="lower center", ncol=6, frameon=False, fontsize=6.5 if texto else 7,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
    if texto:
        fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.4)
        fig.savefig(FIGURAS / ARQUIVO[c], dpi=300)
    else:
        fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=0.7, w_pad=0.8)
        fig.savefig(FIGURAS / "apresentacao" / ARQUIVO[c].replace("figura", "slide_figura"), dpi=300)
    plt.close(fig)
    return maes


(FIGURAS / "apresentacao").mkdir(exist_ok=True)
saida = {}
for c in NOMES:
    saida[c] = desenhar(c, "texto")
    assert desenhar(c, "slide") == saida[c]
    print(c, {k: v["MAE"] for k, v in saida[c].items()})

salvar_json(saida, "5g_mae_municipios.json")
