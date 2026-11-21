"""
ETAPA 5h: figuras dos slides com as safras de seca e de geada, por cenário
==========================================================================

Para a apresentação. Em cada cenário, a produtividade real do IBGE, a estimada
pelo cenário e a persistência histórica M(t) nas safras pontuais em que houve
seca ou geada, em Nova Resende e Cabo Verde, com os valores numa tabela sob
cada painel, em tamanho de projeção. São os mesmos dados das Figuras 3 a 6, 9 e 10 do
TCC (etapa 5g), recortados nessas safras.

Cenários 1 a 4: safras seguintes à geada de 1994 (1995, 1996), à seca de 2014
(2014, 2015) e à geada de 2021 (2022, 2023). Cenários 5 e 6: as safras de teste
de cada previsor (janela expansiva, etapa 4b).

Também gera o resumo das safras de evento na média dos dez municípios
(Tabela 3 do TCC).

    python etapa5_avaliacao/5h_eventos_slides.py   (~5 s)

Saídas: resultados/figuras/apresentacao/slide_eventos_C1.png ... C6.png
        resultados/figuras/apresentacao/slide_eventos_media.png
"""
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ANO, FIGURAS, MUN, RESULTADOS  # noqa: E402

# paleta dos slides (fundo café escuro)
FUNDO, CREME, BEGE, APAGADO, LINHA = "#3D2B21", "#F4EADB", "#D2C2AE", "#A8957F", "#5C473A"
VERDE, OURO = "#8FD18B", "#E8B86A"
ACIMA, ABAIXO = "#F0905F", "#79AEEA"     # estimativa acima / abaixo da real
plt.rcParams.update({"font.family": "Inter", "font.size": 12, "text.color": CREME})

DOIS = [("Nova Resende", "Nova Resende"), ("Cabo Verde", "Cabo Verde")]   # os dois municípios da apresentação
GRUPOS = {
    "kfold": [("geada 1994", (1995, 1996)), ("seca 2014", (2014, 2015)), ("geada 2021", (2022, 2023))],
    "C5": [("seca 2014 e seguintes", (2014, 2015, 2016, 2017))],
    "C6": [("geada 1994", (1994, 1995)), ("geada 2000", (2000, 2001)), ("geada 2021", (2021, 2022, 2023))],
}
# safras discutidas nos slides, marcadas com um quadro vermelho (None = os dois municípios)
DESTAQUE = {"C1": [(None, 1995)], "C2": [("Nova Resende", 1995)], "C3": [(None, 1995)],
            "C4": [("Nova Resende", 1995)], "C5": [(None, 2014)], "C6": [(None, 1995), ("Nova Resende", 2001)]}
VERMELHO = "#E5484D"
NOME = {"C1": "Cenário 1", "C2": "Cenário 2", "C3": "Cenário 3", "C4": "Cenário 4", "C5": "Cenário 5",
        "C6": "Cenário 6"}

P = pd.read_csv(RESULTADOS / "cenarios_predicoes.csv")
prev = {"C5": pd.read_csv(RESULTADOS / "previsor_seca.csv"), "C6": pd.read_csv(RESULTADOS / "previsor_geada.csv")}
MAE = json.loads((RESULTADOS / "5g_mae_municipios.json").read_text())
SAIDA = FIGURAS / "apresentacao"
SAIDA.mkdir(parents=True, exist_ok=True)

v = lambda x: f"{x:.1f}".replace(".", ",")          # 13.14 -> "13,1"
v2 = lambda x: f"{x:.2f}".replace(".", ",")


def estimativas(c):
    if c in prev:
        t = prev[c]
        return t[[MUN, ANO, "real"]].assign(estimada=t.rf, persistencia=t.baseline_M)
    return P[[MUN, ANO, "real"]].assign(estimada=P[c], persistencia=P.persistencia_M)


def posicoes(grupos, folga=0.7):
    """Posição x de cada safra, com uma folga entre os eventos."""
    x, pos = 0.0, {}
    for _, anos in grupos:
        for a in anos:
            pos[a] = x
            x += 1
        x += folga
    return pos


def figura_cenario(c):
    """Dois municípios, Nova Resende e Cabo Verde, em tamanho de projeção."""
    W, H = 16.64, 5.15                                   # polegadas = px/100 no slide (1664 x 515)
    fig = plt.figure(figsize=(W, H), facecolor=FUNDO)
    grupos = GRUPOS.get(c, GRUPOS["kfold"])
    pos = posicoes(grupos, 0.55)
    cor = OURO if c in prev else VERDE
    E = estimativas(c)
    esq, dir_, folga_p = 1.50, 0.05, 0.60                # margens em polegadas
    larg = (W - esq - dir_ - folga_p) / 2
    base, alt = 1.95, 1.90                               # área do gráfico (polegadas)
    y_lin = [1.30, 0.88, 0.46]                           # linhas da tabela
    for i, (mun, rot) in enumerate(DOIS):
        x0 = esq + i * (larg + folga_p)
        ax = fig.add_axes([x0 / W, base / H, larg / W, alt / H], facecolor=FUNDO)
        e = E[(E[MUN] == mun) & E[ANO].isin(pos)].set_index(ANO).loc[list(pos)]
        xs = [pos[a] for a in e.index]
        for nome, anos in grupos:
            a0, a1 = pos[anos[0]] - 0.42, pos[anos[-1]] + 0.42
            ax.axvspan(a0, a1, color=CREME, alpha=0.05, lw=0, zorder=0)
            ax.text((a0 + a1) / 2, 41.2, nome, ha="center", va="bottom", fontsize=15, style="italic", color=BEGE)
        for x, (_, r) in zip(xs, e.iterrows()):
            ax.plot([x, x], [r.real, r.estimada], color=ACIMA if r.estimada > r.real else ABAIXO, lw=6,
                    alpha=0.9, solid_capstyle="butt", zorder=2)
        ax.plot(xs, e.persistencia, ls="none", marker="o", ms=19, mfc="none", mec=APAGADO, mew=2.6, zorder=3)
        ax.plot(xs, e.real, ls="none", marker="o", ms=19, color=CREME, zorder=4)
        ax.plot(xs, e.estimada, ls="none", marker="o", ms=13, color=cor, mec=FUNDO, mew=1.6, zorder=5)
        ax.set_xlim(-0.6, max(pos.values()) + 0.6)
        ax.set_ylim(5, 40)
        ax.set_yticks([10, 20, 30, 40])
        ax.grid(axis="y", color=LINHA, lw=1, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(a) for a in e.index], fontsize=15, color=BEGE)
        ax.tick_params(axis="both", length=0, pad=8)
        ax.tick_params(axis="y", labelsize=14, labelcolor=APAGADO)
        if i:
            ax.set_yticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        for mun_d, ano_d in DESTAQUE.get(c, []):
            if mun_d in (None, mun) and ano_d in pos:
                xf = fig.transFigure.inverted().transform(ax.transData.transform((pos[ano_d], 0)))[0] * W
                fig.patches.append(FancyBboxPatch(((xf - 0.40) / W, 0.22 / H), 0.80 / W, (base + alt + 0.10 - 0.22) / H,
                                                  boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                                                  fill=False, ec=VERMELHO, lw=3.2, zorder=10))
        m = MAE[c][rot]
        nota = (f"MAE {v2(m['MAE'])} · persistência {v2(m['MAE_persistencia'])}" if c in prev
                else f"MAE {v2(m['MAE'])} na série · {v2(m['MAE_eventos'])} após os eventos")
        fig.text(x0 / W, 4.45 / H, rot, fontsize=21, fontweight="bold", color=CREME, va="center")
        fig.text((x0 + larg) / W, 4.45 / H, nota, fontsize=14.5, color=BEGE, va="center", ha="right")
        tr = blended_transform_factory(ax.transData, fig.transFigure)
        for x, (_, r) in zip(xs, e.iterrows()):
            perto = abs(r.estimada - r.real) < abs(r.persistencia - r.real)
            fig.text(x, y_lin[0] / H, v(r.real), transform=tr, ha="center", va="center", fontsize=17,
                     color=CREME, fontweight="semibold")
            fig.text(x, y_lin[1] / H, v(r.estimada), transform=tr, ha="center", va="center", fontsize=17,
                     color=cor, fontweight="bold" if perto else "normal")
            fig.text(x, y_lin[2] / H, v(r.persistencia), transform=tr, ha="center", va="center", fontsize=17,
                     color=APAGADO)
    # rótulos das linhas da tabela, que servem de legenda
    for y, (txt, estilo) in zip(y_lin, [("Real (IBGE)", dict(color=CREME, mfc=CREME, mec=CREME)),
                                        (NOME[c], dict(color=cor, mfc=cor, mec=cor)),
                                        ("Persistência", dict(color=APAGADO, mfc="none", mec=APAGADO))]):
        cor_txt = estilo.pop("color")
        fig.lines.append(plt.Line2D([0.14 / W], [y / H], transform=fig.transFigure, marker="o", ms=14,
                                    mew=2.2, ls="none", **estilo))
        fig.text(0.32 / W, y / H, txt, fontsize=15.5, color=cor_txt, va="center",
                 fontweight="semibold" if cor_txt != APAGADO else "normal")
    fig.text(0.14 / W, 3.85 / H, "sacas/ha", fontsize=14, color=APAGADO, va="center")
    xl = esq
    for txt, c_ in [("estimativa acima da real", ACIMA), ("estimativa abaixo da real", ABAIXO)]:
        fig.lines.append(plt.Line2D([xl / W, (xl + 0.30) / W], [4.95 / H, 4.95 / H], transform=fig.transFigure,
                                    color=c_, lw=6))
        fig.text((xl + 0.40) / W, 4.95 / H, txt, fontsize=14, color=BEGE, va="center")
        xl += 3.2
    fig.text(xl / W, 4.95 / H, "negrito: mais perto da real que a persistência", fontsize=14, color=BEGE,
             va="center")
    xl += 5.1
    fig.patches.append(FancyBboxPatch((xl / W, 4.85 / H), 0.30 / W, 0.20 / H, boxstyle="round,pad=0,rounding_size=0.004",
                                      transform=fig.transFigure, fill=False, ec=VERMELHO, lw=2.6))
    fig.text((xl + 0.40) / W, 4.95 / H, "safra discutida", fontsize=14, color=BEGE, va="center")
    fig.savefig(SAIDA / f"slide_eventos_{c}.png", dpi=200, facecolor=FUNDO)
    plt.close(fig)


def figura_media():
    """Safras de evento na média dos dez municípios (Tabela 3 do TCC), em forma de gráfico."""
    T = pd.read_csv(RESULTADOS / "tabela3_safras_de_evento.csv")
    W, H = 16.64, 5.30
    fig = plt.figure(figsize=(W, H), facecolor=FUNDO)
    x0, x1 = 6.55, 12.55                                 # área do gráfico (polegadas)
    ax = fig.add_axes([x0 / W, 0.30 / H, (x1 - x0) / W, 4.35 / H], facecolor=FUNDO)
    ys, y = [], 0.0
    for i in range(len(T)):
        if i and i % 2 == 0:
            y += 0.6
        ys.append(-y)
        y += 1
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    cols = [(0.10, "SAFRA", "left", APAGADO), (1.05, "EVENTO", "left", APAGADO), (4.55, "REAL", "right", CREME),
            (5.80, "QUEDA", "right", APAGADO), (13.55, "M(t)", "right", APAGADO),
            (15.15, "CEN. 1 A 4", "right", VERDE), (16.50, "PREVISOR", "right", OURO)]
    for xc, txt, ha, c_ in cols:
        fig.text(xc / W, 4.95 / H, txt, fontsize=11.5, color=c_, ha=ha, va="center", family="DejaVu Sans Mono")
    leg = [("real", dict(marker="o", ms=11, color=CREME)), ("M(t)", dict(marker="o", ms=10, mfc="none", mec=APAGADO, mew=1.8)),
           ("cenários 1 a 4", dict(marker="o", ms=7, color=VERDE)), ("previsor", dict(marker="D", ms=8, color=OURO))]
    xl = x0 + 0.05
    for txt, est in leg:
        fig.lines.append(plt.Line2D([xl / W], [4.95 / H], transform=fig.transFigure, ls="none", **est))
        fig.text((xl + 0.14) / W, 4.95 / H, txt, fontsize=11.5, color=BEGE, va="center")
        xl += 0.30 + 0.085 * len(txt)
    for yy, (_, r) in zip(ys, T.iterrows()):
        ax.axhspan(yy - 0.42, yy + 0.42, color=CREME, alpha=0.04, lw=0, zorder=0)
        cen = [r.C1, r.C2, r.C3, r.C4]
        prv = r.C5 if pd.notna(r.C5) else r.C6
        ax.plot([min(cen), max(cen)], [yy, yy], color=VERDE, lw=9, alpha=0.35, solid_capstyle="round", zorder=1)
        ax.plot(cen, [yy] * 4, ls="none", marker="o", ms=7, color=VERDE, zorder=3)
        ax.plot([r.Persistência], [yy], ls="none", marker="o", ms=12, mfc="none", mec=APAGADO, mew=2, zorder=2)
        if pd.notna(prv):
            ax.plot([prv], [yy], ls="none", marker="D", ms=10, color=OURO, mec=FUNDO, mew=1.2, zorder=4)
        ax.plot([r.Real], [yy], ls="none", marker="o", ms=14, color=CREME, mec=FUNDO, mew=1.5, zorder=5)
        evento, fase = r.Evento.split(", ")
        queda = r.Queda
        txt = [(0.10, str(r.Safra), "left", CREME, "semibold"), (1.05, f"{evento.lower()}, {fase}", "left", BEGE, "normal"),
               (4.55, v(r.Real), "right", CREME, "semibold"),
               (5.80, ("−" if queda < 0 else "+") + v(abs(queda)), "right", OURO if queda < -2 else BEGE, "normal"),
               (13.55, v(r.Persistência), "right", APAGADO, "normal"),
               (15.15, f"{v(min(cen))} a {v(max(cen))}", "right", VERDE, "normal"),
               (16.50, v(prv) if pd.notna(prv) else "–", "right", OURO, "normal")]
        for xc, t, ha, c_, fw in txt:
            fig.text(xc / W, yy, t, transform=tr, fontsize=14, color=c_, ha=ha, va="center", fontweight=fw)
    ax.set_xlim(10, 31)
    ax.set_ylim(min(ys) - 0.6, 0.6)
    ax.set_xticks([10, 15, 20, 25, 30])
    ax.set_yticks([])
    ax.grid(axis="x", color=LINHA, lw=0.8, zorder=0)
    ax.tick_params(axis="x", length=0, labelsize=11.5, labelcolor=APAGADO, pad=4)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(SAIDA / "slide_eventos_media.png", dpi=200, facecolor=FUNDO)
    plt.close(fig)


for c in NOME:
    figura_cenario(c)
figura_media()
print("ok")
