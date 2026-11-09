"""
ETAPA 5a: atribuição do dano climático por variável e por fase (Figuras 3 e 4)
==============================================================================

Pergunta
--------
Quais variáveis o Random Forest mais usa, e em que fase do ciclo do café
está o peso do clima?

Duas medidas de importância
---------------------------
* MDI (redução média da variância): quanto cada variável reduz o erro dentro
  das árvores, somado em todas as divisões. É a importância nativa do RF.
* SHAP (Lundberg e Lee, 2017): quanto cada variável empurra cada estimativa
  para cima ou para baixo, em sacas/ha. Mostra também a DIREÇÃO do efeito.
As duas são calculadas no modelo ajustado com as 350 observações (os
hiperparâmetros saem da mesma busca em grade). A concordância entre elas é
medida pelo ρ de Spearman.

Por fase
--------
As importâncias das variáveis climáticas são somadas em três fases:
florada (t-1), inverno (t-1) e enchimento. Para ver se a divisão é estável,
ela é recalculada nos 5 modelos do K-Fold da etapa 4a.

Leitura correta: esses pesos dizem quanto o modelo USOU cada variável no
ajuste, e não relações de causa nem quanto cada uma reduz o erro.

Como rodar
----------
    python etapa5_avaliacao/5a_importancia_mdi_shap.py   (~30 s)

Saídas: resultados/5a_importancia.json,
        resultados/figuras/figura3_importancia.png,
        resultados/figuras/figura4_shap.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, CLIMA, CONTROLES, FASES, FIGURAS, KG_POR_SACA, NDVI, NOMES, VARIAVEIS,  # noqa: E402
                    busca_em_grade, carregar_base, kfold_externo, salvar_json)

ESCURO, CLARO, AZUL = "#0B3D2A", "#13CBA0", "#2E86AB"
plt.rcParams.update({"font.family": "serif", "font.size": 9})

base = carregar_base()
X, y = base[VARIAVEIS], base[ALVO]

# Modelo final, ajustado com todas as observações ---------------------------------
modelo = busca_em_grade().fit(X, y).best_estimator_
mdi = pd.Series(modelo.feature_importances_ * 100, index=VARIAVEIS)
valores_shap = shap.TreeExplainer(modelo).shap_values(X)                 # kg/ha
shap_pct = pd.Series(np.abs(valores_shap).mean(0), index=VARIAVEIS)
shap_pct = shap_pct / shap_pct.sum() * 100
rho = float(mdi.rank().corr(shap_pct.rank(), method="spearman"))

grupos = {"Controles": CONTROLES, **FASES, "NDVI": NDVI}
por_grupo = {g: {"MDI": round(float(mdi[v].sum()), 1), "SHAP": round(float(shap_pct[v].sum()), 1)}
             for g, v in grupos.items()}
# Parcela de cada fase dentro do clima. O TCC soma as importâncias já
# arredondadas a uma casa (9,2 + 3,8 = 13,0 de 36,6), por isso informa 48/36/16.
mdi_1casa = mdi.round(1)
clima_total = sum(mdi_1casa[v].sum() for v in FASES.values())
fase_no_clima = {f: round(float(mdi_1casa[v].sum() / clima_total * 100), 1) for f, v in FASES.items()}

# Estabilidade da divisão por fase nos 5 modelos do K-Fold -----------------------
# Esta checagem foi feita no TCC com a tendência antes da anomalia bienal na
# ordem das colunas. A ordem só muda o desempate interno das árvores; foi
# mantida para reproduzir exatamente as faixas do texto (45-52%, 28-34%, 13-21%).
ordem_checagem = ["tendencia_temporal", "anomalia_bienal_t_1"] + CLIMA + NDVI
estabilidade = []
for treino, _ in kfold_externo().split(X):
    Xc = base[ordem_checagem]
    m = busca_em_grade().fit(Xc.iloc[treino], y.iloc[treino]).best_estimator_
    imp = pd.Series(m.feature_importances_, index=ordem_checagem)
    tot = sum(imp[v].sum() for v in FASES.values())
    estabilidade.append({f: round(float(imp[v].sum() / tot * 100)) for f, v in FASES.items()})

saida = {
    "hiperparametros": {k: modelo.get_params()[k] for k in ["n_estimators", "max_depth", "min_samples_split"]},
    "MDI_pct": mdi.round(1).sort_values(ascending=False).to_dict(),
    "SHAP_pct": shap_pct.round(1).sort_values(ascending=False).to_dict(),
    "spearman_MDI_SHAP": round(rho, 2),
    "por_grupo": por_grupo,
    "fases_dentro_do_clima_pct": fase_no_clima,
    "estabilidade_fases_5_partes": estabilidade,
}
salvar_json(saida, "5a_importancia.json")

# Figura 3: (a) MDI por variável, (b) MDI e SHAP por fase e grupo ----------------
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
ordem = mdi.sort_values()
cores = [ESCURO if v in CONTROLES else (AZUL if v in NDVI else CLARO) for v in ordem.index]
a.barh([NOMES[v] for v in ordem.index], ordem.values, height=0.7, color=cores)
a.set_xlabel("Importância MDI (%)")
a.set_title("(a) Por variável", fontsize=9.5, fontweight="bold", loc="left")
a.spines[["top", "right"]].set_visible(False)
a.legend(handles=[Patch(color=ESCURO, label="Controles"), Patch(color=CLARO, label="Clima"),
                  Patch(color=AZUL, label="NDVI")], frameon=False, fontsize=7.5, loc="lower right")

rotulos = ["Controles", "Florada (t-1)", "NDVI", "Inverno (t-1)", "Enchimento"][::-1]
pos, h = np.arange(len(rotulos)), 0.36
v_mdi = [por_grupo[g]["MDI"] for g in rotulos]
v_shap = [por_grupo[g]["SHAP"] for g in rotulos]
b.barh(pos + h / 2, v_mdi, height=h, color=ESCURO, label="MDI")
b.barh(pos - h / 2, v_shap, height=h, color="white", edgecolor=ESCURO, hatch="////", lw=0.8, label="SHAP")
for yy, v in list(zip(pos + h / 2, v_mdi)) + list(zip(pos - h / 2, v_shap)):
    b.text(v + 0.8, yy, f"{v:.1f}".replace(".", ","), va="center", fontsize=7.5)
b.set_yticks(pos, rotulos)
b.set_xlim(0, 62)
b.set_xlabel("Participação (%)")
b.set_title("(b) Por fase fenológica e grupo", fontsize=9.5, fontweight="bold", loc="left")
b.spines[["top", "right"]].set_visible(False)
b.legend(frameon=False, fontsize=7.5, loc="lower right")
plt.tight_layout()
plt.savefig(FIGURAS / "figura3_importancia.png", dpi=300)
plt.close()

# Figura 4: valores SHAP de cada observação ---------------------------------------
shap.plots._labels.labels["FEATURE_VALUE_LOW"] = "Baixo"    # legenda em português
shap.plots._labels.labels["FEATURE_VALUE_HIGH"] = "Alto"
plt.figure()
shap.summary_plot(valores_shap / KG_POR_SACA, X.rename(columns=NOMES), show=False,
                  plot_size=(7, 4.8), color_bar_label="Valor da variável")
plt.xlabel("Efeito sobre o rendimento estimado (sacas/ha)")
plt.tight_layout()
plt.savefig(FIGURAS / "figura4_shap.png", dpi=300)
plt.close()

print("MDI (%):", saida["MDI_pct"])
print("Por grupo:", por_grupo)
print("Fases dentro do clima (%):", fase_no_clima)
print("Estabilidade nas 5 partes:", estabilidade)
print(f"Spearman MDI x SHAP: {rho:.2f}")
