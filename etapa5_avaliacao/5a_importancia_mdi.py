"""
ETAPA 5a: importância das variáveis (MDI) por variável e por fase (Figura 5)
==========================================================================

Pergunta
--------
Quais variáveis o Random Forest mais usa, e em que fase do ciclo do café
está o peso do clima?

Medida de importância
---------------------
MDI (redução média da variância): a cada divisão de uma árvore, mede-se
quanto a variável reduziu a variância da produtividade nos nós; a importância
é a média dessa redução em todas as árvores, em % do total. É a importância
nativa do Random Forest, calculada no modelo ajustado com as 350 observações
(os hiperparâmetros saem da mesma busca em grade). A análise SHAP, usada em
versões anteriores, foi retirada do TCC (decisão com o orientador).

Por fase
--------
As importâncias das variáveis climáticas são somadas em três fases:
florada (t-1), inverno (t-1) e enchimento. Para ver se a divisão é estável,
ela é recalculada nos 5 modelos do K-Fold da etapa 4a.

Leitura correta: esses pesos dizem quanto o modelo USOU cada variável no
ajuste, e não relações de causa nem quanto cada uma reduz o erro.

Como rodar
----------
    python etapa5_avaliacao/5a_importancia_mdi.py   (~30 s)

Saídas: resultados/5a_importancia.json,
        resultados/figuras/figura5_importancia_mdi.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, CLIMA, CONTROLES, FASES, FIGURAS, KG_POR_SACA, NDVI, NOMES, VARIAVEIS,  # noqa: E402
                    busca_em_grade, carregar_base, kfold_externo, salvar_json)

ESCURO, CLARO, AZUL = "#0B3D2A", "#13CBA0", "#2E86AB"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8.5})

base = carregar_base()
X, y = base[VARIAVEIS], base[ALVO]

# Modelo final, ajustado com todas as observações ---------------------------------
modelo = busca_em_grade().fit(X, y).best_estimator_
mdi = pd.Series(modelo.feature_importances_ * 100, index=VARIAVEIS)

grupos = {"Controles": CONTROLES, **FASES, "NDVI": NDVI}
por_grupo = {g: {"MDI": round(float(mdi[v].sum()), 1)} for g, v in grupos.items()}
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
    "por_grupo": por_grupo,
    "fases_dentro_do_clima_pct": fase_no_clima,
    "estabilidade_fases_5_partes": estabilidade,
}
salvar_json(saida, "5a_importancia.json")

# Figura: (a) MDI por variável, (b) MDI por fase e grupo -----------------------
# tamanho pensado para 15 cm de largura na página A4
fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
ordem = mdi.sort_values()
cores = [ESCURO if v in CONTROLES else (AZUL if v in NDVI else CLARO) for v in ordem.index]
a.barh([NOMES[v] for v in ordem.index], ordem.values, height=0.7, color=cores)
a.set_xlabel("Importância MDI (%)")
a.set_title("(a) Por variável", fontsize=9.5, fontweight="bold", loc="left")
a.spines[["top", "right"]].set_visible(False)
a.legend(handles=[Patch(color=ESCURO, label="Controles"), Patch(color=CLARO, label="Clima"),
                  Patch(color=AZUL, label="NDVI")], frameon=False, fontsize=7.5, loc="lower right")

rotulos = ["Controles", "Florada (t-1)", "NDVI", "Inverno (t-1)", "Enchimento"][::-1]
cor_grupo = {"Controles": ESCURO, "NDVI": AZUL}
pos = np.arange(len(rotulos))
v_mdi = [por_grupo[g]["MDI"] for g in rotulos]
b.barh(pos, v_mdi, height=0.6, color=[cor_grupo.get(g, CLARO) for g in rotulos])
for yy, v in zip(pos, v_mdi):
    b.text(v + 0.8, yy, f"{v:.1f}".replace(".", ","), va="center", fontsize=7.5)
b.set_yticks(pos, [r.replace("t-1", "t−1") for r in rotulos])
b.set_xlim(0, 58)
b.set_xlabel("Importância MDI (%)")
b.set_title("(b) Por fase e grupo", fontsize=9.5, fontweight="bold", loc="left")
b.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURAS / "figura5_importancia_mdi.png", dpi=300)
plt.close()

print("MDI (%):", saida["MDI_pct"])
print("Por grupo:", por_grupo)
print("Fases dentro do clima (%):", fase_no_clima)
print("Estabilidade nas 5 partes:", estabilidade)
