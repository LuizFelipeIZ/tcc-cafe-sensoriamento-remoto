"""
ETAPA 5e: números citados no texto que não estão em outras saídas
==================================================================

1. Efeito da chuva no inverno (SHAP) por faixa de chuva: < 30 mm, 30 a 60 mm
   e > 90 mm, e as safras com inverno abaixo de 30 mm.
2. Efeito da temperatura mínima na florada (SHAP): safra em que ele se
   concentra e a mínima dessa safra.
3. Hiperparâmetros escolhidos no modelo ajustado com as 350 observações.
4. Sentido do efeito (SHAP) de cada variável: correlação de Spearman entre o valor da
   variável e o seu SHAP (positiva: valores altos elevam a estimativa).
5. Resumo do clima na média dos municípios (primeiro parágrafo dos Resultados).

    python etapa5_avaliacao/5e_numeros_do_texto.py   (~10 s)

Saída: resultados/5e_numeros_do_texto.json
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, DADOS_BRUTOS, KG_POR_SACA, VARIAVEIS, busca_em_grade, carregar_base,  # noqa: E402
                    salvar_json)

base = carregar_base()
busca = busca_em_grade().fit(base[VARIAVEIS], base[ALVO])
sv = pd.DataFrame(shap.TreeExplainer(busca.best_estimator_).shap_values(base[VARIAVEIS]) / KG_POR_SACA,
                  columns=VARIAVEIS)

chuva = base["Precip_Inverno_Risco_Geada_t1"]
s_chuva = sv["Precip_Inverno_Risco_Geada_t1"]
faixas = {"abaixo_de_30": chuva < 30, "30_a_60": chuva.between(30, 60), "acima_de_90": chuva > 90}
inverno = {k: round(float(s_chuva[m].mean()), 2) for k, m in faixas.items()}
inverno["mediana_mm"] = round(float(chuva.median()), 0)
inverno["safras_abaixo_de_30"] = sorted(int(a) for a in base.loc[chuva < 30, ANO].unique())

s_tf = sv["Tmin_Abs_Florada_t1"]
por_safra = s_tf.groupby(base[ANO]).mean()
safra_tf = int(por_safra.idxmin())
saida = {
    "chuva_inverno_SHAP": inverno,
    "tmin_florada": {"safra_de_maior_efeito_negativo": safra_tf,
                     "efeito_medio_nessa_safra": round(float(por_safra.min()), 2),
                     "minima_media_nessa_safra_C": round(float(base.loc[base[ANO] == safra_tf,
                                                                        "Tmin_Abs_Florada_t1"].mean()), 1),
                     "fracao_do_efeito_negativo_nessa_safra": round(float(
                         s_tf[(base[ANO] == safra_tf) & (s_tf < 0)].sum() / s_tf[s_tf < 0].sum()), 2)},
    "hiperparametros_modelo_final": busca.best_params_,
    "sentido_SHAP_spearman": {v: round(float(base[v].corr(sv[v], method="spearman")), 2) for v in VARIAVEIS},
}
# 5. clima na média dos municípios: chuva e mínima por ano civil da fase; déficit por safra
cl = pd.read_csv(DADOS_BRUTOS / "clima_era5land_janelas_1988_2023.csv")
cl = cl[cl.Ano.between(1989, 2023)].groupby("Ano").mean(numeric_only=True)
bh = pd.read_csv(DADOS_BRUTOS / "balanco_hidrico_openmeteo_1988_2023.csv")
bh = bh[bh.Ano.between(1989, 2023)].groupby("Ano").mean(numeric_only=True)
saida["clima_resumo"] = {
    "chuva_florada_media_mm": round(float(cl.Precip_Florada.mean())),
    "chuva_florada_2020_mm": round(float(cl.loc[2020, "Precip_Florada"])),
    "tmin_inverno_media_C": round(float(cl.Tmin_Abs_Inverno_Risco_Geada.mean()), 1),
    "tmin_inverno_1994_C": round(float(cl.loc[1994, "Tmin_Abs_Inverno_Risco_Geada"]), 1),
    "tmin_inverno_2021_C": round(float(cl.loc[2021, "Tmin_Abs_Inverno_Risco_Geada"]), 1),
    "def_florada_media_mm": round(float(bh.DEF_Florada.mean())),
    "def_florada_2014_mm": round(float(bh.loc[2015, "DEF_Florada"])),    # florada de 2014 = safra 2015
    "def_florada_2020_mm": round(float(bh.loc[2021, "DEF_Florada"])),    # florada de 2020 = safra 2021
    "def_expansao_media_mm": round(float(bh.DEF_Expansao_Graos.mean())),
    "def_expansao_2014_mm": round(float(bh.loc[2014, "DEF_Expansao_Graos"])),
}
salvar_json(saida, "5e_numeros_do_texto.json")
print(saida)
