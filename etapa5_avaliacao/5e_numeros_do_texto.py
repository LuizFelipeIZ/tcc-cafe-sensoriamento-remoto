"""
ETAPA 5e: números citados no texto que não estão em outras saídas
=================================================================

1. Hiperparâmetros escolhidos no Cenário 4 ajustado com as 350 observações
   (Seção 2.5 do TCC).
2. Resumo do clima na média dos municípios (Seção 3.3 do TCC).

    python etapa5_avaliacao/5e_numeros_do_texto.py   (~10 s)

Saída: resultados/5e_numeros_do_texto.json
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ALVO, DADOS_BRUTOS, VARIAVEIS, busca_em_grade, carregar_base, salvar_json  # noqa: E402

base = carregar_base()
busca = busca_em_grade().fit(base[VARIAVEIS], base[ALVO])
saida = {"hiperparametros_modelo_final": busca.best_params_}
# 2. clima na média dos municípios: chuva e mínima por ano civil da fase; déficit por safra
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
