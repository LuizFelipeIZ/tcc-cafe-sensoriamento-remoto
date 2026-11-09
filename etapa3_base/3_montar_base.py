"""
ETAPA 3: montagem da base de modelagem (N = 350 observações município-safra)
===========================================================================

Cada linha é um município numa safra t (1989 a 2023). A safra de 1988 entra
só para calcular as defasagens. Cada observação tem o rendimento e 12
preditoras:

Controles (2)
    anomalia_bienal_t_1  A(t) = Y(t-1) - M(t)
                         M(t) = [Y(t-1) + Y(t-2) + Y(t-3)] / 3, a média das três
                         safras anteriores ("persistência histórica").
                         Diz em que fase do ciclo bienal o cafeeiro está: se a
                         safra passada foi muito acima da média, a planta tende
                         a produzir menos agora. O modelo NUNCA usa Y(t).
    tendencia_temporal   T(t) = t - 1988, o ganho tecnológico ao longo do tempo.

Clima (6) no ANO AGRÍCOLA da safra t
    A florada que forma a safra t acontece em setembro a novembro de t-1, e
    o inverno de risco de geada que a antecede, em junho a agosto de t-1.
    Por isso essas duas janelas são DEFASADAS em um ano (sufixo _t1).
    O enchimento de grãos (janeiro a abril de t) fica no próprio ano.
    Em cada janela: chuva acumulada e temperatura mínima absoluta.

NDVI (4)
    vigor máximo, estresse máximo, NDVI na florada e amplitude, já na escala
    do Landsat 8 (etapa 2b).

Também ficam na base, para as comparações:
    media_movel_3anos  M(t), a baseline de persistência histórica
    y_t2               Y(t-2), a baseline na mesma fase do ciclo bienal

Como rodar
----------
    python etapa3_base/3_montar_base.py
Saída: dados/processados/base_modelagem.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ALVO, ANO, CLIMA, CONTROLES, DADOS_PROC, KG_POR_SACA, MUN, NDVI, salvar_json  # noqa: E402

df = pd.read_csv(DADOS_PROC / "matriz_mestra.csv").sort_values([MUN, ANO]).reset_index(drop=True)
g = df.groupby(MUN)[ALVO]

# Controles -------------------------------------------------------------------
df["produtividade_t_1"] = g.shift(1)
# shift(1) ANTES da média móvel: M(t) usa só safras anteriores (sem vazamento)
df["media_movel_3anos"] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df["anomalia_bienal_t_1"] = df.produtividade_t_1 - df.media_movel_3anos
df["tendencia_temporal"] = df[ANO] - df[ANO].min()
df["y_t2"] = g.shift(2)

# Clima no ano agrícola: florada e inverno do ano anterior ---------------------
for col in ["Precip_Florada", "Tmin_Abs_Florada", "Precip_Inverno_Risco_Geada",
            "Tmin_Abs_Inverno_Risco_Geada", "Dias_Abaixo_3C"]:
    df[col + "_t1"] = df.groupby(MUN)[col].shift(1)

# NDVI harmonizado --------------------------------------------------------------
ndvi = pd.read_csv(DADOS_PROC / "ndvi_harmonizado_rma.csv")
df = df.merge(ndvi[[ANO, MUN] + NDVI], on=[ANO, MUN], how="left")

# Base final: só linhas completas -------------------------------------------------
base = df.dropna(subset=CLIMA + NDVI + CONTROLES + [ALVO, "media_movel_3anos"]).reset_index(drop=True)
base.to_csv(DADOS_PROC / "base_modelagem.csv", index=False)

resumo = {
    "N": len(base),
    "municipios": int(base[MUN].nunique()),
    "safras": [int(base[ANO].min()), int(base[ANO].max())],
    "rendimento_sacas_ha": {
        "media": round(float(base[ALVO].mean()) / KG_POR_SACA, 2),
        "min": round(float(base[ALVO].min()) / KG_POR_SACA, 1),
        "max": round(float(base[ALVO].max()) / KG_POR_SACA, 1),
    },
}
salvar_json(resumo, "3_base.json")
print(f"Base de modelagem: N = {resumo['N']} ({resumo['municipios']} municípios x "
      f"{base[ANO].nunique()} safras, {resumo['safras'][0]}-{resumo['safras'][1]})")
print("Rendimento (sacas/ha):", resumo["rendimento_sacas_ha"])
print("Salvo: dados/processados/base_modelagem.csv")
