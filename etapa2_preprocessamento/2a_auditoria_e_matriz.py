"""
ETAPA 2a: auditoria da série de produção e montagem da matriz município-ano
===========================================================================

O que faz
---------
1. Confere a série do IBGE e o clima antes de qualquer modelo:
     * cobertura: os 10 municípios têm todas as safras de 1988 a 2023?
     * duplicidade: existe mais de uma linha por município e safra?
     * grafia: os nomes batem entre as bases (sem acento: Guaxupe, Jacui...)?
     * valores faltantes ou não positivos no rendimento
     * consistência: rendimento ~ produção / área colhida
2. Procura safras em que o IBGE REPETE o rendimento do ano anterior. Isso
   acontece em 9 dos 10 municípios, sobretudo de 1991 a 2002, e indica que o
   dado oficial não foi levantado de novo naquela safra. Por isso todas as
   análises são repetidas sem as safras de 1991 a 2002.
3. Junta produção e clima numa única matriz (360 linhas = 10 municípios x
   36 anos) e salva em dados/processados/matriz_mestra.csv.

Como rodar
----------
    python etapa2_preprocessamento/2a_auditoria_e_matriz.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ALVO, ANO, DADOS_BRUTOS, DADOS_PROC, KG_POR_SACA, MUN, MUNICIPIOS, salvar_json  # noqa: E402

prod = pd.read_csv(DADOS_BRUTOS / "producao_ibge_pam_1988_2023.csv")
clima = pd.read_csv(DADOS_BRUTOS / "clima_era5land_janelas_1988_2023.csv")
for tab in (prod, clima):
    tab[MUN] = tab[MUN].str.strip()

relatorio = {}

# 1. Cobertura, duplicidade e grafia ------------------------------------------
anos = range(1988, 2024)
for nome, tab in [("producao", prod), ("clima", clima)]:
    esperado = {(m, a) for m in MUNICIPIOS for a in anos}
    presente = set(zip(tab[MUN], tab[ANO]))
    relatorio[nome] = {
        "linhas": len(tab),
        "faltando": sorted(esperado - presente),
        "sobrando": sorted(presente - esperado),
        "duplicadas": int(tab.duplicated([MUN, ANO]).sum()),
        "nomes_fora_da_lista": sorted(set(tab[MUN]) - set(MUNICIPIOS)),
    }

# 2. Rendimento: faltantes, não positivos e coerência com produção / área ------
r = prod[ALVO]
calc = prod["Producao_t"] * 1000 / prod["Area_Colhida_ha"]
relatorio["rendimento"] = {
    "faltantes": int(r.isna().sum()),
    "nao_positivos": int((r <= 0).sum()),
    "min_sacas_ha": round(float(r.min()) / KG_POR_SACA, 1),
    "max_sacas_ha": round(float(r.max()) / KG_POR_SACA, 1),
    "diferenca_maxima_vs_producao_area_pct": round(float((np.abs(calc - r) / r).max() * 100), 1),
}

# 3. Rendimentos repetidos de uma safra para a seguinte ----------------------
prod = prod.sort_values([MUN, ANO]).reset_index(drop=True)
prod["repete_anterior"] = prod.groupby(MUN)[ALVO].diff().eq(0)
rep = prod[prod.repete_anterior]
relatorio["repeticoes"] = {
    "total_de_safras_repetidas": int(len(rep)),
    "municipios_com_repeticao": int(rep[MUN].nunique()),
    "por_safra": rep.groupby(ANO).size().to_dict(),
    "por_municipio": rep.groupby(MUN).size().to_dict(),
    "fracao_entre_1991_e_2002_pct": round(float(rep[ANO].between(1991, 2002).mean() * 100), 1),
}

# 4. Matriz município-ano ----------------------------------------------------
matriz = (prod.drop(columns="repete_anterior")
          .merge(clima, on=[MUN, ANO], how="left", validate="one_to_one")
          .sort_values([MUN, ANO]).reset_index(drop=True))
relatorio["matriz"] = {"linhas": len(matriz), "colunas": list(matriz.columns),
                       "celulas_vazias": int(matriz.isna().sum().sum())}
matriz.to_csv(DADOS_PROC / "matriz_mestra.csv", index=False)
salvar_json(relatorio, "2a_auditoria.json")

# Resumo na tela --------------------------------------------------------------
for nome in ("producao", "clima"):
    a = relatorio[nome]
    print(f"{nome:9s}: {a['linhas']} linhas | faltando {len(a['faltando'])} | "
          f"duplicadas {a['duplicadas']} | nomes estranhos {a['nomes_fora_da_lista']}")
print("rendimento:", relatorio["rendimento"])
rp = relatorio["repeticoes"]
print(f"safras com rendimento repetido: {rp['total_de_safras_repetidas']} em "
      f"{rp['municipios_com_repeticao']} municípios; {rp['fracao_entre_1991_e_2002_pct']}% entre 1991 e 2002")
print("  por safra:", rp["por_safra"])
print(f"matriz salva: dados/processados/matriz_mestra.csv ({len(matriz)} linhas)")
