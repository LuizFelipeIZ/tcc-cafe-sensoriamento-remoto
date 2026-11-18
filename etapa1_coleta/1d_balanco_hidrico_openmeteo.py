"""
ETAPA 1d: balanço hídrico de Thornthwaite e Mather (1955), CAD = 100 mm
=======================================================================

O que faz
---------
1. Baixa a temperatura (máxima e média) e a chuva diárias de 1987 a 2023 na
   sede de cada município, pela Historical Weather API do Open-Meteo.
   Essa API usa as reanálises ERA5 e ERA5-Land (e, a partir de 2017, o modelo
   IFS do ECMWF), ou seja, a mesma família de dados da etapa 1b; a diferença
   é que aqui o dado é de um PONTO (a sede), e não a média da área de café.
2. Agrega por mês e calcula o balanço hídrico climatológico mensal:
   ETP de Thornthwaite corrigida pelo fotoperíodo de 21° S, armazenamento
   com CAD de 100 mm (Thornthwaite e Mather) e déficit = ETP - ETR.
3. Resume cada safra t em três variáveis, usadas no previsor de seca:
     * DEF_Florada ......... déficit somado de setembro a novembro de t-1
     * DEF_Expansao_Graos .. déficit somado de dezembro de t-1 a março de t
     * Tmax_Expansao_Graos . temperatura máxima média do mesmo período

Como rodar
----------
Precisa de internet (no seu computador ou no Colab):
    pip install pandas numpy requests
    python etapa1_coleta/1d_balanco_hidrico_openmeteo.py

O download é feito em blocos de 5 anos, com novas tentativas e espera
crescente, porque a API pública recusa pedidos longos ou muito seguidos
(erro 429). Se algum bloco falhar, o log mostra qual; basta rodar de novo.

Saída: dados/brutos/balanco_hidrico_openmeteo_1988_2023.csv
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DADOS_BRUTOS  # noqa: E402

SAIDA = DADOS_BRUTOS / "balanco_hidrico_openmeteo_1988_2023.csv"
CAD = 100.0                     # capacidade de água disponível (mm)
# Duração média do dia (h) por mês a 21° S, para corrigir a ETP
ND = [13.2, 12.8, 12.2, 11.6, 11.0, 10.7, 10.8, 11.3, 11.9, 12.6, 13.1, 13.4]
DIAS_MES = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
ANO_INI, ANO_FIM, BLOCO = 1987, 2023, 5

# Sede de cada município. Itamogi e Juruaia foram corrigidas: as coordenadas antigas caíam em
# Monte Santo de Minas e em Guaxupé (conferido na Malha Municipal do IBGE).
SEDES = {   # (latitude, longitude)
    "Arceburgo": (-21.3636, -46.9391), "Cabo Verde": (-21.4727, -46.3980),
    "Guaranesia": (-21.2986, -46.8005), "Guaxupe": (-21.3052, -46.7122),
    "Itamogi": (-21.0758, -47.0478), "Jacui": (-21.0583, -46.6694),
    "Juruaia": (-21.2539, -46.5739), "Monte Belo": (-21.3211, -46.2936),
    "Muzambinho": (-21.3755, -46.5277), "Nova Resende": (-21.1238, -46.4216),
}


def baixar_bloco(lat, lon, ini, fim, mun, tentativas=6):
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={lat}&longitude={lon}&start_date={ini}&end_date={fim}"
           "&daily=temperature_2m_max,temperature_2m_mean,precipitation_sum"
           "&timezone=America%2FSao_Paulo")
    for t in range(1, tentativas + 1):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as erro:
            espera = 2 ** t
            print(f"  [aviso] {mun} {ini}..{fim}, tentativa {t}: {erro}; aguardando {espera} s")
            time.sleep(espera)
    print(f"  [ERRO] {mun} {ini}..{fim}: bloco não baixado")
    return None


def clima_mensal():
    tabelas = []
    for mun, (lat, lon) in SEDES.items():
        blocos = []
        for a in range(ANO_INI, ANO_FIM + 1, BLOCO):
            js = baixar_bloco(lat, lon, f"{a}-01-01", f"{min(a + BLOCO - 1, ANO_FIM)}-12-31", mun)
            if js is None:
                continue
            blocos.append(pd.DataFrame({
                "Data": pd.to_datetime(js["daily"]["time"]),
                "Tmax": js["daily"]["temperature_2m_max"],
                "Tmean": js["daily"]["temperature_2m_mean"],
                "Precip": js["daily"]["precipitation_sum"],
            }))
            time.sleep(1.5)
        d = pd.concat(blocos, ignore_index=True)
        d["Ano"], d["Mes"] = d.Data.dt.year, d.Data.dt.month
        m = d.groupby(["Ano", "Mes"]).agg(Tmax=("Tmax", "mean"), Tmean=("Tmean", "mean"),
                                         Precip=("Precip", "sum")).reset_index()
        m["Municipio"] = mun
        tabelas.append(m)
        print(f"[ok] {mun}: {m.Ano.nunique()} anos")
    return pd.concat(tabelas, ignore_index=True)


def balanco_hidrico(mensal):
    """Thornthwaite e Mather (1955), passo mensal, CAD = 100 mm."""
    mensal = mensal.sort_values(["Municipio", "Ano", "Mes"]).reset_index(drop=True)
    mensal["ETP"] = 0.0
    mensal["DEF"] = 0.0
    for mun in mensal.Municipio.unique():
        sub = mensal[mensal.Municipio == mun]
        # índice de calor anual (I) e expoente a, com a normal de cada mês
        I = sum(max(0, (t / 5.0) ** 1.514) for t in sub.Tmean.groupby(sub.Mes).mean())
        a = 6.75e-7 * I ** 3 - 7.71e-5 * I ** 2 + 1.792e-2 * I + 0.49239
        arm_ant = CAD
        for i, linha in sub.iterrows():
            mes, T, P = int(linha.Mes) - 1, linha.Tmean, linha.Precip
            if T < 0:
                etp_p = 0.0
            elif T >= 26.5:
                etp_p = -415.85 + 32.24 * T - 0.43 * T ** 2
            else:
                etp_p = 16.0 * (10.0 * T / I) ** a
            etp = etp_p * (ND[mes] / 12.0) * (DIAS_MES[mes] / 30.0)
            p_etp = P - etp
            if p_etp < 0:   # perda de água do solo (negativo acumulado)
                neg = (CAD * math.log(arm_ant / CAD) if arm_ant > 0 else -999) + p_etp
                arm = CAD * math.exp(neg / CAD)
            else:
                arm = min(CAD, arm_ant + p_etp)
            etr = P + abs(arm - arm_ant) if p_etp < 0 else etp
            mensal.loc[i, "ETP"] = etp
            mensal.loc[i, "DEF"] = etp - etr
            arm_ant = arm
    return mensal


def por_safra(bh):
    linhas = []
    for mun, g in bh.groupby("Municipio"):
        for safra in range(1988, 2024):
            flo = g[(g.Ano == safra - 1) & g.Mes.isin([9, 10, 11])]
            exp = g[((g.Ano == safra - 1) & (g.Mes == 12)) | ((g.Ano == safra) & g.Mes.isin([1, 2, 3]))]
            linhas.append({
                "Municipio": mun, "Ano": safra,
                "DEF_Florada": round(flo.DEF.sum(), 2) if len(flo) else np.nan,
                "DEF_Expansao_Graos": round(exp.DEF.sum(), 2) if len(exp) else np.nan,
                "Tmax_Expansao_Graos": round(exp.Tmax.mean(), 2) if len(exp) else np.nan,
            })
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    tabela = por_safra(balanco_hidrico(clima_mensal()))
    tabela.to_csv(SAIDA, index=False)
    print(f"Salvo: {SAIDA} ({len(tabela)} linhas, {int(tabela.DEF_Florada.isna().sum())} vazias)")
