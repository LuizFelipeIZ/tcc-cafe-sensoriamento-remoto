"""
=====================================================================
TCC - Balanço Hídrico Climatológico (Thornthwaite & Mather, 1955)
Versão corrigida: download em blocos anuais com retry/backoff,
entrada na matriz mestra já limpa e auditada.

CORREÇÃO EM RELAÇÃO À VERSÃO ANTERIOR:
  1. ARQUIVO_ENTRADA aponta para a matriz já limpa (sem duplicidade de
     acentuação), evitando a divisão Guaxupe/Guaxupé, Jacui/Jacuí,
     Guaranesia/Guaranésia que quebrava o merge final.
  2. O download deixou de ser um único request de 37 anos por município
     (que estourava timeout/rate limit da API pública do Open-Meteo após
     os primeiros 2-3 municípios, falhando silenciosamente) e passou a
     ser feito em blocos de poucos anos, com retry exponencial e
     mensagens de erro explícitas por bloco.
=====================================================================
"""

import pandas as pd
import numpy as np
import requests
import math
import time

# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
ARQUIVO_ENTRADA = "TCC_Matriz_Mestra_FINAL_1988_2023.xlsx"
ARQUIVO_SAIDA = "TCC_Matriz_Mestra_THORNTHWAITE.xlsx"

CAD = 100.0
LATITUDE = -21.0

ND = [13.2, 12.8, 12.2, 11.6, 11.0, 10.7, 10.8, 11.3, 11.9, 12.6, 13.1, 13.4]
DIAS_MES = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

ANO_INICIO_DOWNLOAD = 1987
ANO_FIM_DOWNLOAD = 2023
BLOCO_ANOS = 5          # baixa 5 anos por requisição, não 37 de uma vez
MAX_TENTATIVAS = 4       # retry com backoff exponencial por bloco

COORDENADAS = {
    'Arceburgo': (-21.3636, -46.9391),
    'Cabo Verde': (-21.4727, -46.3980),
    'Guaranesia': (-21.2986, -46.8005),
    'Guaxupe': (-21.3052, -46.7122),
    'Itamogi': (-21.2369, -47.0177),
    'Jacui': (-21.0583, -46.6694),
    'Juruaia': (-21.3197, -46.6027),
    'Monte Belo': (-21.3211, -46.2936),
    'Muzambinho': (-21.3755, -46.5277),
    'Nova Resende': (-21.1238, -46.4216),
}


def baixar_bloco_com_retry(lat, lon, data_ini, data_fim, mun, tentativas=MAX_TENTATIVAS):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={data_ini}&end_date={data_fim}&"
        f"daily=temperature_2m_max,temperature_2m_mean,precipitation_sum&"
        f"timezone=America%2FSao_Paulo"
    )
    for tentativa in range(1, tentativas + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            espera = 2 ** tentativa  # backoff exponencial: 2, 4, 8, 16s
            print(f"  [AVISO] {mun} ({data_ini} a {data_fim}), tentativa {tentativa}/{tentativas} "
                  f"falhou: {e}. Aguardando {espera}s...")
            time.sleep(espera)
    print(f"  [ERRO DEFINITIVO] {mun} ({data_ini} a {data_fim}): todas as {tentativas} tentativas falharam.")
    return None


def baixar_clima_mensal(municipios):
    print("\n[INFO] Iniciando download climático mensal, em blocos anuais...")
    dados_totais = []

    for mun in municipios:
        if mun not in COORDENADAS:
            print(f"[AVISO] Coordenada não encontrada para '{mun}'. Verifique grafia - pulando.")
            continue

        lat, lon = COORDENADAS[mun]
        print(f"\n[INFO] Baixando dados de {mun}...")
        blocos_mun = []

        for ano_bloco_ini in range(ANO_INICIO_DOWNLOAD, ANO_FIM_DOWNLOAD + 1, BLOCO_ANOS):
            ano_bloco_fim = min(ano_bloco_ini + BLOCO_ANOS - 1, ANO_FIM_DOWNLOAD)
            data_ini = f"{ano_bloco_ini}-01-01"
            data_fim = f"{ano_bloco_fim}-12-31"

            json_data = baixar_bloco_com_retry(lat, lon, data_ini, data_fim, mun)
            if json_data is None:
                continue  # bloco perdido - fica registrado no log, não trava o resto

            df_diario = pd.DataFrame({
                'Data': pd.to_datetime(json_data['daily']['time']),
                'Tmax': json_data['daily']['temperature_2m_max'],
                'Tmean': json_data['daily']['temperature_2m_mean'],
                'Precip': json_data['daily']['precipitation_sum']
            })
            blocos_mun.append(df_diario)
            time.sleep(1.5)  # respeito à API entre blocos

        if not blocos_mun:
            print(f"  [ERRO] Nenhum bloco de {mun} foi baixado com sucesso.")
            continue

        df_diario_mun = pd.concat(blocos_mun, ignore_index=True)
        df_diario_mun['Ano'] = df_diario_mun['Data'].dt.year
        df_diario_mun['Mes'] = df_diario_mun['Data'].dt.month

        df_mensal = df_diario_mun.groupby(['Ano', 'Mes']).agg(
            Tmax=('Tmax', 'mean'),
            Tmean=('Tmean', 'mean'),
            Precip=('Precip', 'sum')
        ).reset_index()
        df_mensal['Municipio'] = mun

        anos_obtidos = df_mensal['Ano'].nunique()
        print(f"  [OK] {mun}: {anos_obtidos} anos de dados mensais obtidos "
              f"(esperado: {ANO_FIM_DOWNLOAD - ANO_INICIO_DOWNLOAD + 1}).")

        dados_totais.append(df_mensal)

    if not dados_totais:
        raise RuntimeError("Nenhum município retornou dados. Verifique conexão com a API Open-Meteo.")

    return pd.concat(dados_totais, ignore_index=True)


def calcular_balanco_hidrico(df_mensal):
    print("\n[INFO] Calculando balanço hídrico (Thornthwaite & Mather, 1955)...")

    df_mensal = df_mensal.sort_values(by=['Municipio', 'Ano', 'Mes']).reset_index(drop=True)
    df_mensal['PET'] = 0.0
    df_mensal['DEF'] = 0.0

    municipios = df_mensal['Municipio'].unique()

    for mun in municipios:
        idx_mun = df_mensal['Municipio'] == mun
        df_mun = df_mensal[idx_mun].copy()

        I_historico = sum([max(0, (t / 5.0) ** 1.514) for t in df_mun['Tmean'].groupby(df_mun['Mes']).mean()])
        a = (6.75e-7 * I_historico**3) - (7.71e-5 * I_historico**2) + (1.792e-2 * I_historico) + 0.49239

        arm_anterior = CAD

        for i, row in df_mun.iterrows():
            mes = int(row['Mes']) - 1
            T = row['Tmean']
            P = row['Precip']

            if T < 0:
                pet_unad = 0
            elif T >= 26.5:
                pet_unad = -415.85 + 32.24 * T - 0.43 * T**2
            else:
                pet_unad = 16.0 * ((10.0 * T / I_historico) ** a)

            cor_fator = (ND[mes] / 12.0) * (DIAS_MES[mes] / 30.0)
            pet = pet_unad * cor_fator

            p_minus_pet = P - pet

            if p_minus_pet < 0:
                neg_acum = CAD * math.log(arm_anterior / CAD) if arm_anterior > 0 else -999
                neg_acum += p_minus_pet
                arm = CAD * math.exp(neg_acum / CAD)
            else:
                arm = min(CAD, arm_anterior + p_minus_pet)

            alt = arm - arm_anterior
            etr = P + abs(alt) if p_minus_pet < 0 else pet
            deficit = pet - etr

            df_mensal.loc[row.name, 'DEF'] = deficit
            df_mensal.loc[row.name, 'PET'] = pet
            arm_anterior = arm

    return df_mensal


def fatiar_fases_agronomicas(df_bh):
    print("\n[INFO] Fatiando o balanço hídrico nas fases fenológicas do café...")

    resultados_safra = []
    municipios = df_bh['Municipio'].unique()
    anos_safra = range(1988, 2024)

    for mun in municipios:
        df_mun = df_bh[df_bh['Municipio'] == mun]

        for ano_safra in anos_safra:
            florada_mask = (df_mun['Ano'] == (ano_safra - 1)) & (df_mun['Mes'].isin([9, 10, 11]))
            expansao_mask = ((df_mun['Ano'] == (ano_safra - 1)) & (df_mun['Mes'] == 12)) | \
                             ((df_mun['Ano'] == ano_safra) & (df_mun['Mes'].isin([1, 2, 3])))

            df_florada = df_mun[florada_mask]
            df_expansao = df_mun[expansao_mask]

            def_florada = df_florada['DEF'].sum() if not df_florada.empty else np.nan
            def_expansao = df_expansao['DEF'].sum() if not df_expansao.empty else np.nan
            tmax_expansao = df_expansao['Tmax'].mean() if not df_expansao.empty else np.nan

            resultados_safra.append({
                'Municipio': mun,
                'Ano': ano_safra,
                'DEF_Florada': round(def_florada, 2) if pd.notna(def_florada) else np.nan,
                'DEF_Expansao_Graos': round(def_expansao, 2) if pd.notna(def_expansao) else np.nan,
                'Tmax_Expansao_Graos': round(tmax_expansao, 2) if pd.notna(tmax_expansao) else np.nan,
            })

    return pd.DataFrame(resultados_safra)


if __name__ == "__main__":
    try:
        print("[INFO] Lendo a Matriz Mestra (versão limpa)...")
        df_mestra = pd.read_excel(ARQUIVO_ENTRADA)
        df_mestra['Municipio'] = df_mestra['Municipio'].str.strip()
        municipios_unicos = df_mestra['Municipio'].unique()
        print(f"[INFO] Municípios na matriz: {sorted(municipios_unicos)}")

        df_mensal = baixar_clima_mensal(municipios_unicos)
        df_bh = calcular_balanco_hidrico(df_mensal)
        df_novas_variaveis = fatiar_fases_agronomicas(df_bh)

        print("\n[INFO] Injetando variáveis de déficit hídrico na Matriz Mestra...")
        df_final = pd.merge(df_mestra, df_novas_variaveis, on=['Municipio', 'Ano'], how='left')

        preenchidas = df_final['DEF_Florada'].notna().sum()
        print(f"\n[INFO] Cobertura final: {preenchidas}/{len(df_final)} linhas com DEF_Florada preenchido.")
        if preenchidas < len(df_final):
            faltando = df_final[df_final['DEF_Florada'].isna()][['Municipio', 'Ano']]
            print("[AVISO] Linhas sem cobertura de déficit hídrico:")
            print(faltando.to_string(index=False))

        df_final.to_excel(ARQUIVO_SAIDA, index=False)
        print(f"\n[SUCESSO] Matriz salva: '{ARQUIVO_SAIDA}'")

    except FileNotFoundError:
        print(f"[ERRO] Não encontrei o arquivo '{ARQUIVO_ENTRADA}'. Coloque-o na mesma pasta!")
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro inesperado: {e}")
