import pandas as pd
import numpy as np
import requests
import math
import time
from datetime import datetime

# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
ARQUIVO_ENTRADA = "TCC_Matriz_Mestra_Definitiva_1988_2023_.xlsx"
ARQUIVO_SAIDA = "TCC_Matriz_Mestra_THORNTHWAITE.xlsx"

# Parâmetros de Thornthwaite & Mather (Padrão para Café no Sul de Minas)
CAD = 100.0
LATITUDE = -21.0

# Horas de sol máximas por mês para Latitude -21°S (Fator de Correção Fotoperíodo)
ND = [13.2, 12.8, 12.2, 11.6, 11.0, 10.7, 10.8, 11.3, 11.9, 12.6, 13.1, 13.4]
DIAS_MES = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Dicionário de Coordenadas para baixar o clima exato
COORDENADAS = {
    'Arceburgo': (-21.3636, -46.9391),
    'Bom Jesus da Penha': (-21.0152, -46.5177),
    'Cabo Verde': (-21.4727, -46.3980),
    'Capetinga': (-20.6127, -47.0083),
    'Guaranésia': (-21.2986, -46.8005),
    'Guaxupé': (-21.3052, -46.7122),
    'Itamogi': (-21.2369, -47.0177),
    'Jacuí': (-21.0583, -46.6694),
    'Juruaia': (-21.3197, -46.6027),
    'Monte Belo': (-21.3211, -46.2936),
    'Monte Santo de Minas': (-21.1963, -46.9805),
    'Muzambinho': (-21.3755, -46.5277),
    'Nova Resende': (-21.1238, -46.4216),
    'São Pedro da União': (-21.1352, -46.6122),
    'São Sebastião do Paraíso': (-20.9172, -46.9906),
    'São Tomás de Aquino': (-20.7852, -47.1008)
}

def baixar_clima_mensal(municipios):
    print("\n☁️ INICIANDO DOWNLOAD CLIMÁTICO MENSAL (1987 a 2023)...")
    print("Precisamos de 1987 para calcular a florada da safra de 1988!")

    dados_totais = []

    for mun in municipios:
        if mun not in COORDENADAS:
            print(f"⚠️ Coordenada não encontrada para {mun}. Pulando...")
            continue

        lat, lon = COORDENADAS[mun]
        print(f"Baixando dados de {mun}...")

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date=1987-01-01&end_date=2023-12-31&"
            f"daily=temperature_2m_max,temperature_2m_mean,precipitation_sum&"
            f"timezone=America%2FSao_Paulo"
        )

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            json_data = resp.json()

            df_diario = pd.DataFrame({
                'Data': pd.to_datetime(json_data['daily']['time']),
                'Tmax': json_data['daily']['temperature_2m_max'],
                'Tmean': json_data['daily']['temperature_2m_mean'],
                'Precip': json_data['daily']['precipitation_sum']
            })

            # Agrupa os dias em Mêses
            df_diario['Ano'] = df_diario['Data'].dt.year
            df_diario['Mes'] = df_diario['Data'].dt.month

            df_mensal = df_diario.groupby(['Ano', 'Mes']).agg(
                Tmax=('Tmax', 'mean'),   # Média das máximas do mês
                Tmean=('Tmean', 'mean'), # Média das médias
                Precip=('Precip', 'sum') # Soma da chuva
            ).reset_index()

            df_mensal['Municipio'] = mun
            dados_totais.append(df_mensal)
            time.sleep(1) # Respeito à API

        except Exception as e:
            print(f"❌ Erro ao baixar {mun}: {e}")

    return pd.concat(dados_totais, ignore_index=True)

def calcular_balanco_hidrico(df_mensal):
    print("\n💧 CALCULANDO A FÍSICA DO SOLO (Thornthwaite & Mather, 1955)...")

    df_mensal = df_mensal.sort_values(by=['Municipio', 'Ano', 'Mes']).reset_index(drop=True)
    df_mensal['PET'] = 0.0
    df_mensal['DEF'] = 0.0

    municipios = df_mensal['Municipio'].unique()

    for mun in municipios:
        idx_mun = df_mensal['Municipio'] == mun
        df_mun = df_mensal[idx_mun].copy()

        # O Índice de Calor (I) exige a soma anual das temperaturas.
        # Simplificação agronômica: Usar a média histórica do município para evitar saltos.
        I_historico = sum([max(0, (t / 5.0)**1.514) for t in df_mun['Tmean'].groupby(df_mun['Mes']).mean()])
        a = (6.75e-7 * I_historico**3) - (7.71e-5 * I_historico**2) + (1.792e-2 * I_historico) + 0.49239

        arm_anterior = CAD # O solo começa cheio no verão de 1987

        for i, row in df_mun.iterrows():
            mes = int(row['Mes']) - 1
            T = row['Tmean']
            P = row['Precip']

            # Evapotranspiração Potencial (PET)
            if T < 0:
                pet_unad = 0
            elif T >= 26.5:
                pet_unad = -415.85 + 32.24 * T - 0.43 * T**2
            else:
                pet_unad = 16.0 * ((10.0 * T / I_historico) ** a)

            cor_fator = (ND[mes] / 12.0) * (DIAS_MES[mes] / 30.0)
            pet = pet_unad * cor_fator

            p_minus_pet = P - pet

            # Armazenamento (ARM)
            if p_minus_pet < 0:
                neg_acum = CAD * math.log(arm_anterior / CAD) if arm_anterior > 0 else -999
                neg_acum += p_minus_pet
                arm = CAD * math.exp(neg_acum / CAD)
            else:
                arm = min(CAD, arm_anterior + p_minus_pet)

            # Alteração e Evapotranspiração Real (ETR)
            alt = arm - arm_anterior
            etr = P + abs(alt) if p_minus_pet < 0 else pet

            # Déficit Hídrico (O alvo do nosso ML!)
            deficit = pet - etr

            df_mensal.loc[row.name, 'DEF'] = deficit
            arm_anterior = arm

    return df_mensal

def fatiar_fases_agronomicas(df_bh):
    print("\n✂️ FATIANDO O CLIMA NAS FASES FENOLÓGICAS (Adequação da Safra)...")

    resultados_safra = []

    municipios = df_bh['Municipio'].unique()
    anos_safra = range(1988, 2024)

    for mun in municipios:
        df_mun = df_bh[df_bh['Municipio'] == mun]

        for ano_safra in anos_safra:
            # A SAFRA DO CAFÉ:
            # A florada ocorre no ano anterior (Setembro a Novembro de Ano-1)
            florada_mask = (df_mun['Ano'] == (ano_safra - 1)) & (df_mun['Mes'].isin([9, 10, 11]))
            # A expansão ocorre de Dezembro (Ano-1) até Março (Ano_Safra)
            expansao_mask = ((df_mun['Ano'] == (ano_safra - 1)) & (df_mun['Mes'] == 12)) | \
                            ((df_mun['Ano'] == ano_safra) & (df_mun['Mes'].isin([1, 2, 3])))

            df_florada = df_mun[florada_mask]
            df_expansao = df_mun[expansao_mask]

            # Extraindo as variáveis superpoderosas
            def_florada = df_florada['DEF'].sum() if not df_florada.empty else 0
            def_expansao = df_expansao['DEF'].sum() if not df_expansao.empty else 0
            tmax_expansao = df_expansao['Tmax'].mean() if not df_expansao.empty else 0

            resultados_safra.append({
                'Municipio': mun,
                'Ano': ano_safra,
                'DEF_Florada': round(def_florada, 2),
                'DEF_Expansao_Graos': round(def_expansao, 2),
                'Tmax_Expansao_Graos': round(tmax_expansao, 2)
            })

    return pd.DataFrame(resultados_safra)

if __name__ == "__main__":
    try:
        print("1. Lendo a Matriz Mestra original...")
        df_mestra = pd.read_excel(ARQUIVO_ENTRADA)
        municipios_unicos = df_mestra['Municipio'].str.strip().str.title().unique()

        # Garante a formatação do nome na mestra
        df_mestra['Municipio'] = df_mestra['Municipio'].str.strip().str.title()

        # Etapas de Inteligência
        df_mensal = baixar_clima_mensal(municipios_unicos)
        df_bh = calcular_balanco_hidrico(df_mensal)
        df_novas_variaveis = fatiar_fases_agronomicas(df_bh)

        # Mesclando as novas variáveis de Inteligência Agronômica com a sua Matriz Mestra
        print("\n2. Injetando variáveis na Matriz Mestra...")
        df_final = pd.merge(df_mestra, df_novas_variaveis, on=['Municipio', 'Ano'], how='left')

        # Salvando a obra-prima
        df_final.to_excel(ARQUIVO_SAIDA, index=False)

        print("\n=========================================================")
        print(f"🎉 MATRIZ SUPREMA GERADA COM SUCESSO: '{ARQUIVO_SAIDA}'")
        print("Três novas colunas foram adicionadas perfeitamente alinhadas ao calendário do café:")
        print(" - DEF_Florada (Soma do Déficit Hídrico de Set a Nov do ano anterior)")
        print(" - DEF_Expansao_Graos (Déficit Hídrico de Dez a Março)")
        print(" - Tmax_Expansao_Graos (Média da Temp Máxima no crescimento dos grãos)")
        print("=========================================================")

    except FileNotFoundError:
        print(f"❌ Erro: Não encontrei o arquivo {ARQUIVO_ENTRADA}. Coloque-o na mesma pasta!")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}")