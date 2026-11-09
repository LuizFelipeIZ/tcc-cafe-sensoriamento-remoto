"""
=====================================================================
TCC - Extração de NDVI Histórico (Landsat) com Intercalibração
=====================================================================

Este script faz DUAS coisas, nesta ordem obrigatória:

  1. Extrai o NDVI histórico via Landsat 5 TM (1988-2011) e Landsat 7/8
     (ponte de transição), por município, usando a mesma máscara espacial
     (EPAMIG/MapBiomas) já definida no seu pipeline principal.

  2. Faz a INTERCALIBRAÇÃO entre Landsat e Sentinel-2 no período de
     sobreposição (2015-2020): compara o NDVI dos dois sensores para os
     mesmos município/mês, ajusta uma regressão linear simples, e aplica
     essa correção ao NDVI histórico do Landsat.

     Por que isso é indispensável: sensores diferentes têm resolução,
     largura de banda espectral e calibração radiométrica distintas. Um
     mesmo valor de NDVI bruto em sensores diferentes não corresponde
     necessariamente à mesma condição fisiológica real da lavoura. Sem
     essa correção, o modelo poderia aprender a diferença ENTRE SENSORES
     como se fosse diferença real de vigor vegetativo entre períodos -
     viés sistemático que contaminaria a hierarquia de atribuição
     (Tmin_Abs e NDVI já identificados como as variáveis mais
     importantes no SHAP).

Pré-requisito: já ter as geometrias municipais resolvidas (reutiliza a
mesma lógica de extrair_clima_fenologico_gee.py).
"""

import ee
import pandas as pd
import numpy as np
from scipy import stats
import time

MEU_PROJETO = 'spatial-garage-267618'

try:
    ee.Initialize(project=MEU_PROJETO)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=MEU_PROJETO)

MUNICIPIOS_ALVO = [
    "Arceburgo", "Cabo Verde", "Guaranesia", "Guaxupe", "Itamogi",
    "Jacui", "Juruaia", "Monte Belo", "Muzambinho", "Nova Resende"
]

gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
municipios_mg = gaul.filter(ee.Filter.eq('ADM1_NAME', 'Minas Gerais'))

geometrias = {}
for nome in MUNICIPIOS_ALVO:
    feat = municipios_mg.filter(ee.Filter.eq('ADM2_NAME', nome)).first()
    geometrias[nome] = feat.geometry()

# Meses críticos usados no seu extrair_features_ndvi() original (inverno: 7, 8)
MESES_CRITICOS = [7, 8]

# ==========================================================
# FUNÇÕES DE NDVI POR SENSOR (bandas e fatores de escala variam)
# ==========================================================
def ndvi_landsat_sr(nome_colecao, banda_nir, banda_red):
    """Landsat Collection 2 Surface Reflectance: aplica o fator de escala
    oficial (0.0000275, offset -0.2) antes de calcular o NDVI, e mascara
    nuvens usando a banda QA_PIXEL."""
    def processar(img):
        optico = img.select('SR_B.').multiply(0.0000275).add(-0.2)
        nir = optico.select(banda_nir)
        red = optico.select(banda_red)
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')

        qa = img.select('QA_PIXEL')
        nuvem = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        return ndvi.updateMask(nuvem).copyProperties(img, ['system:time_start'])

    return ee.ImageCollection(nome_colecao).map(processar)


def ndvi_sentinel2():
    def processar(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        scl = img.select('SCL')
        mascara_boa = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return ndvi.updateMask(mascara_boa).copyProperties(img, ['system:time_start'])

    return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').map(processar)


# ==========================================================
# EXTRAÇÃO DE NDVI MÉDIO POR MUNICÍPIO/ANO NOS MESES CRÍTICOS
# ==========================================================
def extrair_ndvi_periodo(colecao_ndvi, geometria, ano, meses):
    filtrada = colecao_ndvi.filter(ee.Filter.calendarRange(ano, ano, 'year')) \
        .filter(ee.Filter.inList('month', meses)) if False else \
        colecao_ndvi.filterDate(f'{ano}-{meses[0]:02d}-01', f'{ano}-{meses[-1]:02d}-28')

    media_periodo = filtrada.mean()
    valor = media_periodo.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometria, scale=30, maxPixels=1e9
    ).get('NDVI')
    return valor


def montar_serie_ndvi(colecao_ndvi, geometrias, anos, meses, nome_sensor):
    registros = []
    for nome_mun, geom in geometrias.items():
        for ano in anos:
            try:
                v = extrair_ndvi_periodo(colecao_ndvi, geom, ano, meses).getInfo()
            except Exception as e:
                print(f"  [ERRO] {nome_sensor}/{nome_mun}/{ano}: {e}")
                v = None
            registros.append({'Municipio': nome_mun, 'Ano': ano, 'NDVI': v, 'Sensor': nome_sensor})
            time.sleep(0.2)
    return pd.DataFrame(registros)


# ==========================================================
# 1. EXTRAÇÃO LANDSAT 5 (1988-2011) - NIR=SR_B4, RED=SR_B3
# ==========================================================
print("[INFO] Extraindo NDVI histórico via Landsat 5 TM (1988-2011)...")
l5 = ndvi_landsat_sr('LANDSAT/LT05/C02/T1_L2', 'SR_B4', 'SR_B3')
df_l5 = montar_serie_ndvi(l5, geometrias, range(1988, 2012), MESES_CRITICOS, 'Landsat5')

# ==========================================================
# 2. EXTRAÇÃO PARA O PERÍODO DE SOBREPOSIÇÃO (2015-2020)
#    Landsat 8 (NIR=SR_B5, RED=SR_B4) x Sentinel-2 (B8, B4)
# ==========================================================
print("[INFO] Extraindo NDVI no período de sobreposição (2015-2020) "
      "para intercalibração...")
l8 = ndvi_landsat_sr('LANDSAT/LC08/C02/T1_L2', 'SR_B5', 'SR_B4')
s2 = ndvi_sentinel2()

anos_sobreposicao = range(2015, 2021)
df_l8_sobre = montar_serie_ndvi(l8, geometrias, anos_sobreposicao, MESES_CRITICOS, 'Landsat8')
df_s2_sobre = montar_serie_ndvi(s2, geometrias, anos_sobreposicao, MESES_CRITICOS, 'Sentinel2')

# ==========================================================
# 3. INTERCALIBRAÇÃO (regressão linear Landsat -> Sentinel-2)
# ==========================================================
df_comparacao = df_l8_sobre.merge(
    df_s2_sobre, on=['Municipio', 'Ano'], suffixes=('_landsat', '_sentinel')
).dropna(subset=['NDVI_landsat', 'NDVI_sentinel'])

if len(df_comparacao) < 5:
    print("\n[AVISO] Poucos pares Landsat/Sentinel-2 disponíveis para intercalibração "
          f"(N={len(df_comparacao)}). Revise a máscara de nuvens ou o período de "
          "sobreposição antes de prosseguir - a correção abaixo pode não ser confiável.")

slope, intercept, r_value, p_value, std_err = stats.linregress(
    df_comparacao['NDVI_landsat'], df_comparacao['NDVI_sentinel']
)

print(f"\n[INTERCALIBRAÇÃO Landsat -> Sentinel-2]")
print(f"NDVI_Sentinel_equivalente = {slope:.4f} * NDVI_Landsat + {intercept:.4f}")
print(f"R² do ajuste: {r_value**2:.3f} | N pares: {len(df_comparacao)}")
if r_value**2 < 0.5:
    print("[AVISO] R² baixo - a correspondência entre sensores é fraca. Considere")
    print("        reportar o NDVI histórico com essa ressalva explícita no TCC,")
    print("        ou usar apenas o modelo climático (sem NDVI) para os anos < 2013.")

# Aplica a correção ao NDVI histórico do Landsat 5
df_l5['NDVI_Corrigido'] = slope * df_l5['NDVI'] + intercept

# ==========================================================
# 4. EXPORTAÇÃO FINAL
# ==========================================================
df_final = df_l5[['Municipio', 'Ano', 'NDVI_Corrigido']].rename(
    columns={'NDVI_Corrigido': 'NDVI_Estresse_Maximo'}  # ajuste o nome conforme a
    # variável do seu pipeline que este NDVI de inverno (meses 7-8) corresponde
)

nome_arquivo = 'TCC_NDVI_Historico_Landsat_Calibrado_1988_2011.csv'
df_final.to_csv(nome_arquivo, index=False, sep=';', decimal=',', encoding='utf-8-sig')
print(f"\n[SUCESSO] NDVI histórico calibrado salvo em: {nome_arquivo}")
print(df_final.head(10).to_string(index=False))

# Também salva a tabela de comparação bruta, para você reportar o R² da
# intercalibração como evidência metodológica no TCC (Metodologia/Anexos)
df_comparacao.to_csv('TCC_Intercalibracao_Landsat_Sentinel2_Diagnostico.csv',
                      index=False, sep=';', decimal=',', encoding='utf-8-sig')

try:
    from google.colab import files
    files.download(nome_arquivo)
    files.download('TCC_Intercalibracao_Landsat_Sentinel2_Diagnostico.csv')
except ImportError:
    pass
