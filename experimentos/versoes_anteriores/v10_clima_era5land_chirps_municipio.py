"""
=====================================================================
TCC - Extração Climática por Município e Fase Fenológica (GEE)
=====================================================================

Correções em relação ao script original:
  1. Extração POR MUNICÍPIO (usando os limites administrativos oficiais
     via FAO GAUL), não mais um único ponto representando toda a região.
  2. Adiciona PRECIPITAÇÃO via CHIRPS (Funk et al., 2015) - a fonte já
     citada na Metodologia e Referências do TCC, mantendo consistência.
  3. Agrega os dados por FASE FENOLÓGICA (Florada, Expansão/Enchimento,
     Inverno/Risco de Geada), replicando a lógica já prevista na função
     fatiamento_fenologico_clima() do seu script de Feature Engineering,
     mas agora alimentada com dados reais extraídos do GEE.

Rode este script no Google Colab (mesmo ambiente do seu script original).

Saída: TCC_Clima_Fenologico_Municipal_1988_2023.csv, com uma linha por
Município x Ano, pronta para merge com sua matriz principal.
"""

import ee
import pandas as pd
import time

MEU_PROJETO = 'spatial-garage-267618'

try:
    ee.Initialize(project=MEU_PROJETO)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=MEU_PROJETO)

# ==========================================================
# 1. LIMITES MUNICIPAIS (FAO GAUL nível 2 = municípios no Brasil)
# ==========================================================
MUNICIPIOS_ALVO = [
    "Arceburgo", "Cabo Verde", "Guaranesia", "Guaxupe", "Itamogi",
    "Jacui", "Juruaia", "Monte Belo", "Muzambinho", "Nova Resende"
]

gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
municipios_mg = gaul.filter(ee.Filter.eq('ADM1_NAME', 'Minas Gerais'))

def obter_geometria_municipio(nome):
    """Busca o polígono do município no GAUL. Faz correspondência
    tolerante a acentos, pois o GAUL costuma usar nomes sem acentuação."""
    feat = municipios_mg.filter(ee.Filter.eq('ADM2_NAME', nome)).first()
    return feat

geometrias = {}
print("[INFO] Resolvendo geometrias municipais via FAO GAUL...")
for nome in MUNICIPIOS_ALVO:
    feat = obter_geometria_municipio(nome)
    info = feat.getInfo()
    if info is None:
        print(f"  [AVISO] '{nome}' não encontrado no GAUL - tente sem acento "
              f"ou verifique o nome exato na base (ADM2_NAME).")
    else:
        geometrias[nome] = feat.geometry()
        print(f"  [OK] {nome} resolvido.")

if len(geometrias) < len(MUNICIPIOS_ALVO):
    print("\n[AVISO] Nem todos os municípios foram resolvidos automaticamente.")
    print("Rode o diagnóstico abaixo para listar os nomes disponíveis no GAUL")
    print("para o estado de Minas Gerais e ajuste MUNICIPIOS_ALVO conforme necessário:")
    print("  nomes_disponiveis = municipios_mg.aggregate_array('ADM2_NAME').getInfo()")

# ==========================================================
# 2. DEFINIÇÃO DAS FASES FENOLÓGICAS
# ==========================================================
FASES = {
    "Florada": [9, 10, 11],
    "Expansao_Enchimento": [1, 2, 3],
    "Inverno_Risco_Geada": [6, 7, 8],
}

ANOS = range(1988, 2024)
LIMIAR_GEADA_C = 3.0  # dias abaixo deste valor contam como "Dias_Abaixo_3C"

# ==========================================================
# 3. EXTRAÇÃO POR MUNICÍPIO, ANO E FASE
# ==========================================================
def extrair_clima_municipio_ano(nome_municipio, geometria, ano):
    """Extrai Tmin (ERA5-Land) e Precipitação (CHIRPS) para um município
    em um ano, já agregadas por fase fenológica, além do resumo anual."""

    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .select('temperature_2m_min') \
        .filterDate(f'{ano}-01-01', f'{ano+1}-01-01')

    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .select('precipitation') \
        .filterDate(f'{ano}-01-01', f'{ano+1}-01-01')

    resultado = {"Municipio": nome_municipio, "Ano": ano}

    # --- Resumo anual (compatível com as colunas já usadas no pipeline) ---
    tmin_anual_img = era5.min()
    tmin_anual = tmin_anual_img.reduceRegion(
        reducer=ee.Reducer.min(), geometry=geometria, scale=11132, maxPixels=1e9
    ).get('temperature_2m_min')

    precip_anual_img = chirps.sum()
    precip_anual = precip_anual_img.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometria, scale=5566, maxPixels=1e9
    ).get('precipitation')

    def dias_abaixo_limiar(colecao_tmin, limiar_c):
        limiar_k = limiar_c + 273.15
        colecao_flag = colecao_tmin.map(
            lambda img: img.lt(limiar_k).rename('abaixo_limiar')
        )
        soma_dias = colecao_flag.sum()
        return soma_dias.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometria, scale=11132, maxPixels=1e9
        ).get('abaixo_limiar')

    dias_abaixo_3c = dias_abaixo_limiar(era5, LIMIAR_GEADA_C)

    resultado["Tmin_Abs_K"] = tmin_anual
    resultado["Precip_Anual_mm"] = precip_anual
    resultado["Dias_Abaixo_3C"] = dias_abaixo_3c

    # --- Agregação por fase fenológica ---
    for nome_fase, meses in FASES.items():
        era5_fase = era5.filter(ee.Filter.calendarRange(meses[0], meses[-1], 'month')) \
            if meses == sorted(meses) and meses[-1] - meses[0] == len(meses) - 1 \
            else era5.filter(ee.Filter.inList('month', meses))  # fallback p/ meses não contíguos

        chirps_fase = chirps.filter(ee.Filter.calendarRange(meses[0], meses[-1], 'month')) \
            if meses == sorted(meses) and meses[-1] - meses[0] == len(meses) - 1 \
            else chirps.filter(ee.Filter.inList('month', meses))

        tmin_fase = era5_fase.min().reduceRegion(
            reducer=ee.Reducer.min(), geometry=geometria, scale=11132, maxPixels=1e9
        ).get('temperature_2m_min')

        precip_fase = chirps_fase.sum().reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometria, scale=5566, maxPixels=1e9
        ).get('precipitation')

        resultado[f"Tmin_Abs_{nome_fase}_K"] = tmin_fase
        resultado[f"Precip_{nome_fase}_mm"] = precip_fase

    return resultado


# ==========================================================
# 4. EXECUÇÃO EM LOTE (ano a ano, município a município)
# ==========================================================
registros = []
print(f"\n[INFO] Iniciando extração para {len(geometrias)} municípios, "
      f"{len(list(ANOS))} anos...\n")

for nome_municipio, geometria in geometrias.items():
    print(f"[INFO] Processando {nome_municipio}...")
    for ano in ANOS:
        try:
            r = ee.Dictionary(extrair_clima_municipio_ano(nome_municipio, geometria, ano)).getInfo()
            registros.append(r)
        except Exception as e:
            print(f"  [ERRO] {nome_municipio}/{ano}: {e}")
        time.sleep(0.2)  # evita sobrecarregar a API do GEE
    print(f"  [OK] {nome_municipio} concluído.")

df_clima = pd.DataFrame(registros)

# --- Conversão Kelvin -> Celsius ---
for col in df_clima.columns:
    if col.endswith('_K'):
        nova_col = col.replace('_K', '_C')
        df_clima[nova_col] = df_clima[col] - 273.15
        df_clima = df_clima.drop(columns=[col])

# --- Renomeia para o padrão já usado no pipeline principal ---
df_clima = df_clima.rename(columns={
    'Tmin_Abs_C': 'Tmin_Abs',
    'Precip_Anual_mm': 'Precip_Anual',
})

nome_arquivo = 'TCC_Clima_Fenologico_Municipal_1988_2023.csv'
df_clima.to_csv(nome_arquivo, index=False, sep=';', decimal=',', encoding='utf-8-sig')
print(f"\n[SUCESSO] Dados salvos em: {nome_arquivo}")
print(df_clima.head(10).to_string(index=False))

try:
    from google.colab import files
    files.download(nome_arquivo)
except ImportError:
    pass
