# =====================================================================
# TCC - Indicadores de NDVI (Landsat) por município e safra, VERSÃO CORRIGIDA
# Rode no Google Colab. Gera: TCC_NDVI_Landsat_Corrigido.csv
#
# Correções em relação ao notebook dataset_fase2 (célula 1):
#  1. As máscaras de nuvem e de café passam a ser aplicadas À BANDA DE NDVI.
#     Antes, o NDVI era calculado e acrescentado DEPOIS do updateMask, então
#     ficava sem máscara: a média incluía nuvens e todo o município.
#  2. Máscara de nuvem mais completa (QA_PIXEL bits 1 dilatação, 2 cirros,
#     3 sombra, 4 nuvem) e saturação radiométrica (QA_RADSAT).
#  3. Além do ano civil (igual ao original, para comparação), calcula os
#     indicadores no ANO AGRÍCOLA da safra t (julho de t-1 a junho de t) e a
#     florada que origina a safra t (setembro a novembro de t-1).
#  4. Registra o número de imagens e de pixels de café usados.
# =====================================================================
import ee, pandas as pd
from google.colab import auth, files
auth.authenticate_user()
ee.Initialize(project='spatial-garage-267618')

MUNICIPIOS = ['Arceburgo', 'Cabo Verde', 'Guaranesia', 'Guaxupe', 'Itamogi',
              'Jacui', 'Juruaia', 'Monte Belo', 'Muzambinho', 'Nova Resende']
gaul = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM1_NAME', 'Minas Gerais'))
municipios_fc = gaul.filter(ee.Filter.inList('ADM2_NAME', MUNICIPIOS))
print('Municípios encontrados no GAUL:', municipios_fc.aggregate_array('ADM2_NAME').getInfo())

mapbiomas = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1')

CORTE_L8 = '2013-04-01'  # Landsat 8 a partir de abril de 2013; Landsat 7 antes

def colecao_landsat(ini, fim):
    """Datas no formato 'AAAA-MM-DD'. Evita intervalos vazios ou invertidos."""
    partes = []
    if ini < CORTE_L8:  # trecho anterior ao Landsat 8
        partes.append(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
                      .filterDate(ini, min(fim, CORTE_L8))
                      .map(lambda i: i.set({'NIR': 'SR_B4', 'RED': 'SR_B3'})))
    if fim > CORTE_L8:  # trecho do Landsat 8
        partes.append(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                      .filterDate(max(ini, CORTE_L8), fim)
                      .map(lambda i: i.set({'NIR': 'SR_B5', 'RED': 'SR_B4'})))
    col = partes[0] if len(partes) == 1 else partes[0].merge(partes[1])
    return col.filterBounds(municipios_fc)

def ndvi_mascarado(img, mascara_cafe):
    qa = img.select('QA_PIXEL')
    limpo = (qa.bitwiseAnd(1 << 1).eq(0)   # nuvem dilatada
             .And(qa.bitwiseAnd(1 << 2).eq(0))  # cirros
             .And(qa.bitwiseAnd(1 << 3).eq(0))  # sombra de nuvem
             .And(qa.bitwiseAnd(1 << 4).eq(0))  # nuvem
             .And(img.select('QA_RADSAT').eq(0)))
    refl = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    nir = refl.select(ee.List([img.get('NIR')])); red = refl.select(ee.List([img.get('RED')]))
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    # CORREÇÃO PRINCIPAL: a máscara é aplicada ao próprio NDVI
    return ee.Image(ndvi.updateMask(limpo).updateMask(mascara_cafe)
                    .copyProperties(img, ['system:time_start']))

def indicadores(ini, fim, ini_flor, fim_flor, mascara_cafe):
    col = colecao_landsat(ini, fim).map(lambda i: ndvi_mascarado(i, mascara_cafe))
    flor = colecao_landsat(ini_flor, fim_flor).map(lambda i: ndvi_mascarado(i, mascara_cafe))
    img = ee.Image.cat([
        col.reduce(ee.Reducer.percentile([90])).rename('Vigor'),
        col.reduce(ee.Reducer.percentile([10])).rename('Estresse'),
        flor.mean().rename('Florada'),
        col.count().rename('N_obs'),
    ])
    return img, col.size()

linhas = []
for safra in range(2011, 2024):
    masc = mapbiomas.select(f'classification_{min(safra, 2023)}').eq(46).selfMask()
    versoes = {
        'civil': (f'{safra}-01-01', f'{safra+1}-01-01', f'{safra}-09-01', f'{safra}-12-01'),
        'agricola': (f'{safra-1}-07-01', f'{safra}-07-01', f'{safra-1}-09-01', f'{safra-1}-12-01'),
    }
    for nome, (ini, fim, fi, ff) in versoes.items():
        img, n_img = indicadores(ini, fim, fi, ff, masc)
        est = img.addBands(masc.rename('Pixels_Cafe')).reduceRegions(
            collection=municipios_fc,
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            scale=30, tileScale=4).getInfo()
        n = n_img.getInfo()
        for f in est['features']:
            p = f['properties']
            linhas.append({'Ano': safra, 'Municipio': p['ADM2_NAME'], 'Janela': nome,
                           'NDVI_Vigor_Maximo': p.get('Vigor_mean'), 'NDVI_Estresse_Maximo': p.get('Estresse_mean'),
                           'NDVI_Florada': p.get('Florada_mean'), 'N_obs_medio_pixel': p.get('N_obs_mean'),
                           'Pixels_Cafe': p.get('Pixels_Cafe_count'), 'N_imagens': n})
        print(f'Safra {safra} ({nome}): {n} imagens')

df = pd.DataFrame(linhas)
df['NDVI_Amplitude'] = df['NDVI_Vigor_Maximo'] - df['NDVI_Estresse_Maximo']
df.to_csv('TCC_NDVI_Landsat_Corrigido.csv', index=False)
print(df.groupby('Janela')[['NDVI_Vigor_Maximo', 'NDVI_Estresse_Maximo', 'NDVI_Florada']].describe().T)
files.download('TCC_NDVI_Landsat_Corrigido.csv')
