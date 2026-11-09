# =====================================================================
# TCC - Imagens de satélite para a apresentação de defesa
# Rode em uma célula nova do Google Colab (precisa de conta no Earth Engine).
# No fim, baixa um arquivo imagens_apresentacao.zip com 7 imagens PNG.
# =====================================================================
import ee, requests, zipfile, os

MEU_PROJETO = 'spatial-garage-267618'   # o mesmo Project ID dos scripts anteriores
try:
    ee.Initialize(project=MEU_PROJETO)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=MEU_PROJETO)

# ---------------------------------------------------------------- municípios
NOMES = ['Arceburgo', 'Cabo Verde', 'Guaranésia', 'Guaranesia', 'Guaxupé', 'Guaxupe', 'Itamogi',
         'Jacuí', 'Jacui', 'Juruaia', 'Monte Belo', 'Muzambinho', 'Nova Resende']
gaul = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM1_NAME', 'Minas Gerais'))
mun = gaul.filter(ee.Filter.inList('ADM2_NAME', NOMES))
print('Municípios encontrados:', mun.size().getInfo(), mun.aggregate_array('ADM2_NAME').getInfo())
regiao = mun.geometry().bounds().buffer(3000).bounds()
contorno = ee.Image().byte().paint(featureCollection=mun, color=1, width=2).visualize(palette=['ffffff'])

# ---------------------------------------------------------------- máscara de café (MapBiomas Col. 9)
mapbiomas = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1')
def cafe(ano):
    return mapbiomas.select(f'classification_{ano}').eq(46).selfMask()

# ---------------------------------------------------------------- Landsat 8, Coleção 2, nível 2
def l8(ini, fim):
    def prep(img):
        qa = img.select('QA_PIXEL')
        ok = qa.bitwiseAnd(0b11110).eq(0)          # sem borda de nuvem, cirros, nuvem e sombra
        sr = img.select('SR_B.').multiply(0.0000275).add(-0.2)
        return sr.updateMask(ok)
    return (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(regiao)
            .filterDate(ini, fim).map(prep).median().clip(regiao))

def ndvi(img):
    return img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

def cor_real(img, escurecer=1.0):
    v = img.visualize(bands=['SR_B4', 'SR_B3', 'SR_B2'], min=0.0, max=0.25, gamma=1.3)
    return v.multiply(escurecer).uint8() if escurecer != 1.0 else v

PAL_NDVI = ['8c510a', 'd8b365', 'f6e8c3', 'c7eae5', '5ab4ac', '1b7837', '00441b']
PAL_DIF = ['b2182b', 'ef8a62', 'fddbc7', 'f7f7f7', 'd1e5f0', '67a9cf', '2166ac']

seca23 = l8('2023-06-01', '2023-09-30')
antes = l8('2021-06-01', '2021-07-19')        # antes da geada de 20 de julho de 2021
depois = l8('2021-08-01', '2021-09-20')       # depois da geada
m21 = cafe(2021)

IMAGENS = {
    '01_cor_real_regiao': ee.ImageCollection([cor_real(seca23), contorno]).mosaic(),
    '02_mascara_cafe_2023': ee.ImageCollection([
        cor_real(seca23, 0.45), cafe(2023).visualize(palette=['2ee66b']), contorno]).mosaic(),
    '03_ndvi_antes_geada_2021': ee.ImageCollection([
        cor_real(antes, 0.4), ndvi(antes).updateMask(m21).visualize(min=0.2, max=0.9, palette=PAL_NDVI), contorno]).mosaic(),
    '04_ndvi_depois_geada_2021': ee.ImageCollection([
        cor_real(depois, 0.4), ndvi(depois).updateMask(m21).visualize(min=0.2, max=0.9, palette=PAL_NDVI), contorno]).mosaic(),
    '05_queda_ndvi_geada_2021': ee.ImageCollection([
        cor_real(depois, 0.4), ndvi(depois).subtract(ndvi(antes)).updateMask(m21)
        .visualize(min=-0.25, max=0.25, palette=PAL_DIF), contorno]).mosaic(),
}

# ---------------------------------------------------------------- recorte aproximado em uma área de café (Sentinel-2, 10 m)
muz = mun.filter(ee.Filter.eq('ADM2_NAME', 'Muzambinho')).geometry()
centro = (ee.Image.pixelLonLat().updateMask(cafe(2023))
          .reduceRegion(ee.Reducer.mean(), muz, 30, maxPixels=1e9))
ponto = ee.Geometry.Point([centro.get('longitude'), centro.get('latitude')])
zoom = ponto.buffer(2500).bounds()
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(zoom)
      .filterDate('2023-06-01', '2023-09-30').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
      .median().divide(10000).clip(zoom))
IMAGENS['06_cafe_zoom_sentinel2'] = s2.visualize(bands=['B4', 'B3', 'B2'], min=0.0, max=0.22, gamma=1.3)
IMAGENS['07_cafe_zoom_mascara'] = ee.ImageCollection([
    s2.visualize(bands=['B4', 'B3', 'B2'], min=0.0, max=0.22, gamma=1.3).multiply(0.55).uint8(),
    cafe(2023).clip(zoom).visualize(palette=['2ee66b'], opacity=0.75)]).mosaic()

# ---------------------------------------------------------------- download
os.makedirs('imagens_apresentacao', exist_ok=True)
for nome, img in IMAGENS.items():
    reg = zoom if 'zoom' in nome else regiao
    url = img.getThumbURL({'region': reg, 'dimensions': 1800, 'format': 'png'})
    arq = f'imagens_apresentacao/{nome}.png'
    r = requests.get(url, timeout=300); r.raise_for_status()
    open(arq, 'wb').write(r.content)
    print('OK', arq, round(len(r.content) / 1e6, 1), 'MB')

print('Centro do recorte (lon, lat):', ponto.coordinates().getInfo())
with zipfile.ZipFile('imagens_apresentacao.zip', 'w') as z:
    for f in sorted(os.listdir('imagens_apresentacao')):
        z.write(f'imagens_apresentacao/{f}', f)
try:
    from google.colab import files
    files.download('imagens_apresentacao.zip')
except Exception:
    print('Baixe imagens_apresentacao.zip pelo painel de arquivos do Colab.')
