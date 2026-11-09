"""
ETAPA 1c: NDVI das lavouras de café com Landsat 5, 7 e 8 (Google Earth Engine)
=============================================================================

O que faz
---------
Para cada sensor, cada safra e cada município, calcula o NDVI só nos pixels
de café e resume o ano agrícola em quatro indicadores:

  * NDVI_Vigor_Maximo ... percentil 90 das imagens de julho de t-1 a junho de t
  * NDVI_Estresse_Maximo  percentil 10 do mesmo período
  * NDVI_Florada ........ média de setembro a novembro de t-1
  * NDVI_Amplitude ...... vigor - estresse (quanto a lavoura perdeu de folha)

Detalhes do processamento
-------------------------
* Reflectância de superfície, Coleção 2, nível 2 (fator 0,0000275 e -0,2).
* Remoção de nuvem, sombra, cirro, neve e pixels saturados (QA_PIXEL, QA_RADSAT).
* Máscara de café do MapBiomas Coleção 9, classe 46, do PRÓPRIO ano da safra
  (aqui a resolução de 30 m permite máscara anual, ao contrário do clima).
* Percentis por pixel e depois média dos pixels do município.

Por que extrair os sensores separadamente
-----------------------------------------
Landsat 5, 7 e 8 têm bandas e calibrações diferentes. Cada sensor é extraído
em TODAS as safras em que tem imagens, para que os anos em que dois sensores
se sobrepõem (L5 x L7 em 2000-2011; L7 x L8 em 2014-2022) sirvam para colocar
a série na mesma escala. Essa calibração é feita na etapa 2b, fora do GEE.

Como rodar
----------
No Google Colab:
    !pip install earthengine-api pandas
    %run etapa1_coleta/1c_ndvi_landsat_gee.py

Saída: ndvi_landsat_por_sensor_1989_2023.csv (já está em dados/brutos/).
"""
import ee
import pandas as pd

MEU_PROJETO = "spatial-garage-267618"
SAIDA = "ndvi_landsat_por_sensor_1989_2023.csv"

try:
    ee.Initialize(project=MEU_PROJETO)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=MEU_PROJETO)

MUNICIPIOS = ["Arceburgo", "Cabo Verde", "Guaranesia", "Guaxupe", "Itamogi",
              "Jacui", "Juruaia", "Monte Belo", "Muzambinho", "Nova Resende"]
gaul = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", "Minas Gerais"))
municipios_fc = gaul.filter(ee.Filter.inList("ADM2_NAME", MUNICIPIOS))
print("Municípios encontrados no GAUL:", municipios_fc.aggregate_array("ADM2_NAME").getInfo())

MAPBIOMAS = ee.Image("projects/mapbiomas-public/assets/brazil/lulc/collection9/"
                     "mapbiomas_collection90_integration_v1")

# sensor: (coleção, banda NIR, banda RED, primeira safra, última safra)
SENSORES = {
    "L5": ("LANDSAT/LT05/C02/T1_L2", "SR_B4", "SR_B3", 1989, 2012),
    "L7": ("LANDSAT/LE07/C02/T1_L2", "SR_B4", "SR_B3", 2000, 2022),
    "L8": ("LANDSAT/LC08/C02/T1_L2", "SR_B5", "SR_B4", 2014, 2023),
}


def ndvi_mascarado(img, nir, red, mascara_cafe):
    qa = img.select("QA_PIXEL")
    limpo = (qa.bitwiseAnd(1 << 1).eq(0)          # dilated cloud
             .And(qa.bitwiseAnd(1 << 2).eq(0))     # cirro
             .And(qa.bitwiseAnd(1 << 3).eq(0))     # nuvem
             .And(qa.bitwiseAnd(1 << 4).eq(0))     # sombra de nuvem
             .And(img.select("QA_RADSAT").eq(0)))  # sem saturação
    refl = img.select([nir, red]).multiply(0.0000275).add(-0.2)
    ndvi = refl.normalizedDifference([nir, red]).rename("NDVI")
    return ee.Image(ndvi.updateMask(limpo).updateMask(mascara_cafe)
                    .copyProperties(img, ["system:time_start"]))


linhas = []
for sensor, (colecao, nir, red, s_ini, s_fim) in SENSORES.items():
    for safra in range(s_ini, s_fim + 1):
        masc = MAPBIOMAS.select(f"classification_{min(safra, 2023)}").eq(46).selfMask()
        base = ee.ImageCollection(colecao).filterBounds(municipios_fc)
        ano_agricola = base.filterDate(f"{safra - 1}-07-01", f"{safra}-07-01")
        florada = base.filterDate(f"{safra - 1}-09-01", f"{safra - 1}-12-01")
        n = ano_agricola.size().getInfo()
        if n == 0:
            print(f"{sensor} safra {safra}: sem imagens")
            continue
        col = ano_agricola.map(lambda i: ndvi_mascarado(i, nir, red, masc))
        flo = florada.map(lambda i: ndvi_mascarado(i, nir, red, masc))
        bandas = [col.reduce(ee.Reducer.percentile([90])).rename("Vigor"),
                  col.reduce(ee.Reducer.percentile([10])).rename("Estresse"),
                  col.count().rename("N_obs")]
        if florada.size().getInfo() > 0:
            bandas.append(flo.mean().rename("Florada"))
        est = ee.Image.cat(bandas).reduceRegions(collection=municipios_fc, reducer=ee.Reducer.mean(),
                                                 scale=30, tileScale=4).getInfo()
        for f in est["features"]:
            p = f["properties"]
            linhas.append({"Ano": safra, "Municipio": p["ADM2_NAME"], "Sensor": sensor,
                           "NDVI_Vigor_Maximo": p.get("Vigor"), "NDVI_Estresse_Maximo": p.get("Estresse"),
                           "NDVI_Florada": p.get("Florada"), "N_obs_medio_pixel": p.get("N_obs"),
                           "N_imagens": n})
        print(f"{sensor} safra {safra}: {n} imagens")

df = pd.DataFrame(linhas)
df["NDVI_Amplitude"] = df["NDVI_Vigor_Maximo"] - df["NDVI_Estresse_Maximo"]
df.to_csv(SAIDA, index=False)
print(df.groupby("Sensor")[["NDVI_Vigor_Maximo", "NDVI_Estresse_Maximo"]].describe().T)
print(f"Salvo: {SAIDA}")
