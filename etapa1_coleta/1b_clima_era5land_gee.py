"""
ETAPA 1b: clima diário do ERA5-Land sobre a área de café (Google Earth Engine)
==============================================================================

O que faz
---------
1. Pega o limite de cada município (FAO GAUL, nível 2).
2. Monta a "pegada estável de café" de cada município: a união dos pixels de
   café (classe 46 do MapBiomas, Coleção 9) em cinco anos-âncora
   (2000, 2010, 2015, 2020 e 2023). A grade do ERA5-Land tem ~11 km, então
   uma máscara anual não mudaria o pixel de clima; a união evita que vales
   e matas sem café entrem na média.
3. Extrai, dia a dia de 1988 a 2023, a temperatura mínima (°C) e a chuva (mm)
   do ERA5-Land (ECMWF/ERA5_LAND/DAILY_AGGR), na média da pegada de café.
4. Resume cada ano civil em janelas fenológicas:
     * Florada ............ setembro a novembro
     * Enchimento de grãos  dezembro e janeiro a abril (do mesmo ano civil)
     * Inverno (geada) .... junho a agosto
   com a chuva acumulada e a temperatura mínima absoluta de cada janela,
   além dos dias com mínima abaixo de 3 °C no inverno.

As janelas são do ANO CIVIL. A defasagem para o ano agrícola (florada e
inverno do ano t-1 explicando a safra t) é feita depois, na etapa 3.

Como rodar
----------
No Google Colab, com uma conta do Earth Engine:
    !pip install earthengine-api pandas
    # troque MEU_PROJETO pelo ID do seu projeto no Earth Engine
    %run etapa1_coleta/1b_clima_era5land_gee.py

Leva algumas horas (10 municípios x 36 anos x 365 dias).
Saída: clima_era5land_janelas_1988_2023.csv (já está em dados/brutos/).
"""
import ee
import numpy as np
import pandas as pd

MEU_PROJETO = "spatial-garage-267618"   # ID do projeto no Earth Engine
ANO_INICIO, ANO_FIM = 1988, 2023
SAIDA = "clima_era5land_janelas_1988_2023.csv"

try:
    ee.Initialize(project=MEU_PROJETO)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=MEU_PROJETO)

# ---------------------------------------------------------------------------
# 1. Geometria dos municípios
# ---------------------------------------------------------------------------
# O GAUL é filtrado por um ponto dentro de cada município, e não pelo nome,
# para não depender de acentos e grafias.
SEDES = {   # (longitude, latitude) aproximadas da sede
    "Arceburgo": (-46.9611, -21.3781), "Cabo Verde": (-46.3961, -21.4586),
    "Guaranesia": (-46.9922, -21.3892), "Guaxupe": (-46.7136, -21.3053),
    "Itamogi": (-46.7758, -21.2914), "Jacui": (-46.6353, -21.0175),
    "Juruaia": (-46.8983, -21.3494), "Monte Belo": (-46.3689, -21.2861),
    "Muzambinho": (-46.5236, -21.3739), "Nova Resende": (-46.4319, -21.1808),
}
gaul = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", "Minas Gerais"))


def geometria_municipio(nome):
    lon, lat = SEDES[nome]
    ponto = ee.Geometry.Point([lon, lat])
    feat = gaul.filterBounds(ponto).first()
    if ee.Feature(feat).geometry().getInfo() is not None:
        return ee.Feature(feat).geometry()
    print(f"[AVISO] {nome}: GAUL sem polígono; usando raio de 5 km na sede")
    return ponto.buffer(5000)


# ---------------------------------------------------------------------------
# 2. Máscara de café (MapBiomas Coleção 9, classe 46)
# ---------------------------------------------------------------------------
MAPBIOMAS = ee.Image("projects/mapbiomas-public/assets/brazil/lulc/collection9/"
                     "mapbiomas_collection90_integration_v1")
CLASSE_CAFE = 46


def mascara_cafe(ano):
    ano = min(max(ano, 1985), 2023)
    return MAPBIOMAS.select(f"classification_{ano}").eq(CLASSE_CAFE).selfMask()


def pegada_cafe(nome, geom_municipio, anos_ancora=(2000, 2010, 2015, 2020, 2023)):
    """União do café nos anos-âncora, vetorizada e recortada ao município."""
    uniao = None
    for ano in anos_ancora:
        m = mascara_cafe(ano)
        uniao = m if uniao is None else uniao.unmask(0).Or(m.unmask(0))
    uniao = uniao.selfMask().clip(geom_municipio)
    vetores = uniao.reduceToVectors(geometry=geom_municipio, scale=30, geometryType="polygon",
                                    eightConnected=True, maxPixels=1e9, bestEffort=True)
    geom = vetores.geometry()
    if (geom.area(1).getInfo() or 0) < 1000:   # menos de 0,1 ha: ruído
        print(f"[AVISO] {nome}: sem café detectado; usando o município inteiro")
        return geom_municipio
    return geom


# ---------------------------------------------------------------------------
# 3. Série diária do ERA5-Land na pegada de café
# ---------------------------------------------------------------------------
def clima_diario(nome, geom):
    colecao = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
               .select(["temperature_2m_min", "total_precipitation_sum"])
               .filterBounds(geom))

    def por_dia(img):
        v = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=11132, bestEffort=True)
        d = img.date()
        return ee.Feature(None, {
            "Ano": d.get("year"), "Mes": d.get("month"), "Dia": d.get("day"),
            "Tmin_C": ee.Number(v.get("temperature_2m_min")).subtract(273.15),   # K -> °C
            "Precip_mm": ee.Number(v.get("total_precipitation_sum")).multiply(1000),  # m -> mm
        })

    registros = []
    for ano in range(ANO_INICIO, ANO_FIM + 1):   # ano a ano: limite de 5000 elementos do getInfo
        feats = colecao.filterDate(f"{ano}-01-01", f"{ano + 1}-01-01").map(por_dia).getInfo()["features"]
        registros += [f["properties"] for f in feats if f["properties"].get("Tmin_C") is not None]
        print(f"  {nome} {ano}: {len(feats)} dias")
    df = pd.DataFrame(registros)
    df["Municipio"] = nome
    return df


# ---------------------------------------------------------------------------
# 4. Janelas fenológicas por ano civil
# ---------------------------------------------------------------------------
JANELAS = {
    "Florada": [9, 10, 11],
    "Enchimento_Graos": [12, 1, 2, 3, 4],
    "Inverno_Risco_Geada": [6, 7, 8],
}


def resumir_janelas(diario):
    linhas = []
    for (mun, ano), g in diario.groupby(["Municipio", "Ano"]):
        reg = {
            "Municipio": mun, "Ano": ano,
            "Tmin_Abs": g.Tmin_C.min(),
            "Precip_Anual": g.Precip_mm.sum(),
            "Dias_Abaixo_3C": int((g[g.Mes.isin(JANELAS["Inverno_Risco_Geada"])].Tmin_C < 3).sum()),
        }
        for janela, meses in JANELAS.items():
            sub = g[g.Mes.isin(meses)]
            reg[f"Precip_{janela}"] = sub.Precip_mm.sum()
            reg[f"Tmin_Abs_{janela}"] = sub.Tmin_C.min() if len(sub) else np.nan
        linhas.append(reg)
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    diarios = []
    for nome in SEDES:
        geom = pegada_cafe(nome, geometria_municipio(nome))
        diarios.append(clima_diario(nome, geom))
    tabela = resumir_janelas(pd.concat(diarios, ignore_index=True))
    tabela.to_csv(SAIDA, index=False)
    print(f"Salvo: {SAIDA} ({len(tabela)} linhas)")
