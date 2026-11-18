"""
ETAPA 1b2: janela de enchimento corrigida, de dezembro de t-1 a abril de t
==========================================================================

Por que existe
--------------
Na etapa 1b, o enchimento de grãos foi resumido no ANO CIVIL: janeiro a abril
e dezembro do mesmo ano. Mas a colheita da safra t termina em agosto de t, e
o dezembro de t já pertence ao enchimento da safra t+1. A janela certa da
safra t é de dezembro de t-1 a abril de t (revisão do orientador).

Este script recalcula SÓ as duas variáveis de enchimento, com a mesma área
(pegada de café do MapBiomas dentro do município, como na etapa 1b) e o mesmo
ERA5-Land diário, mas somando a chuva e tirando a mínima absoluta de
1º de dezembro de t-1 a 30 de abril de t. A soma e o mínimo são feitos no
próprio Earth Engine, então leva minutos, e não horas.

Como rodar (Google Colab)
-------------------------
    !pip install earthengine-api pandas
    # troque MEU_PROJETO pelo ID do seu projeto no Earth Engine, se preciso
    %run etapa1_coleta/1b2_enchimento_dez_abr_gee.py

Saída: clima_enchimento_dez_abr_1988_2023.csv (360 linhas), que vai para
dados/brutos/. A etapa 2a usa esse arquivo no lugar das colunas de
enchimento antigas, se ele existir.
"""
import os
import time
import unicodedata

import ee
import pandas as pd

MEU_PROJETO = "spatial-garage-267618"   # ID do projeto no Earth Engine
SAFRA_INICIO, SAFRA_FIM = 1988, 2023
SAIDA = "clima_enchimento_dez_abr_1988_2023.csv"

try:
    ee.Initialize(project=MEU_PROJETO)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=MEU_PROJETO)

# --- mesma geometria da etapa 1b -------------------------------------------
# Um ponto bem dentro de cada município (polo de inacessibilidade do polígono da Malha
# Municipal do IBGE, a 5-9 km da divisa), usado para achar o polígono no GAUL. As
# coordenadas antigas, "da sede", caíam em municípios vizinhos em quatro casos
# (Guaranésia, Itamogi, Juruaia e Nova Resende); por isso o nome do polígono achado
# agora é conferido (ver geometria_municipio).
PONTOS_INTERNOS = {   # (longitude, latitude)
    "Arceburgo": (-46.9460, -21.3723), "Cabo Verde": (-46.3969, -21.4668),
    "Guaranesia": (-46.8068, -21.2706), "Guaxupe": (-46.6752, -21.3024),
    "Itamogi": (-47.0405, -21.0673), "Jacui": (-46.7261, -21.0363),
    "Juruaia": (-46.5268, -21.2033), "Monte Belo": (-46.3269, -21.3044),
    "Muzambinho": (-46.5216, -21.3678), "Nova Resende": (-46.4012, -21.1101),
}
gaul = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", "Minas Gerais"))
MAPBIOMAS = ee.Image("projects/mapbiomas-public/assets/brazil/lulc/collection9/"
                     "mapbiomas_collection90_integration_v1")
CLASSE_CAFE = 46


def _sem_acento(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def geometria_municipio(nome):
    """Polígono do GAUL que contém o ponto interno do município, com o nome conferido."""
    lon, lat = PONTOS_INTERNOS[nome]
    feat = ee.Feature(gaul.filterBounds(ee.Geometry.Point([lon, lat])).first())
    achado = feat.get("ADM2_NAME").getInfo()
    if _sem_acento(achado) != _sem_acento(nome):
        raise SystemExit(f"[ERRO] o ponto de {nome} caiu em {achado} no GAUL; corrija PONTOS_INTERNOS")
    print(f"{nome}: polígono GAUL '{achado}'")
    return feat.geometry()


def mascara_cafe(ano):
    return MAPBIOMAS.select(f"classification_{ano}").eq(CLASSE_CAFE).selfMask()


def pegada_cafe(nome, geom_municipio, anos_ancora=(2000, 2010, 2015, 2020, 2023)):
    """União do café nos anos-âncora, vetorizada e recortada ao município (igual à etapa 1b)."""
    uniao = None
    for ano in anos_ancora:
        m = mascara_cafe(ano)
        uniao = m if uniao is None else uniao.unmask(0).Or(m.unmask(0))
    uniao = uniao.selfMask().clip(geom_municipio)
    vetores = uniao.reduceToVectors(geometry=geom_municipio, scale=30, geometryType="polygon",
                                    eightConnected=True, maxPixels=1e9, bestEffort=True)
    geom = vetores.geometry()
    if (geom.area(1).getInfo() or 0) < 1000:
        print(f"[AVISO] {nome}: sem café detectado; usando o município inteiro")
        return geom_municipio
    return geom


ERA5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select(["temperature_2m_min", "total_precipitation_sum"])


def janela_enchimento(geom, safra):
    """Chuva acumulada (mm) e mínima absoluta (°C) de 1/dez de t-1 a 30/abr de t, na média da pegada."""
    dias = ERA5.filterDate(f"{safra - 1}-12-01", f"{safra}-05-01")

    def por_dia(img):
        v = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=11132, bestEffort=True)
        return ee.Feature(None, {
            "tmin": ee.Number(v.get("temperature_2m_min")).subtract(273.15),
            "chuva": ee.Number(v.get("total_precipitation_sum")).multiply(1000),
        })

    fc = dias.map(por_dia).filter(ee.Filter.notNull(["tmin"]))
    return ee.Dictionary({"Precip": fc.aggregate_sum("chuva"), "Tmin": fc.aggregate_min("tmin"),
                          "dias": fc.size()})


if __name__ == "__main__":
    feitos = pd.read_csv(SAIDA) if os.path.exists(SAIDA) else pd.DataFrame()
    linhas = feitos.to_dict("records")
    ja = {(r["Municipio"], r["Ano"]) for r in linhas}
    for nome in PONTOS_INTERNOS:
        geom = pegada_cafe(nome, geometria_municipio(nome))
        for safra in range(SAFRA_INICIO, SAFRA_FIM + 1):
            if (nome, safra) in ja:
                continue
            for tentativa in range(4):   # o Earth Engine às vezes recusa por excesso de pedidos
                try:
                    r = janela_enchimento(geom, safra).getInfo()
                    break
                except Exception as erro:
                    print(f"  {nome} {safra}: tentativa {tentativa + 1} falhou ({erro}); esperando")
                    time.sleep(10 * (tentativa + 1))
            else:
                raise SystemExit(f"Parou em {nome} {safra}. Rode de novo: o script continua de onde parou.")
            linhas.append({"Municipio": nome, "Ano": safra, "Precip_Enchimento_Graos": r["Precip"],
                           "Tmin_Abs_Enchimento_Graos": r["Tmin"], "Dias": r["dias"]})
            pd.DataFrame(linhas).to_csv(SAIDA, index=False)   # salva a cada safra: dá para retomar
            print(f"  {nome} {safra}: {r['dias']} dias, {r['Precip']:.0f} mm, mínima {r['Tmin']:.1f} °C")
    tabela = pd.DataFrame(linhas).sort_values(["Municipio", "Ano"])
    assert len(tabela) == 10 * (SAFRA_FIM - SAFRA_INICIO + 1)
    assert tabela.Dias.between(150, 152).all(), "alguma janela ficou incompleta"
    tabela.to_csv(SAIDA, index=False)
    print(f"Salvo: {SAIDA} ({len(tabela)} linhas). Envie este arquivo.")
