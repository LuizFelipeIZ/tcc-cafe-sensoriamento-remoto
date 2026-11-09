"""
=====================================================================
TCC - Extração Fenológica de Clima (Tmin/Precipitação) e NDVI Histórico
via Google Earth Engine - Sul de Minas Gerais

Corrige o script original em três pontos:
  1. Extração POR MUNICÍPIO (não um único ponto para toda a microrregião)
  2. Adiciona PRECIPITAÇÃO, não só temperatura
  3. Agrega em JANELAS FENOLÓGICAS (florada, enchimento de grãos, inverno)
     em vez de série diária bruta

Além disso, adiciona a extração de NDVI histórico via Landsat 5/7/8
(1988-2014, antes do Sentinel-2) COM calibração cruzada no período de
sobreposição (2015-2020), para evitar viés de sensor ao unir com a
série moderna de Sentinel-2.

Rode este script no Google Colab (precisa de conta Earth Engine ativa).
=====================================================================
"""

import ee
import pandas as pd
import numpy as np

MEU_PROJETO = 'spatial-garage-267618'  # ajuste para o seu Project ID

try:
    ee.Initialize(project=MEU_PROJETO)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=MEU_PROJETO)

# =====================================================================
# 1. GEOMETRIAS DOS 10 MUNICÍPIOS
# =====================================================================
# Tenta resolver via FAO GAUL (limites administrativos nível 2). Se o
# nome não bater exatamente (acentuação, grafia), cai no fallback de
# centróide fixo (coordenadas aproximadas da sede municipal).
MUNICIPIOS = [
    "Arceburgo", "Cabo Verde", "Guaranesia", "Guaxupe", "Itamogi",
    "Jacui", "Juruaia", "Monte Belo", "Muzambinho", "Nova Resende"
]

# Fallback: coordenadas aproximadas da sede de cada município (lon, lat).
# Usadas apenas se o município não for encontrado no GAUL.
CENTROIDES_FALLBACK = {
    "Arceburgo":    (-46.9611, -21.3781),
    "Cabo Verde":   (-46.3961, -21.4586),
    "Guaranesia":   (-46.9922, -21.3892),
    "Guaxupe":      (-46.7136, -21.3053),
    "Itamogi":      (-46.7758, -21.2914),
    "Jacui":        (-46.6353, -21.0175),
    "Juruaia":      (-46.8983, -21.3494),
    "Monte Belo":   (-46.3689, -21.2861),
    "Muzambinho":   (-46.5236, -21.3739),
    "Nova Resende": (-46.4319, -21.1808),
}

gaul = ee.FeatureCollection("FAO/GAUL/2015/level2") \
    .filter(ee.Filter.eq('ADM1_NAME', 'Minas Gerais'))


def obter_geometria_municipio(nome):
    """Resolve a geometria do município filtrando o GAUL pelo ponto do
    centróide conhecido (filterBounds), em vez de casar pelo nome em texto.
    Mais robusto contra diferenças de acentuação/grafia entre a nossa lista
    e o nome oficial no GAUL - o ponto sempre cai dentro do polígono certo,
    independente de como o nome está escrito."""
    lon, lat = CENTROIDES_FALLBACK[nome]
    ponto = ee.Geometry.Point([lon, lat])
    try:
        feat = gaul.filterBounds(ponto).first()
        geom_info = ee.Feature(feat).geometry().getInfo()
        if geom_info is not None:
            return ee.Feature(feat).geometry(), 'GAUL_POR_PONTO'
    except Exception:
        pass

    print(f"[AVISO] {nome}: GAUL não retornou polígono para o ponto {lon},{lat} - "
          f"usando buffer de 5km no centróide como fallback.")
    return ponto.buffer(5000), 'FALLBACK_BUFFER_5KM'


geometrias = {}
for nome in MUNICIPIOS:
    geom, fonte = obter_geometria_municipio(nome)
    geometrias[nome] = geom
    print(f"[INFO] {nome}: geometria resolvida via {fonte}")


# =====================================================================
# 2. JANELAS FENOLÓGICAS (ajuste os meses conforme sua revisão bibliográfica)
# =====================================================================
JANELAS_FENOLOGICAS = {
    'Florada': [9, 10, 11],                 # set-out-nov
    'Enchimento_Graos': [12, 1, 2, 3, 4],    # dez-abr (expansão/enchimento)
    'Inverno_Risco_Geada': [6, 7, 8],        # jun-jul-ago
}


def construir_pegada_cafe_municipio(nome, geom_municipio, anos_ancora=(2000, 2010, 2015, 2020, 2023)):
    """Constrói a 'pegada estável de café' do município: a união dos pixels
    classificados como café em qualquer um dos anos-âncora informados,
    vetorizada e recortada ao limite municipal.

    Usada como geometria de extração tanto para clima (ERA5-Land) quanto
    para NDVI, em vez do polígono administrativo inteiro - importante
    especialmente em municípios que se estendem por mais de um pixel de
    clima (~11km), onde a área de café tende a se concentrar em encostas,
    não em fundos de vale (Assad et al., 2004).

    SIMPLIFICAÇÃO DECLARADA: usa uma pegada estática (união multi-anos) em
    vez de recalcular a máscara ano a ano para o clima, por custo
    computacional. Isso deve ser mencionado como limitação na Metodologia:
    a extensão exata da lavoura pode ter mudado dentro do período 1988-2023,
    e a pegada estável é uma aproximação da área historicamente cafeeira.
    """
    mascara_uniao = None
    for ano in anos_ancora:
        m = obter_mascara_cafe(ano)
        mascara_uniao = m if mascara_uniao is None else mascara_uniao.unmask(0).Or(m.unmask(0))

    mascara_uniao = mascara_uniao.selfMask().clip(geom_municipio)

    vetores = mascara_uniao.reduceToVectors(
        geometry=geom_municipio, scale=30, geometryType='polygon',
        eightConnected=True, maxPixels=1e9, bestEffort=True
    )
    geometria_cafe = vetores.geometry()

    # Fallback: se a pegada de café ficar vazia (ex.: erro de classificação
    # ou município sem café detectado nos anos-âncora), usa o polígono
    # municipal inteiro para não perder o município da análise.
    area_cafe = geometria_cafe.area(1).getInfo()
    if area_cafe is None or area_cafe < 1000:  # menos de 0.1 ha é ruído
        print(f"[AVISO] Pegada de café vazia/insignificante em {nome} - "
              f"usando o polígono municipal inteiro como fallback.")
        return geom_municipio, False

    return geometria_cafe, True


# =====================================================================
# 3. EXTRAÇÃO DE CLIMA (ERA5-Land) POR MUNICÍPIO E ANO
# =====================================================================
def extrair_clima_municipio(nome, geom, ano_inicio=1988, ano_fim=2024):
    print(f"\n[INFO] Extraindo clima de {nome} ({ano_inicio}-{ano_fim})...")

    colecao = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .select(['temperature_2m_min', 'total_precipitation_sum']) \
        .filterBounds(geom) \
        .filterDate(f'{ano_inicio}-01-01', f'{ano_fim}-01-01')

    def extrair_dia(img):
        data = img.date()
        valores = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=11132,
            bestEffort=True
        )
        tmin_c = ee.Number(valores.get('temperature_2m_min')).subtract(273.15)
        precip_mm = ee.Number(valores.get('total_precipitation_sum')).multiply(1000)  # m -> mm
        return ee.Feature(None, {
            'Ano': data.get('year'),
            'Mes': data.get('month'),
            'Dia': data.get('day'),
            'Tmin_C': tmin_c,
            'Precip_mm': precip_mm,
        })

    # Processa ano a ano para não estourar o limite de elementos do getInfo()
    registros = []
    for ano in range(ano_inicio, ano_fim):
        colecao_ano = colecao.filterDate(f'{ano}-01-01', f'{ano+1}-01-01')
        feats = colecao_ano.map(extrair_dia).getInfo()['features']
        for f in feats:
            p = f['properties']
            if p.get('Tmin_C') is not None:
                registros.append(p)

    df = pd.DataFrame(registros)
    df['Municipio'] = nome
    return df


def agregar_indicadores_anuais(df_diario):
    """Converte a série diária em indicadores anuais e fenológicos."""
    linhas = []
    for (mun, ano), grupo in df_diario.groupby(['Municipio', 'Ano']):
        registro = {
            'Municipio': mun,
            'Ano': ano,
            'Tmin_Abs': grupo['Tmin_C'].min(),
            'Precip_Anual': grupo['Precip_mm'].sum(),
            'Dias_Abaixo_3C': (grupo[grupo['Mes'].isin(JANELAS_FENOLOGICAS['Inverno_Risco_Geada'])]['Tmin_C'] < 3).sum(),
        }
        for janela, meses in JANELAS_FENOLOGICAS.items():
            sub = grupo[grupo['Mes'].isin(meses)]
            registro[f'Precip_{janela}'] = sub['Precip_mm'].sum()
            registro[f'Tmin_Abs_{janela}'] = sub['Tmin_C'].min() if not sub.empty else np.nan
        linhas.append(registro)
    return pd.DataFrame(linhas)


# =====================================================================
# 4. MÁSCARA DE CAFÉ (MapBiomas) - restringe a extração de NDVI aos
#    pixels efetivamente classificados como lavoura cafeeira, eliminando
#    ruído espectral de pastagem, mata ciliar e vegetação nativa
#    adjacente às lavouras (o mesmo princípio da máscara EPAMIG proposta
#    originalmente na Metodologia, mas usando uma fonte nativa do GEE,
#    disponível também para os anos históricos de 1985 em diante).
# =====================================================================
MAPBIOMAS_ASSET = 'projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1'
CODIGO_CLASSE_CAFE = 46  # "Café" na legenda MapBiomas (Coleções 7-10)
_MAPBIOMAS_IMAGEM = None  # cache: a imagem "larga" (uma banda por ano) só precisa ser carregada 1x


def _obter_imagem_mapbiomas():
    global _MAPBIOMAS_IMAGEM
    if _MAPBIOMAS_IMAGEM is None:
        _MAPBIOMAS_IMAGEM = ee.Image(MAPBIOMAS_ASSET)
    return _MAPBIOMAS_IMAGEM


def obter_mascara_cafe(ano):
    """Retorna uma ee.Image binária (1 = café, mascarado nos demais pixels)
    para o ano especificado, selecionando a banda 'classification_AAAA'
    correspondente na imagem larga do MapBiomas (Coleção 9). Anos fora do
    intervalo coberto usam o ano válido mais próximo."""
    ano_valido = min(max(ano, 1985), 2023)
    imagem = _obter_imagem_mapbiomas()
    classificacao = imagem.select(f'classification_{ano_valido}')
    mascara_cafe = classificacao.eq(CODIGO_CLASSE_CAFE).selfMask()
    return mascara_cafe


def calcular_area_cafe_municipio(nome, geom, ano):
    """Calcula a área (ha) classificada como café dentro do município,
    no ano especificado - útil como métrica complementar/validação
    cruzada contra a área colhida oficial do SIDRA."""
    mascara = obter_mascara_cafe(ano)
    area_img = mascara.multiply(ee.Image.pixelArea()).divide(10000)  # m² -> ha
    area_ha = area_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom, scale=30, bestEffort=True, maxPixels=1e9
    ).get('classification')
    valor = area_ha.getInfo() if area_ha is not None else 0
    return valor if valor is not None else 0


def exportar_mascara_cafe_microrregiao(geometrias_dict, ano_referencia=2023,
                                        nome_arquivo='TCC_Mascara_Cafe_MapBiomas_SSP'):
    """Exporta a máscara de café (vetorizada) da microrregião inteira para
    o Google Drive, em formato SHP, para inspeção visual em QGIS/ArcGIS.
    Roda de forma assíncrona - acompanhe o progresso na aba 'Tasks' do
    editor de código do Earth Engine (code.earthengine.google.com/tasks)."""
    microrregiao = ee.FeatureCollection(list(geometrias_dict.values())).geometry().dissolve()
    mascara = obter_mascara_cafe(ano_referencia).clip(microrregiao)

    vetores = mascara.reduceToVectors(
        geometry=microrregiao,
        scale=30,
        geometryType='polygon',
        eightConnected=True,
        maxPixels=1e10,
        bestEffort=True
    )

    tarefa = ee.batch.Export.table.toDrive(
        collection=vetores,
        description=nome_arquivo,
        fileFormat='SHP'
    )
    tarefa.start()
    print(f"[INFO] Exportação da máscara de café ({ano_referencia}) iniciada.")
    print("       Acompanhe em: https://code.earthengine.google.com/tasks")
    print(f"       Arquivo aparecerá no seu Google Drive como '{nome_arquivo}.shp' (+ auxiliares).")
    return tarefa


# =====================================================================
# 5. NDVI HISTÓRICO VIA LANDSAT (1988-2014) COM CALIBRAÇÃO CRUZADA
#    (Agora restrito à máscara de café MapBiomas do respectivo ano)
# =====================================================================
def mascara_nuvem_landsat_c2(img):
    qa = img.select('QA_PIXEL')
    mascara_nuvem = qa.bitwiseAnd(1 << 3).eq(0)  # bit 3 = nuvem
    mascara_sombra = qa.bitwiseAnd(1 << 4).eq(0)  # bit 4 = sombra de nuvem
    return img.updateMask(mascara_nuvem).updateMask(mascara_sombra)


def calcular_ndvi_landsat(img, banda_nir, banda_red):
    # Fator de escala do Collection 2, Level 2 (reflectância de superfície)
    nir = img.select(banda_nir).multiply(0.0000275).add(-0.2)
    red = img.select(banda_red).multiply(0.0000275).add(-0.2)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    return img.addBands(ndvi)


def obter_colecao_landsat_ndvi(ano_inicio, ano_fim):
    """Combina Landsat 5, 7 e 8 em uma única coleção de NDVI, incluindo
    apenas os sensores cuja janela de cobertura realmente se sobrepõe ao
    intervalo [ano_inicio, ano_fim] solicitado - evita intervalos de data
    invertidos (ex.: pedir 2015-2021 e tentar filtrar o Landsat 5, que só
    cobre até 2012, gerando um range com fim antes do início)."""

    colecoes = []

    # Landsat 5: cobertura útil até ~2012
    if ano_inicio <= 2012 and ano_fim >= 1988:
        ini, fim = max(ano_inicio, 1988), min(ano_fim, 2012)
        if ini <= fim:
            l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2") \
                .filterDate(f'{ini}-01-01', f'{fim + 1}-01-01') \
                .map(mascara_nuvem_landsat_c2) \
                .map(lambda img: calcular_ndvi_landsat(img, 'SR_B4', 'SR_B3'))
            colecoes.append(l5)

    # Landsat 7: cobertura a partir de 1999 (SLC-off após 2003, mas ainda utilizável)
    if ano_fim >= 1999:
        ini, fim = max(ano_inicio, 1999), ano_fim
        if ini <= fim:
            l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2") \
                .filterDate(f'{ini}-01-01', f'{fim + 1}-01-01') \
                .map(mascara_nuvem_landsat_c2) \
                .map(lambda img: calcular_ndvi_landsat(img, 'SR_B4', 'SR_B3'))
            colecoes.append(l7)

    # Landsat 8: cobertura a partir de 2013
    if ano_fim >= 2013:
        ini, fim = max(ano_inicio, 2013), ano_fim
        if ini <= fim:
            l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                .filterDate(f'{ini}-01-01', f'{fim + 1}-01-01') \
                .map(mascara_nuvem_landsat_c2) \
                .map(lambda img: calcular_ndvi_landsat(img, 'SR_B5', 'SR_B4'))
            colecoes.append(l8)

    if not colecoes:
        raise ValueError(f"Nenhum sensor Landsat cobre o intervalo {ano_inicio}-{ano_fim}.")

    colecao_final = colecoes[0]
    for c in colecoes[1:]:
        colecao_final = colecao_final.merge(c)
    return colecao_final


def extrair_ndvi_landsat_municipio(nome, geom, ano_inicio=1988, ano_fim=2015, usar_mascara_cafe=True):
    """Extrai o NDVI médio do Landsat em CADA janela fenológica (Florada,
    Enchimento_Graos, Inverno_Risco_Geada), não apenas no inverno - espelha
    a mesma lógica de JANELAS_FENOLOGICAS usada na extração de clima."""
    print(f"[INFO] Extraindo NDVI histórico (Landsat) de {nome}, todas as fases fenológicas...")
    colecao = obter_colecao_landsat_ndvi(ano_inicio, ano_fim).select('NDVI').filterBounds(geom)

    registros = []
    for ano in range(ano_inicio, ano_fim):
        registro = {'Municipio': nome, 'Ano': ano}
        algum_dado = False

        for nome_janela, meses in JANELAS_FENOLOGICAS.items():
            # Janela de Enchimento_Graos cruza o ano (dez do ano corrente até
            # abr do ano seguinte) - trata a data de início/fim corretamente
            if nome_janela == 'Enchimento_Graos':
                data_ini = f'{ano}-12-01'
                data_fim = f'{ano + 1}-05-01'
            else:
                mes_ini, mes_fim = min(meses), max(meses)
                data_ini = f'{ano}-{mes_ini:02d}-01'
                data_fim = f'{ano}-{mes_fim:02d}-{"28" if mes_fim==2 else "30"}'
                # usa o primeiro dia do mês seguinte para não cortar o último dia
                mes_fim_seguinte = mes_fim + 1 if mes_fim < 12 else 1
                ano_fim_seguinte = ano if mes_fim < 12 else ano + 1
                data_fim = f'{ano_fim_seguinte}-{mes_fim_seguinte:02d}-01'

            colecao_janela = colecao.filterDate(data_ini, data_fim)
            n_imagens = colecao_janela.size().getInfo()
            if n_imagens == 0:
                registro[f'NDVI_{nome_janela}_Landsat'] = None
                registro[f'N_Imagens_{nome_janela}'] = 0
                continue

            img_media = colecao_janela.mean()
            if usar_mascara_cafe:
                img_media = img_media.updateMask(obter_mascara_cafe(ano))
            media = img_media.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=geom, scale=30, bestEffort=True
            ).get('NDVI').getInfo()

            registro[f'NDVI_{nome_janela}_Landsat'] = media
            registro[f'N_Imagens_{nome_janela}'] = n_imagens
            if media is not None:
                algum_dado = True

        if algum_dado:
            registros.append(registro)

    return pd.DataFrame(registros)


def extrair_ndvi_sentinel_municipio(nome, geom, ano_inicio=2015, ano_fim=2024, usar_mascara_cafe=True):
    """Réplica em Sentinel-2 para todas as janelas fenológicas, usada tanto
    para o modelo moderno quanto para calcular o fator de calibração
    cruzada com o Landsat no período de sobreposição."""
    print(f"[INFO] Extraindo NDVI Sentinel-2 de {nome}, todas as fases fenológicas...")

    def mascara_nuvem_s2(img):
        qa = img.select('QA60')
        mascara = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(mascara)

    colecao = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(geom) \
        .map(mascara_nuvem_s2) \
        .map(lambda img: img.addBands(
            img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        )).select('NDVI')

    registros = []
    for ano in range(ano_inicio, ano_fim):
        registro = {'Municipio': nome, 'Ano': ano}
        algum_dado = False

        for nome_janela, meses in JANELAS_FENOLOGICAS.items():
            if nome_janela == 'Enchimento_Graos':
                data_ini = f'{ano}-12-01'
                data_fim = f'{ano + 1}-05-01'
            else:
                mes_ini, mes_fim = min(meses), max(meses)
                data_ini = f'{ano}-{mes_ini:02d}-01'
                mes_fim_seguinte = mes_fim + 1 if mes_fim < 12 else 1
                ano_fim_seguinte = ano if mes_fim < 12 else ano + 1
                data_fim = f'{ano_fim_seguinte}-{mes_fim_seguinte:02d}-01'

            colecao_janela = colecao.filterDate(data_ini, data_fim)
            n_imagens = colecao_janela.size().getInfo()
            if n_imagens == 0:
                registro[f'NDVI_{nome_janela}_Sentinel'] = None
                registro[f'N_Imagens_{nome_janela}'] = 0
                continue

            img_media = colecao_janela.mean()
            if usar_mascara_cafe:
                img_media = img_media.updateMask(obter_mascara_cafe(ano))
            media = img_media.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=geom, scale=10, bestEffort=True
            ).get('NDVI').getInfo()

            registro[f'NDVI_{nome_janela}_Sentinel'] = media
            registro[f'N_Imagens_{nome_janela}'] = n_imagens
            if media is not None:
                algum_dado = True

        if algum_dado:
            registros.append(registro)

    return pd.DataFrame(registros)


def calcular_fator_calibracao(df_landsat_overlap, df_sentinel_overlap, janelas):
    """Calcula o fator de calibração linear (slope, intercept) entre Landsat
    e Sentinel-2 no período de sobreposição, SEPARADAMENTE para cada janela
    fenológica - sensores podem ter viés diferente conforme a época do ano
    (cobertura de nuvem varia entre florada, enchimento e inverno)."""
    from scipy.stats import linregress
    fatores = {}
    for janela in janelas:
        col_l = f'NDVI_{janela}_Landsat'
        col_s = f'NDVI_{janela}_Sentinel'
        merged = df_landsat_overlap[['Municipio', 'Ano', col_l]].merge(
            df_sentinel_overlap[['Municipio', 'Ano', col_s]], on=['Municipio', 'Ano']
        ).dropna()

        if len(merged) < 5:
            print(f"[AVISO] Sobreposição insuficiente para calibrar '{janela}' (N={len(merged)}). "
                  f"Usando offset=0, slope=1 (sem correção) - trate com cautela na Metodologia.")
            fatores[janela] = (0.0, 1.0)
            continue

        slope, intercept, r_value, p_value, std_err = linregress(merged[col_l], merged[col_s])
        print(f"[OK] Calibração '{janela}': slope={slope:.4f}, intercept={intercept:.4f}, "
              f"R²={r_value**2:.3f} (N={len(merged)})")
        fatores[janela] = (intercept, slope)

    return fatores


# =====================================================================
# EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    # --- -1. Construir a pegada estável de café por município ---
    print("\n[INFO] Construindo pegadas de café (MapBiomas, união de anos-âncora)...")
    geometrias_cafe = {}
    for nome in MUNICIPIOS:
        geom_cafe, encontrou_cafe = construir_pegada_cafe_municipio(nome, geometrias[nome])
        geometrias_cafe[nome] = geom_cafe
        status = "pegada de café" if encontrou_cafe else "fallback (município inteiro)"
        print(f"  {nome}: usando {status}")

    # --- 0. Exportar a máscara de café vetorizada para inspeção (QGIS) ---
    exportar_mascara_cafe_microrregiao(geometrias, ano_referencia=2023)

    # --- 0b. Área de café por município (validação cruzada com SIDRA) ---
    print("\n[INFO] Calculando área de café (MapBiomas) por município, anos-âncora...")
    registros_area = []
    for nome in MUNICIPIOS:
        for ano in [1994, 2000, 2013, 2018, 2023]:
            area_ha = calcular_area_cafe_municipio(nome, geometrias[nome], ano)
            registros_area.append({'Municipio': nome, 'Ano': ano, 'Area_Cafe_MapBiomas_ha': area_ha})
    pd.DataFrame(registros_area).to_csv('TCC_Area_Cafe_MapBiomas_Validacao.csv', index=False)
    print("[OK] Área de café por município salva em TCC_Area_Cafe_MapBiomas_Validacao.csv "
          "- compare com Area_Colhida_ha do SIDRA como checagem de consistência.")

    # --- 1. Clima fenológico por município, restrito à pegada de café ---
    todos_climas = []
    for nome in MUNICIPIOS:
        df_diario = extrair_clima_municipio(nome, geometrias_cafe[nome], 1988, 2024)
        todos_climas.append(df_diario)
    df_clima_diario = pd.concat(todos_climas, ignore_index=True)
    df_clima_anual = agregar_indicadores_anuais(df_clima_diario)
    df_clima_anual.to_csv('TCC_Clima_Fenologico_1988_2024.csv', index=False)
    print(f"\n[SUCESSO] Clima fenológico salvo: TCC_Clima_Fenologico_1988_2024.csv "
          f"({len(df_clima_anual)} linhas)")

    # --- 2. NDVI histórico (Landsat) e período de sobreposição (Sentinel-2) ---
    todos_ndvi_landsat = []
    todos_ndvi_landsat_overlap = []
    todos_ndvi_sentinel_overlap = []
    for nome in MUNICIPIOS:
        geom = geometrias_cafe[nome]
        df_l_hist = extrair_ndvi_landsat_municipio(nome, geom, 1988, 2015)
        todos_ndvi_landsat.append(df_l_hist)

        df_l_overlap = extrair_ndvi_landsat_municipio(nome, geom, 2015, 2021)
        todos_ndvi_landsat_overlap.append(df_l_overlap)

        df_s_overlap = extrair_ndvi_sentinel_municipio(nome, geom, 2015, 2021)
        todos_ndvi_sentinel_overlap.append(df_s_overlap)

    df_ndvi_landsat_hist = pd.concat(todos_ndvi_landsat, ignore_index=True)
    df_ndvi_landsat_overlap = pd.concat(todos_ndvi_landsat_overlap, ignore_index=True)
    df_ndvi_sentinel_overlap = pd.concat(todos_ndvi_sentinel_overlap, ignore_index=True)

    # --- 3. Calibração cruzada e correção do NDVI histórico, por janela fenológica ---
    fatores = calcular_fator_calibracao(
        df_ndvi_landsat_overlap, df_ndvi_sentinel_overlap, JANELAS_FENOLOGICAS.keys()
    )
    for janela, (intercept, slope) in fatores.items():
        col_bruta = f'NDVI_{janela}_Landsat'
        col_calibrada = f'NDVI_{janela}_Calibrado'
        df_ndvi_landsat_hist[col_calibrada] = df_ndvi_landsat_hist[col_bruta] * slope + intercept

    df_ndvi_landsat_hist.to_csv('TCC_NDVI_Historico_Landsat_Calibrado_1988_2014.csv', index=False)
    print(f"\n[SUCESSO] NDVI histórico calibrado (todas as fases fenológicas) salvo: "
          f"TCC_NDVI_Historico_Landsat_Calibrado_1988_2014.csv "
          f"({len(df_ndvi_landsat_hist)} linhas)")

    print("\n[PRÓXIMO PASSO] Mescle TCC_Clima_Fenologico_1988_2024.csv com o NDVI")
    print("histórico calibrado (para o período 1988-2014) e com a matriz moderna")
    print("já existente (2013-2023, NDVI Sentinel-2) para formar a matriz unificada.")
    print("Use as colunas NDVI_<Fase>_Calibrado no lugar do NDVI Landsat bruto para")
    print("evitar o viés de sensor discutido na Metodologia.")
