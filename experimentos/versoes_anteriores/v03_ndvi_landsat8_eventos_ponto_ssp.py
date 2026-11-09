!pip install earthengine-api pandas matplotlib seaborn --quiet

from google.colab import auth
import ee
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

print("🔐 Autenticando...")
auth.authenticate_user()

print("🌍 Conectando ao Earth Engine...")
# ATENÇÃO: Coloque o ID do seu projeto do GEE aqui dentro das aspas!
ee.Initialize(project='spatial-garage-267618')

# 1. Definir a Área de Estudo (A SOLUÇÃO INFALÍVEL)
# Usamos um ponto central e filtramos o polígono oficial que "encosta" nesse ponto.
ponto_ssp = ee.Geometry.Point([-46.991, -20.917])
municipio = ee.FeatureCollection("FAO/GAUL/2015/level2").filterBounds(ponto_ssp)
roi = municipio.geometry()

# 2. Carregar o MapBiomas (Coleção 9.0 Atualizada)
mapbiomas = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1')

# Criar máscaras dinâmicas de Café (Classe 46) para os anos exatos
mascara_cafe_2014 = mapbiomas.select('classification_2014').eq(46)
mascara_cafe_2021 = mapbiomas.select('classification_2021').eq(46)

# 3. Função para processar o Landsat 8 (Nuvem + NDVI + Máscara)
def processar_landsat(image, mascara):
    qa = image.select('QA_PIXEL')
    cloud_shadow_bit_mask = (1 << 3)
    clouds_bit_mask = (1 << 4)
    mask_nuvem = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0) \
                 .And(qa.bitwiseAnd(clouds_bit_mask).eq(0))

    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    mask_final = mask_nuvem.And(mascara)

    return image.updateMask(mask_final).addBands(ndvi).copyProperties(image, ["system:time_start"])

# 4. Configurar as duas janelas de extração
janelas = {
    'Seca (2014-2016)': {'inicio': '2014-01-01', 'fim': '2016-12-31', 'mascara': mascara_cafe_2014},
    'Geada (2021-2023)': {'inicio': '2021-01-01', 'fim': '2023-12-31', 'mascara': mascara_cafe_2021}
}

resultados_finais = []

print("🛰️ Iniciando varredura do Landsat 8. Aguarde...\n")

for nome_evento, params in janelas.items():
    colecao = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
               .filterBounds(roi)
               .filterDate(params['inicio'], params['fim'])
               .filter(ee.Filter.lt('CLOUD_COVER', 30))
               .map(lambda img: processar_landsat(img, params['mascara'])))

    # Aviso no console para monitorar o andamento
    tamanho = colecao.size().getInfo()
    print(f"📡 {nome_evento}: {tamanho} imagens orbitais encontradas e processadas.")

    if tamanho == 0:
        continue

    def extrair_media(image):
        media = image.select('NDVI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=30, # Resolução nativa do Landsat e do MapBiomas
            maxPixels=1e9
        )
        return ee.Feature(None, {
            'Data': image.date().format('YYYY-MM-dd'),
            'NDVI': media.get('NDVI'),
            'Evento': nome_evento
        })

    serie = colecao.map(extrair_media).getInfo()
    dados = [feat['properties'] for feat in serie['features'] if feat['properties']['NDVI'] is not None]
    resultados_finais.extend(dados)

# 5. Organizar os dados e gerar gráficos
if not resultados_finais:
    print("\n❌ ERRO: Nenhum dado de café foi encontrado. O limite da cidade pode não ter cruzado com os dados do MapBiomas.")
else:
    print("\n📊 Desenhando os gráficos de resiliência fisiológica...")
    df = pd.DataFrame(resultados_finais)
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data')

    df_mensal = df.groupby(['Evento', pd.Grouper(key='Data', freq='ME')])['NDVI'].mean().reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharey=True)
    sns.set_style("whitegrid")

    df_seca = df_mensal[df_mensal['Evento'] == 'Seca (2014-2016)']
    sns.lineplot(data=df_seca, x='Data', y='NDVI', ax=ax1, color='#d95f02', linewidth=2.5, marker='o')
    ax1.axvspan(pd.to_datetime('2014-01-01'), pd.to_datetime('2014-11-01'), color='#d95f02', alpha=0.15, label='Período Crítico de Seca')
    ax1.set_title('Impacto da Seca Severa (2014-2016) no Vigor do Café', fontsize=14, fontweight='bold')
    ax1.set_ylabel('NDVI Médio', fontsize=12)
    ax1.set_xlabel('')
    ax1.legend(loc='lower right')

    df_geada = df_mensal[df_mensal['Evento'] == 'Geada (2021-2023)']
    sns.lineplot(data=df_geada, x='Data', y='NDVI', ax=ax2, color='#1b9e77', linewidth=2.5, marker='o')
    ax2.axvline(pd.to_datetime('2021-07-20'), color='#e7298a', linestyle='--', linewidth=2, label='Geada Severa (Julho/2021)')
    ax2.set_title('Impacto da Geada e Podas Subsequentes (2021-2023) no Vigor do Café', fontsize=14, fontweight='bold')
    ax2.set_ylabel('NDVI Médio', fontsize=12)
    ax2.set_xlabel('Linha do Tempo', fontsize=12)
    ax2.legend(loc='lower right')

    plt.tight_layout()

    nome_grafico = 'Comparativo_NDVI_MapBiomas_Corrigido.png'
    nome_tabela = 'Dados_NDVI_MapBiomas_Landsat.csv'
    plt.savefig(nome_grafico, dpi=300)
    df_mensal.to_csv(nome_tabela, index=False, sep=';', decimal=',')

    files.download(nome_grafico)
    files.download(nome_tabela)

    print("✅ Sucesso absoluto! O gráfico final e a tabela estão sendo baixados para o seu computador.")