# Dados

Todas as bases são públicas. Os arquivos de `brutos/` são as saídas da etapa 1 (coleta), já baixadas, para que o projeto rode sem internet a partir da etapa 2. Os de `processados/` são gerados pelas etapas 2 e 3 e também ficam no repositório.

Os nomes dos municípios estão sem acento em todas as bases (Guaranesia, Guaxupe, Jacui), para que as junções não falhem por grafia.

## `brutos/`

### `producao_ibge_pam_1988_2023.csv`
IBGE, Produção Agrícola Municipal, Tabela 1613 (SIDRA). Gerado por `etapa1_coleta/1a_producao_ibge_sidra.py`. 360 linhas (10 municípios × 36 safras).

| Coluna | Descrição | Unidade |
|---|---|---|
| `Municipio` | município | |
| `Ano` | ano da safra (colheita de maio a agosto) | |
| `Area_Colhida_ha` | área colhida | ha |
| `Producao_t` | quantidade produzida | t |
| `Rendimento_kg_ha` | rendimento médio (variável alvo) | kg/ha; ÷ 60 = sacas/ha |

Café total de 1988 a 2012, porque a tabela não separa o arábica nos anos iniciais, e café arábica de 2013 a 2023.

### `clima_era5land_janelas_1988_2023.csv`
Reanálise ERA5-Land (ECMWF/ERA5_LAND/DAILY_AGGR), média diária sobre a área de café de cada município (união do café do MapBiomas em 2000, 2010, 2015, 2020 e 2023). Gerado por `etapa1_coleta/1b_clima_era5land_gee.py`. As janelas são do **ano civil**; a defasagem para o ano agrícola é feita na etapa 3.

| Coluna | Descrição | Unidade |
|---|---|---|
| `Tmin_Abs` | temperatura mínima absoluta do ano | °C |
| `Precip_Anual` | chuva acumulada no ano | mm |
| `Dias_Abaixo_3C` | dias com mínima abaixo de 3 °C em junho a agosto | dias |
| `Precip_Florada`, `Tmin_Abs_Florada` | chuva e mínima absoluta de setembro a novembro | mm, °C |
| `Precip_Enchimento_Graos`, `Tmin_Abs_Enchimento_Graos` | chuva e mínima absoluta de janeiro a abril e dezembro do ano civil (substituídas na etapa 2a pelas do arquivo abaixo) | mm, °C |
| `Precip_Inverno_Risco_Geada`, `Tmin_Abs_Inverno_Risco_Geada` | chuva e mínima absoluta de junho a agosto | mm, °C |

### `ndvi_landsat_por_sensor_1989_2023.csv`
NDVI das lavouras de café (MapBiomas Coleção 9, classe 46, do ano da safra), por **sensor**, Landsat 5, 7 e 8, Coleção 2, nível 2. Gerado por `etapa1_coleta/1c_ndvi_landsat_gee.py`. 570 linhas, porque nos anos de sobreposição há mais de um sensor por município e safra.

| Coluna | Descrição |
|---|---|
| `Sensor` | L5, L7 ou L8 |
| `NDVI_Vigor_Maximo` | percentil 90 do NDVI de julho de t−1 a junho de t |
| `NDVI_Estresse_Maximo` | percentil 10 no mesmo período |
| `NDVI_Florada` | média do NDVI de setembro a novembro de t−1 |
| `NDVI_Amplitude` | vigor − estresse |
| `N_obs_medio_pixel` | observações limpas por pixel no ano agrícola (média) |
| `N_imagens` | cenas do sensor no ano agrícola |

### `clima_enchimento_dez_abr_1988_2023.csv`
Mesma reanálise e mesma área de café, mas só a janela de enchimento do ano agrícola: de 1º de dezembro de t−1 a 30 de abril de t (151 ou 152 dias). Gerado por `etapa1_coleta/1b2_enchimento_dez_abr_gee.py`; a etapa 2a usa estas colunas no lugar das de enchimento do arquivo acima.

| Coluna | Descrição | Unidade |
|---|---|---|
| `Municipio`, `Ano` | município e safra t | |
| `Precip_Enchimento_Graos` | chuva acumulada de dez de t−1 a abr de t | mm |
| `Tmin_Abs_Enchimento_Graos` | mínima absoluta de dez de t−1 a abr de t | °C |
| `Dias` | dias com dado na janela | dias |

### `balanco_hidrico_openmeteo_1988_2023.csv`
Balanço hídrico de Thornthwaite e Mather (1955), CAD de 100 mm, com temperatura e chuva diárias da Historical Weather API do Open-Meteo (ERA5 e ERA5-Land; IFS do ECMWF a partir de 2017) num ponto na sede de cada município. Gerado por `etapa1_coleta/1d_balanco_hidrico_openmeteo.py`. Usado só no previsor de seca.

| Coluna | Descrição | Unidade |
|---|---|---|
| `DEF_Florada` | déficit hídrico somado de setembro a novembro de t−1 | mm |
| `DEF_Expansao_Graos` | déficit somado de dezembro de t−1 a março de t | mm |
| `Tmax_Expansao_Graos` | temperatura máxima média de dezembro de t−1 a março de t | °C |

## `limites/municipios_recorte_ssp.gpkg`
Limites municipais do IBGE (2023) no entorno de São Sebastião do Paraíso, usados só nos mapas.

## `processados/` (gerados)

| Arquivo | Etapa | Conteúdo |
|---|---|---|
| `matriz_mestra.csv` | 2a | produção + clima, 360 linhas |
| `ndvi_harmonizado_rma.csv` | 2b | NDVI na escala do Landsat 8, 350 linhas (1989 a 2023) |
| `base_modelagem.csv` | 3 | base final, N = 350, com as 12 preditoras, M(t) e Y(t−2) |
