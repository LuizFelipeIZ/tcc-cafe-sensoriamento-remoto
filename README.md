# Estimativa intra-safra da produtividade e atribuição de danos climáticos no café arábica com sensoriamento remoto

Trabalho de Conclusão de Curso em **Engenharia Agronômica**, Instituto Federal do Sul de Minas Gerais (IFSULDEMINAS), Campus Muzambinho.

**Autor:** Luiz Felipe Viana Dias da Silva
**Orientador:** Prof. Dr. Allan Arantes Pereira
**Defesa:** 25 de novembro de 2026

Este repositório reúne os dados, os códigos e os resultados do TCC, na ordem em que o trabalho foi feito: coleta dos dados, pré-processamento, montagem da base, modelagem e avaliação. Todos os números das tabelas do texto são reproduzidos pelos scripts daqui.

---

## Sumário

1. [Resumo](#1-resumo)
2. [Área de estudo](#2-área-de-estudo)
3. [Perguntas e hipótese](#3-perguntas-e-hipótese)
4. [Dados](#4-dados)
5. [Fluxo do trabalho e organização do repositório](#5-fluxo-do-trabalho-e-organização-do-repositório)
6. [Como rodar](#6-como-rodar)
7. [Metodologia, etapa por etapa](#7-metodologia-etapa-por-etapa)
8. [Resultados](#8-resultados)
9. [Experimentos e versões anteriores](#9-experimentos-e-versões-anteriores)
10. [Limitações](#10-limitações)
11. [Reprodutibilidade](#11-reprodutibilidade)
12. [Referências principais](#12-referências-principais)

---

## 1. Resumo

A produtividade do café arábica depende do clima e da **bienalidade**, a alternância entre safras de alta e de baixa carga. Separar o que é dano climático do que é ciclo bienal ou ganho tecnológico é difícil, porque as séries oficiais misturam tudo.

Este trabalho montou uma base municipal de **1989 a 2023** (10 municípios × 35 safras = **350 observações**) com o rendimento do IBGE, o clima do ERA5-Land por fase fenológica e o NDVI das lavouras de café (Landsat 5, 7 e 8, só nos pixels de café do MapBiomas). Com ela, um **Random Forest** foi comparado à persistência histórica da produção, o dano climático foi atribuído por fase (MDI e SHAP) e dois previsores intra-safra, de seca e de geada, foram testados em janela expansiva.

Principais resultados:

* O Random Forest reduziu o erro médio absoluto (MAE) de **6,26 para 4,80 sacas/ha (−23%)**, com diferença significativa por observação (p < 0,0001) e por safra (p = 0,048). Sem as safras de 1991 a 2002, em que o IBGE repete valores, o MAE caiu para **4,14 sacas/ha (−33%)**.
* O ganho vem da **tendência temporal** e da **anomalia bienal**: retirar o clima ou o NDVI não muda o erro de forma significativa.
* Na atribuição, a **florada (t−1)** concentra 48% do peso do clima, o **inverno (t−1)** 36% e o **enchimento** 16%. Após a geada de 2021, o clima reduziu o rendimento estimado em **2,1 sacas/ha** e o NDVI em mais **1,8**, de uma queda observada de 6,8.
* Os previsores intra-safra **não superaram** a baseline (seca: 3,99 contra 4,21 sacas/ha; geada: 6,90 contra 5,92), pelo pequeno número de eventos na série.

---

## 2. Área de estudo

Dez municípios do entorno de São Sebastião do Paraíso, no Sul de Minas Gerais: **Arceburgo, Cabo Verde, Guaranésia, Guaxupé, Itamogi, Jacuí, Juruaia, Monte Belo, Muzambinho e Nova Resende**. Somaram 59,7 mil ha de café colhidos em 2023 (IBGE). Clima Cwa e Cwb, com chuvas de outubro a março e inverno seco, quando ocorrem as geadas.

A **unidade de observação** é o município em cada safra, porque a produção vem do levantamento municipal do IBGE.

![Área de estudo](resultados/figuras/figura2_area_estudo.png)

---

## 3. Perguntas e hipótese

1. Quanto da variação da safra se deve ao clima, separado da bienalidade e da tendência?
2. Antes da colheita, clima e satélite estimam a safra melhor que o histórico?

**Hipótese:** o Random Forest, com clima e NDVI, estima a produtividade com erro menor que a persistência histórica.

**Objetivos específicos:** (i) construir a base municipal 1989-2023; (ii) comparar o Random Forest com a persistência histórica; (iii) atribuir o dano climático por fase (MDI e SHAP) em sacas/ha; (iv) testar a previsão intra-safra nos eventos de seca e de geada.

---

## 4. Dados

Todas as bases são públicas e cobrem 1988 a 2023. A safra de 1988 serve só para as defasagens.

| Dado | Fonte | Acesso | Arquivo em `dados/brutos/` |
|---|---|---|---|
| Produção (área, produção, rendimento) | IBGE, PAM, Tabela 1613 | API do SIDRA | `producao_ibge_pam_1988_2023.csv` |
| Clima diário (mínima e chuva) | ERA5-Land, ~11 km | Google Earth Engine | `clima_era5land_janelas_1988_2023.csv` |
| Imagens (NDVI) | Landsat 5, 7 e 8, Coleção 2, 30 m | Google Earth Engine | `ndvi_landsat_por_sensor_1989_2023.csv` |
| Área de café (máscara) | MapBiomas, Coleção 9, classe 46 | Google Earth Engine | usada nas extrações |
| Balanço hídrico | ERA5 e ERA5-Land via Open-Meteo, na sede | API Open-Meteo | `balanco_hidrico_openmeteo_1988_2023.csv` |

O dicionário de cada coluna está em [`dados/README.md`](dados/README.md).

> **Uma observação sobre o clima.** Todo o clima vem da família ERA5. O que muda é o acesso: para as fases fenológicas, a média sobre a área de café, no Earth Engine; para o balanço hídrico do previsor de seca, um ponto na sede do município, pelo Open-Meteo.

---

## 5. Fluxo do trabalho e organização do repositório

![Fluxograma](resultados/figuras/figura1_fluxograma.png)

```
tcc-cafe-sensoriamento-remoto/
├── config.py                      caminhos, variáveis, hiperparâmetros e funções comuns
├── executar_tudo.py               roda as etapas 2 a 5 em ordem
├── requirements.txt
│
├── etapa1_coleta/                 ETAPA 1: de onde vêm os dados (precisa de internet)
│   ├── 1a_producao_ibge_sidra.py
│   ├── 1b_clima_era5land_gee.py
│   ├── 1c_ndvi_landsat_gee.py
│   └── 1d_balanco_hidrico_openmeteo.py
│
├── etapa2_preprocessamento/       ETAPA 2: auditoria e calibração
│   ├── 2a_auditoria_e_matriz.py
│   └── 2b_calibracao_ndvi.py
│
├── etapa3_base/                   ETAPA 3: base de modelagem (N = 350)
│   └── 3_montar_base.py
│
├── etapa4_modelagem/              ETAPA 4: Random Forest e previsores
│   ├── 4a_random_forest_kfold_aninhado.py      Tabelas 1 e 2
│   └── 4b_previsores_intrasafra.py             Tabela 4
│
├── etapa5_avaliacao/              ETAPA 5: atribuição e diagnósticos
│   ├── 5a_importancia_mdi_shap.py              Figuras 3 e 4
│   ├── 5b_dano_por_evento.py                   Tabela 3
│   ├── 5c_erros_e_diagnosticos.py              erros por município, viés, mapa
│   └── 5d_figuras_metodologia.py               Figuras 1 e 2, séries
│
├── experimentos/                  testes que guiaram as escolhas + versões anteriores
├── extras/                        imagens de satélite para a apresentação (GEE)
├── dados/
│   ├── brutos/                    saídas da etapa 1 (já baixadas)
│   ├── processados/               geradas pelas etapas 2 e 3
│   └── limites/                   limites municipais (mapas)
└── resultados/                    tabelas, JSON e figuras gerados
```

---

## 6. Como rodar

### Etapas 2 a 5 (sem internet, ~4 minutos)

```bash
git clone https://github.com/LuizFelipeIZ/tcc-cafe-sensoriamento-remoto.git
cd tcc-cafe-sensoriamento-remoto
python -m venv .venv && source .venv/bin/activate      # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
python executar_tudo.py                                 # ou: python executar_tudo.py --com-experimentos
```

Cada script também roda sozinho, por exemplo `python etapa4_modelagem/4a_random_forest_kfold_aninhado.py`. Os resultados vão para `resultados/` e as figuras para `resultados/figuras/`.

No Google Colab, depois de clonar: `%cd tcc-cafe-sensoriamento-remoto` e `!python executar_tudo.py`.

### Etapa 1 (coleta, opcional)

Os arquivos da coleta já estão em `dados/brutos/`, então esta etapa só é necessária para refazer o download.

* `1a` e `1d` precisam de internet (APIs do IBGE e do Open-Meteo): rode no seu computador ou no Colab.
* `1b` e `1c` precisam de uma conta no **Google Earth Engine**: rode no Colab, trocando `MEU_PROJETO` pelo ID do seu projeto. A extração do clima leva algumas horas.

---

## 7. Metodologia, etapa por etapa

### Etapa 1: coleta

| Script | O que faz |
|---|---|
| `1a_producao_ibge_sidra.py` | Baixa área colhida, produção e rendimento da Tabela 1613. **Café total** de 1988 a 2012 (a tabela não separa o arábica nos anos iniciais) e **café arábica** de 2013 a 2023. |
| `1b_clima_era5land_gee.py` | Monta a "pegada" de café de cada município (união do café do MapBiomas em 2000, 2010, 2015, 2020 e 2023), extrai dia a dia a mínima e a chuva do ERA5-Land nessa área e resume em janelas: florada (set-nov), enchimento (jan-abr e dez) e inverno (jun-ago). |
| `1c_ndvi_landsat_gee.py` | Calcula o NDVI só nos pixels de café do próprio ano, sem nuvem, sombra e saturação, e resume o ano agrícola (jul t−1 a jun t) em vigor (P90), estresse (P10), florada (média set-nov t−1) e amplitude. Cada sensor é extraído separadamente. |
| `1d_balanco_hidrico_openmeteo.py` | Balanço hídrico mensal de Thornthwaite e Mather (CAD 100 mm) na sede, resumido em déficit da florada, déficit da expansão dos grãos (dez t−1 a mar t) e máxima média da expansão. |

### Etapa 2: pré-processamento

**2a. Auditoria da série e matriz.** Antes de modelar, a série é conferida: cobertura das 36 safras nos 10 municípios, duplicidade, grafia dos nomes, coerência entre rendimento, produção e área. O script também procura safras em que o IBGE **repete** o rendimento do ano anterior: são 42 casos em 9 municípios, 83% deles entre 1991 e 2002. Isso indica que o dado oficial não foi levantado de novo nessas safras, e por isso todas as análises são repetidas sem 1991-2002.

**2b. NDVI na mesma escala.** Landsat 5, 7 e 8 medem o NDVI de forma um pouco diferente. Nos anos em que dois sensores se sobrepõem, ajusta-se uma reta pela **regressão do eixo maior reduzido (RMA)**, que leva o Landsat 5 ao 7 (2000-2011) e o 7 ao 8 (2014-2022). A série final usa o Landsat 5 em 1989-1999, o 7 em 2000-2013 e o 8 em 2014-2023, toda na escala do Landsat 8.

Por que RMA: a regressão comum supõe erro só no y e "achata" a série corrigida. Na validação deixando uma safra de fora, o RMA preservou **100% a 107%** da variabilidade, contra **77% a 87%** da regressão comum, com viés < 0,002 e MAE de 0,008 a 0,028.

### Etapa 3: base de modelagem

Cada observação (município *i*, safra *t*) tem o rendimento e **12 preditoras**:

**Controles**

* Persistência histórica: M(t) = [Y(t−1) + Y(t−2) + Y(t−3)] / 3
* Anomalia bienal: A(t) = Y(t−1) − M(t)
* Tendência temporal: T(t) = t − 1988

O modelo **nunca** usa o rendimento da safra que estima. A média móvel é calculada com `shift(1)` antes do `rolling`, o que evita vazamento do alvo (uma versão anterior tinha esse erro; veja `experimentos/versoes_anteriores/v01`).

**Clima (6), no ano agrícola.** A florada que forma a safra *t* acontece em setembro a novembro de *t−1*, e o inverno de risco de geada, em junho a agosto de *t−1*. Por isso essas janelas são defasadas em um ano.

| Variável | Fase | Período |
|---|---|---|
| Chuva na florada (t−1) | florada | set-nov de t−1 |
| Temp. mínima na florada (t−1) | florada | set-nov de t−1 |
| Chuva no inverno (t−1) | inverno | jun-ago de t−1 |
| Temp. mínima no inverno (t−1) | inverno | jun-ago de t−1 |
| Chuva no enchimento | enchimento | jan-abr e dez de t |
| Temp. mínima no enchimento | enchimento | jan-abr e dez de t |

**NDVI (4):** vigor máximo, estresse máximo, NDVI na florada e amplitude.

### Etapa 4: modelagem

**4a. Random Forest com validação cruzada aninhada.**
* Ciclo externo: 5 partes; 4 treinam e 1 é estimada, até todas serem estimadas.
* Ciclo interno: busca em grade em 3 partes, só no treino (100 ou 150 árvores, profundidade 3 ou 5, 5 ou 10 amostras por divisão).
* Comparações: persistência histórica M(t) e Y(t−2), o rendimento de dois anos antes, na mesma fase do ciclo bienal.
* Métricas: MAE, RMSE e R² fora da amostra, em sacas/ha.
* Teste de Wilcoxon pareado nos erros absolutos, **por observação** e **por safra** (média dos 10 municípios, porque eles compartilham o clima).
* Modelos reduzidos, sem NDVI, sem clima e só com controles, com ajuste de Holm.
* Tudo repetido sem 1991-2002 (N = 230).

> **O que o K-Fold mede.** Como outros municípios da mesma safra ficam no treino, o K-Fold mede a capacidade de **completar a série** (estimar um município numa safra em que os vizinhos são conhecidos), e não de prever safras futuras. O experimento `e2` mostra isso. A previsão do futuro é testada no 4b.

**4b. Previsores intra-safra em janela expansiva.** Cada safra de teste é estimada por um modelo treinado **só com as safras anteriores**.
* **Seca:** controles + déficit na florada + déficit na expansão + máxima da expansão; testado em 2014-2017 (N = 40).
* **Geada:** controles + mínima, dias abaixo de 3 °C e chuva do inverno anterior; testado em 1994, 1995, 2000, 2001, 2021, 2022 e 2023 (N = 70).
* Hiperparâmetros fixos (200 árvores, profundidade 5, mínimo de 5 por divisão), porque o primeiro treino tem só 50 observações.
* Incerteza pelo intervalo P10-P90 das 200 árvores.

### Etapa 5: avaliação e atribuição

* **5a.** Importância por **MDI** (redução de variância) e por **SHAP** (efeito de cada variável em sacas/ha, com direção), no modelo ajustado com as 350 observações; soma por fase e estabilidade nos 5 modelos do K-Fold.
* **5b.** Dano em sacas/ha: soma dos valores SHAP das 6 variáveis climáticas (e das 4 de NDVI), na média dos 10 municípios, em cada safra de evento.
* **5c.** Viés por faixa de rendimento, erro por município (com mapa) e a mudança de sinal da anomalia bienal com e sem 1991-2002.
* **5d.** Figuras de apoio: fluxograma, área de estudo e séries.

---

## 8. Resultados

Valores em sacas/ha. Gerados por `executar_tudo.py`; os arquivos completos estão em `resultados/`.

### Tabela 1: Random Forest contra as baselines (K-Fold aninhado)

| Conjunto | Modelo | MAE (dp entre partes) | RMSE | R² |
|---|---|---|---|---|
| Série completa (N = 350) | Persistência histórica M(t) | 6,26 | 8,11 | 0,05 |
| | Y(t−2) (N = 340) | 6,24 | 8,66 | −0,06 |
| | **Random Forest** | **4,80** (0,38) | **6,48** | **0,40** |
| Sem 1991-2002 (N = 230) | Persistência histórica M(t) | 6,20 | 7,79 | 0,10 |
| | Y(t−2) (N = 220) | 5,73 | 7,52 | 0,19 |
| | **Random Forest** | **4,14** (0,37) | **5,36** | **0,58** |

Random Forest contra M(t): p < 0,0001 por observação e p = 0,048 por safra (menor erro em 21 de 35 safras). Sem 1991-2002: p = 0,028 por safra. Contra Y(t−2): p = 0,026 por safra.

### Tabela 2: de onde vem o ganho (modelos reduzidos)

| Modelo | MAE, série completa | p obs / p safra [Holm] | MAE, sem 1991-2002 | p obs / p safra |
|---|---|---|---|---|
| Completo (12 variáveis) | 4,80 | | 4,14 | |
| Sem NDVI | 4,90 | 0,018 / 0,18 [0,11 / 0,54] | 4,17 | 0,44 / 0,64 |
| Sem clima | 4,76 | 0,58 / 0,83 | 4,34 | 0,19 / 0,086 |
| Só controles | **4,54** | 0,052 / 0,067 | 4,18 | 0,59 / 0,75 |

O modelo só com tendência e anomalia bienal teve o menor erro na série completa. A falta de ganho do NDVI não vem de redundância com o clima (Spearman de no máximo 0,34), e sim do pequeno número de safras e eventos e do ruído da série.

### Atribuição por variável e por fase (Figura 3)

![Importância](resultados/figuras/figura3_importancia.png)

* Tendência temporal 30,7%, anomalia bienal 16,6%, chuva na florada (t−1) 9,3%.
* Controles 47,3%, clima 36,6%, NDVI 16,0% (MDI); concordância MDI × SHAP: ρ = 0,94.
* Dentro do clima: florada (t−1) 48%, inverno (t−1) 36%, enchimento 16%, estáveis nos 5 modelos do K-Fold (45-52%, 28-34% e 13-21%).

### Direção dos efeitos (Figura 4)

![SHAP](resultados/figuras/figura4_shap.png)

Mais chuva na florada e NDVI na florada mais alto aumentam o rendimento estimado; amplitude maior (mais perda de folha no ano) o reduz. Invernos muito secos reduziram a estimativa sobretudo em 2021 e 2022, ligados à florada seca de 2020 e à geada de 2021.

### Tabela 3: dano nas safras de evento (média dos 10 municípios)

| Safra | Evento | Observado | M(t) | Obs − M(t) | Clima | NDVI |
|---|---|---|---|---|---|---|
| 1994 | Geada de 1994 | 21,2 | 19,7 | +1,5 | −0,4 | +0,2 |
| 1995 | Após a geada de 1994 | 13,1 | 20,3 | −7,2 | −0,7 | −1,3 |
| 2000 | Geada de 2000 | 22,4 | 28,5 | −6,1 | +1,2 | −0,3 |
| 2001 | Após a geada de 2000 | 20,2 | 26,2 | −6,1 | +1,0 | +0,2 |
| 2014 | Seca de 2014 | 26,9 | 25,6 | +1,4 | −0,3 | +1,0 |
| 2015 | Após a seca de 2014 | 26,6 | 27,5 | −0,9 | +0,2 | +0,1 |
| 2021 | Geada de 2021 | 21,0 | 29,9 | −8,9 | −1,6 | −1,0 |
| **2022** | **Após a geada de 2021** | 20,5 | 27,3 | **−6,8** | **−2,1** | **−1,8** |
| 2023 | Segunda safra após a geada | 24,6 | 24,6 | 0,0 | −0,2 | −1,1 |

Numa safra qualquer, o efeito médio do clima é de 1,4 sacas/ha. A queda observada inclui a bienalidade, por isso é maior que o dano atribuído.

### Erros por município

![Mapa de erros](resultados/figuras/figura_mapa_erro_municipios.png)

O Random Forest errou menos que a baseline em 9 dos 10 municípios, de 12% (Juruaia) a 50% (Guaxupé). Os maiores erros foram de Itamogi (6,92) e Monte Belo (6,84), o único em que a baseline errou menos. O modelo comprime os extremos: superestima safras abaixo de ~12 sacas/ha e subestima as acima de 35.

![Observado x estimado](resultados/figuras/figura_obs_vs_estimado.png)

### Tabela 4: previsores intra-safra (janela expansiva)

| Previsor | Safras de teste | RF | Só controles | Baseline M(t) | p (RF × baseline) |
|---|---|---|---|---|---|
| Seca | 2014 a 2017 (N = 40) | 3,99 | 4,27 | 4,21 | 0,91 |
| Geada | 1994, 1995, 2000, 2001, 2021, 2022, 2023 (N = 70) | 6,90 | 7,28 | 5,91 | 0,98 |
| Geada, sem 2000 e 2001* | (N = 50) | 4,61 | 5,87 | 5,71 | 0,015 |

\* Exclusão definida depois de ver os resultados (safras com valores repetidos no IBGE); vale só como sensibilidade.

Os intervalos P10-P90 cobriram 67,5% (seca) e 70,0% (geada) dos valores reais, abaixo dos 80% esperados. Erro percentual médio do previsor de seca: 13,8%.

---

## 9. Experimentos e versões anteriores

A pasta [`experimentos/`](experimentos/README.md) guarda os testes que guiaram as escolhas do desenho final:

* **e1:** generalização para municípios inteiros fora do treino e para o bloco 2014-2023;
* **e2:** por que o K-Fold mede "completar a série" (só a tendência já erra 4,33; com a safra inteira fora, o RF erra 6,68);
* **e3:** residualização da tendência e da bienalidade antes do clima;
* **e4:** o balanço hídrico no modelo principal (sem ganho significativo).

Em `experimentos/versoes_anteriores/` estão os códigos de fases anteriores (cenários separados, Sentinel-2, CHIRPS, a primeira baseline com vazamento etc.), com uma tabela explicando o que cada um fazia e por que foi substituído.

---

## 10. Limitações

* Em 9 dos 10 municípios o IBGE repete o rendimento por 2 a 4 safras, sobretudo em 1999-2002; a PAM consolida estimativas de informantes locais.
* O K-Fold não mede a previsão de safras futuras; há poucos eventos extremos (uma seca e três geadas).
* Os municípios de uma mesma safra compartilham o clima, por isso o teste por safra é o mais honesto.
* O rendimento até 2012 é de café total; a partir de 2013, de café arábica.
* O ERA5-Land, com ~11 km, suaviza a temperatura mínima; a máscara de café do clima é fixa.
* A janela de enchimento inclui dezembro do mesmo ano civil, depois da colheita.
* O balanço hídrico usa um ponto na sede, e não a área de café.
* MDI e SHAP descrevem como o modelo **usou** as variáveis no ajuste, e não relações de causa.

O protocolo é replicável com dados públicos e tende a melhorar com dados mais precisos de produtividade e de clima, como os de cooperativas e de estações meteorológicas.

---

## 11. Reprodutibilidade

* Semente fixa (`random_state = 42`) em todos os modelos e divisões; os números se repetem a cada execução com as versões de `requirements.txt`.
* Diferenças de arredondamento em relação ao texto do TCC: a baseline do previsor de geada sai 5,91 aqui (5,92 no texto) e a parcela do inverno, 35,5% (36% no texto).
* A etapa 5a reproduz as faixas de estabilidade por fase com a ordem de colunas usada na checagem original (comentado no script).
* As etapas de coleta (1a a 1d) não foram executadas neste repositório porque dependem de internet e do Earth Engine; os arquivos que elas geram são os de `dados/brutos/`, usados no TCC.

---

## 12. Referências principais

* BREIMAN, L. Random forests. *Machine Learning*, v. 45, p. 5-32, 2001.
* CAMARGO, A. P.; CAMARGO, M. B. P. Definição e esquematização das fases fenológicas do cafeeiro arábica nas condições tropicais do Brasil. *Bragantia*, v. 60, p. 65-68, 2001.
* DAMATTA, F. M.; RAMALHO, J. D. C. Impacts of drought and temperature stress on coffee physiology and production. *Brazilian Journal of Plant Physiology*, v. 18, p. 55-81, 2006.
* GORELICK, N. et al. Google Earth Engine: planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, v. 202, p. 18-27, 2017.
* LUNDBERG, S. M.; LEE, S.-I. A unified approach to interpreting model predictions. *NeurIPS*, 2017.
* MUÑOZ-SABATER, J. et al. ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, v. 13, p. 4349-4383, 2021.
* ROY, D. P. et al. Characterization of Landsat-7 to Landsat-8 reflective wavelength and normalized difference vegetation index continuity. *Remote Sensing of Environment*, v. 185, p. 57-70, 2016.
* SOUZA, C. M. et al. Reconstructing three decades of land use and land cover changes in Brazilian biomes with Landsat archive and Earth Engine. *Remote Sensing*, v. 12, 2735, 2020.
* THORNTHWAITE, C. W.; MATHER, J. R. *The water balance*. Centerton: Drexel Institute of Technology, 1955.

A lista completa está no texto do TCC.

---

**Como citar:** SILVA, L. F. V. D. *Estimativa intra-safra da produtividade e atribuição de danos climáticos no café arábica com sensoriamento remoto*. Trabalho de Conclusão de Curso (Engenharia Agronômica). IFSULDEMINAS, Campus Muzambinho, 2026.
