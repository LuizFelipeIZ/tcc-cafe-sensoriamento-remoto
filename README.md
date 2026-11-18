# Sensoriamento remoto e aprendizado de máquina na estimativa da produtividade e do dano climático no café arábica

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

Este trabalho montou uma base municipal de **1989 a 2023** (10 municípios × 35 safras = **350 observações**) com o rendimento do IBGE, o clima do ERA5-Land por fase fenológica e o NDVI das lavouras de café (Landsat 5, 7 e 8, só nos pixels de café do MapBiomas). Com ela, um **Random Forest** foi comparado à persistência histórica da produção, o dano climático foi atribuído por fase (MDI e SHAP) e dois previsores intrassafra, de seca e de geada, foram testados em janela expansiva.

Principais resultados:

* Ao completar a série (K-Fold aninhado), o Random Forest reduziu o erro médio absoluto (MAE) de **6,26 para 4,69 sacas/ha (−25%)**, com diferença significativa por observação (p < 0,0001) e por safra (p = 0,014). Sem as safras de 1991 a 2002, em que o IBGE repete valores, o MAE caiu para **4,19 sacas/ha (−32%)**.
* Com a **safra inteira fora do treino**, o erro foi de **6,05 sacas/ha**, sem diferença significativa da persistência (p = 0,73 por safra): o ganho depende dos outros municípios da mesma safra no treino.
* O ganho vem da **tendência temporal** e da **anomalia bienal**: retirar o clima ou o NDVI não muda o erro de forma significativa, e a hipótese foi rejeitada.
* Na atribuição, a **florada (t−1)** concentra 44% do peso do clima no modelo, o **inverno (t−1)** 33% e o **enchimento** 23%. Na primeira safra após a geada de 2021, o modelo atribuiu **2,0 sacas/ha** ao clima e **1,7** ao NDVI, de uma queda observada de 6,8.
* Os previsores intrassafra **não superaram** a persistência (seca: 4,16 contra 4,21 sacas/ha; geada: 6,90 contra 5,91), pelo pequeno número de eventos na série.

---

## 2. Área de estudo

Dez dos 14 municípios da microrregião de São Sebastião do Paraíso, no Sul de Minas Gerais: **Arceburgo, Cabo Verde, Guaranésia, Guaxupé, Itamogi, Jacuí, Juruaia, Monte Belo, Muzambinho e Nova Resende**, os que têm a série de produtividade de café completa na PAM de 1988 a 2023. Somaram 59,7 mil ha de café colhidos em 2023 (IBGE). Clima Cwa e Cwb, com chuvas de outubro a março e inverno seco, quando ocorrem as geadas.

A **unidade de observação** é o município em cada safra, porque a produção vem do levantamento municipal do IBGE.

![Área de estudo](resultados/figuras/figura2_area_estudo.png)

---

## 3. Perguntas e hipótese

1. Quanto da variação da safra se deve ao clima, separado da bienalidade e da tendência?
2. Antes da colheita, clima e satélite estimam a safra melhor que o histórico?

**Hipótese:** o clima e o NDVI reduzem o erro de estimativa do Random Forest além do obtido só com a tendência e a bienalidade.

**Objetivos específicos:** (i) construir a base municipal 1989-2023; (ii) comparar o Random Forest com a persistência histórica; (iii) atribuir o dano climático por fase (MDI e SHAP) em sacas/ha; (iv) testar a previsão intrassafra nos eventos de seca e de geada.

---

## 4. Dados

Todas as bases são públicas e cobrem 1988 a 2023. A safra de 1988 serve só para as defasagens.

| Dado | Fonte | Acesso | Arquivo em `dados/brutos/` |
|---|---|---|---|
| Produção (área, produção, rendimento) | IBGE, PAM, Tabela 1613 | API do SIDRA | `producao_ibge_pam_1988_2023.csv` |
| Clima diário (mínima e chuva) | ERA5-Land, ~11 km | Google Earth Engine | `clima_era5land_janelas_1988_2023.csv` |
| Enchimento (dez t−1 a abr t) | ERA5-Land, ~11 km | Google Earth Engine | `clima_enchimento_dez_abr_1988_2023.csv` |
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
│   ├── 1b2_enchimento_dez_abr_gee.py   janela de enchimento corrigida
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
│   ├── 4a_random_forest_kfold_aninhado.py      Tabelas 1 e 2 (com a safra inteira fora)
│   └── 4b_previsores_intrasafra.py             Tabela 4
│
├── etapa5_avaliacao/              ETAPA 5: atribuição e diagnósticos
│   ├── 5a_importancia_mdi_shap.py              Figuras 3 e 4
│   ├── 5b_dano_por_evento.py                   Tabela 3, 1ª/2ª safra afetada, SHAP fora da partição
│   ├── 5c_erros_e_diagnosticos.py              erros por município, viés, mapa
│   ├── 5d_figuras_metodologia.py               Figuras 1 e 2, séries
│   └── 5e_numeros_do_texto.py                  números citados no texto
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
* `1b`, `1b2` e `1c` precisam de uma conta no **Google Earth Engine**: rode no Colab, trocando `MEU_PROJETO` pelo ID do seu projeto. A extração do clima (`1b`) leva algumas horas; a do enchimento (`1b2`), poucos minutos.

---

## 7. Metodologia, etapa por etapa

### Etapa 1: coleta

| Script | O que faz |
|---|---|
| `1a_producao_ibge_sidra.py` | Baixa área colhida, produção e rendimento da Tabela 1613. **Café total** de 1988 a 2012 (a tabela não separa o arábica nos anos iniciais) e **café arábica** de 2013 a 2023. |
| `1b_clima_era5land_gee.py` | Monta a "pegada" de café de cada município (união do café do MapBiomas em 2000, 2010, 2015, 2020 e 2023), extrai dia a dia a mínima e a chuva do ERA5-Land nessa área e resume em janelas: florada (set-nov) e inverno (jun-ago). |
| `1b2_enchimento_dez_abr_gee.py` | Recalcula só o enchimento, na janela do ano agrícola (dezembro de t−1 a abril de t), com a soma e o mínimo feitos no próprio Earth Engine. A etapa 2a usa este arquivo no lugar das colunas de enchimento do `1b`. |
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
| Chuva no enchimento | enchimento | dez de t−1 a abr de t |
| Temp. mínima no enchimento | enchimento | dez de t−1 a abr de t |

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
* Validação com a **safra inteira fora**: cada safra, com os 10 municípios, estimada pelo modelo ajustado com as outras 34 (mesma busca em grade, só no treino).
* Regressão linear de diagnóstico da produtividade sobre a tendência e a anomalia bienal, com e sem 1991-2002.

> **O que o K-Fold mede.** Como outros municípios da mesma safra ficam no treino, o K-Fold mede a capacidade de **completar a série** (estimar um município numa safra em que os vizinhos são conhecidos), e não de prever safras futuras. A validação com a safra inteira fora e o experimento `e2` mostram isso. A previsão do futuro é testada no 4b.

**4b. Previsores intrassafra em janela expansiva.** Cada safra de teste é estimada por um modelo treinado **só com as safras anteriores**.
* **Seca:** controles + déficit na florada + déficit na expansão + máxima da expansão; testado em 2014-2017 (N = 40).
* **Geada:** controles + mínima, dias abaixo de 3 °C e chuva do inverno anterior; testado em 1994, 1995, 2000, 2001, 2021, 2022 e 2023 (N = 70).
* Hiperparâmetros fixos (200 árvores, profundidade 5, mínimo de 5 por divisão), porque o primeiro treino tem só 50 observações.
* Incerteza pelo intervalo P10-P90 das 200 árvores.

### Etapa 5: avaliação e atribuição

* **5a.** Importância por **MDI** (redução de variância) e por **SHAP** (efeito de cada variável em sacas/ha, com direção), no modelo ajustado com as 350 observações; soma por fase e estabilidade nos 5 modelos do K-Fold.
* **5b.** Dano em sacas/ha: soma dos valores SHAP das 6 variáveis climáticas (e das 4 de NDVI), na média dos 10 municípios, em cada safra de evento. Na parte 2, acompanha a 1ª e a 2ª safra afetada por evento, com a referência fixa nas três safras antes do evento, e calcula também o SHAP **fora da partição** (cada observação explicada pelo modelo do K-Fold que não a usou no treino).
* **5c.** Viés por faixa de rendimento, erro por município (com mapa) e a mudança de sinal da anomalia bienal com e sem 1991-2002.
* **5d.** Figuras de apoio: fluxograma, área de estudo e séries.
* **5e.** Números citados no texto que não estão nas outras saídas (efeito da chuva no inverno por faixa, sentido de cada SHAP, resumo do clima, hiperparâmetros do modelo final).

---

## 8. Resultados

Valores em sacas/ha. Gerados por `executar_tudo.py`; os arquivos completos estão em `resultados/`.

### Tabela 1: Random Forest contra a persistência histórica

| Conjunto | Modelo | MAE (dp entre partes) | RMSE | R² | p (obs.; safra) |
|---|---|---|---|---|---|
| Série completa (N = 350) | Persistência histórica M(t) | 6,26 | 8,11 | 0,05 | |
| | Y(t−2) (N = 340) | 6,24 | 8,66 | −0,06 | |
| | **Random Forest** | **4,69** (0,45) | **6,38** | **0,41** | < 0,0001; 0,014 |
| | Random Forest, safra inteira fora | 6,05 | 7,86 | 0,11 | 0,31; 0,73 |
| Sem 1991-2002 (N = 230) | Persistência histórica M(t) | 6,20 | 7,79 | 0,10 | |
| | Y(t−2) (N = 220) | 5,73 | 7,52 | 0,19 | |
| | **Random Forest** | **4,19** (0,40) | **5,40** | **0,57** | < 0,0001; 0,030 |
| | Random Forest, safra inteira fora | 6,08 | 7,47 | 0,18 | 0,98; 0,71 |

p: Wilcoxon pareado contra M(t), por observação e por safra. No K-Fold, o Random Forest teve menor erro em 23 de 35 safras; contra Y(t−2), p = 0,021 por safra. Com a safra inteira fora do treino, o ganho some: sem os outros municípios da mesma safra, o modelo erra quase como a persistência.

### Tabela 2: de onde vem o ganho (modelos reduzidos)

| Modelo | MAE, série completa | p obs; p safra [Holm] | MAE, sem 1991-2002 | p obs; p safra [Holm] |
|---|---|---|---|---|
| Completo (12 variáveis) | 4,69 | | 4,19 | |
| Sem NDVI | 4,71 | 0,31; 1 [1; 1] | 4,24 | 0,087; 0,25 [0,52; 0,99] |
| Sem clima | 4,76 | 0,71; 0,81 [1; 1] | 4,34 | 0,59; 0,16 [1; 0,80] |
| Só controles | **4,54** | 0,44; 0,29 [1; 1] | 4,18 | 0,81; 0,71 [1; 1] |

O modelo só com tendência e anomalia bienal teve o menor erro na série completa, e nenhuma retirada foi significativa: a hipótese de que o clima e o NDVI reduzem o erro além dessas variáveis foi rejeitada. A falta de ganho do NDVI não vem de redundância com o clima (Spearman de no máximo 0,34), e sim do pequeno número de safras e eventos e do ruído da série.

### Atribuição por variável e por fase (Figura 3)

![Importância](resultados/figuras/figura3_importancia.png)

* Tendência temporal 30,0%, anomalia bienal 16,2%, chuva na florada (t−1) e chuva no inverno (t−1), 8,7% cada.
* Controles 46,2%, clima 38,3%, NDVI 15,5% (MDI); concordância MDI × SHAP: ρ = 0,91.
* Dentro do clima: florada (t−1) 44%, inverno (t−1) 33%, enchimento 23%, estáveis nos 5 modelos do K-Fold (41-51%, 24-33% e 19-26%).
* Importância não é contribuição: o clima tem 38% da importância, mas retirá-lo não aumenta o erro (Tabela 2). As parcelas mostram onde o modelo encontrou sinal climático, e não quanto o clima pesa na produtividade.

### Direção dos efeitos (Figura 4)

![SHAP](resultados/figuras/figura4_shap.png)

Mais chuva na florada e NDVI na florada mais alto aumentam o rendimento estimado; amplitude maior (mais perda de folha no ano) o reduz. Invernos com menos de 30 mm (antes das safras de 2011, 2021 e 2022) reduziram a estimativa em 2,4 sacas/ha em média, e os de 30 a 60 mm a elevaram em 0,6; os mais secos coincidem com a florada seca de 2020 e a geada de 2021.

### Tabela 3: dano nas safras de evento (média dos 10 municípios)

| Safra | Evento | Observado | M(t) | Obs − M(t) | Clima | NDVI |
|---|---|---|---|---|---|---|
| 1994 | Geada de 1994 | 21,2 | 19,7 | +1,5 | +0,1 | +0,2 |
| 1995 | Após a geada de 1994 | 13,1 | 20,3 | −7,2 | −0,3 | −1,4 |
| 2000 | Geada de 2000 | 22,4 | 28,5 | −6,1 | +1,2 | −0,3 |
| 2001 | Após a geada de 2000 | 20,2 | 26,2 | −6,1 | +0,6 | +0,3 |
| 2014 | Seca de 2014 | 26,9 | 25,6 | +1,4 | +0,1 | +1,0 |
| 2015 | Após a seca de 2014 | 26,6 | 27,5 | −0,9 | +0,6 | +0,1 |
| 2021 | Geada de 2021 | 21,0 | 29,9 | −8,9 | −1,6 | −1,2 |
| **2022** | **Após a geada de 2021** | 20,5 | 27,3 | **−6,8** | **−2,0** | **−1,7** |
| 2023 | Segunda safra após a geada | 24,6 | 24,6 | 0,0 | −0,1 | −1,1 |

Numa safra qualquer, o efeito médio do clima é de 1,5 saca/ha. A queda observada inclui a bienalidade, por isso é maior que o dano atribuído.

### Tabela 3b: 1ª e 2ª safra afetada, com referência fixa

Na Tabela 3, a segunda safra após a geada de 2021 (2023) aparece com queda de 0,0 porque M(2023) é a média de 2020, 2021 e 2022 e já inclui as duas safras danificadas. Aqui a referência fica fixa nas três safras **antes do evento**, em cada município (`resultados/tabela3b_dano_safras_seguintes.csv`). As colunas "fora" são o SHAP fora da partição: cada observação explicada pelo modelo do K-Fold que não a usou no treino.

| Evento | Safra | Afetada | Referência | Observado | Queda | Clima | NDVI | Clima (fora) | NDVI (fora) | Municípios com queda |
|---|---|---|---|---|---|---|---|---|---|---|
| Geada de 2021 | 2022 | 1ª | 27,3 (2019-2021) | 20,5 | **−6,8** (−25%) | −2,0 | −1,7 | −1,3 | −2,0 | 10 |
| Geada de 2021 | 2023 | 2ª | 27,3 (2019-2021) | 24,6 | **−2,7** (−10%) | −0,1 | **−1,1** | +0,1 | −1,2 | 9 |
| Geada de 1994 | 1995 | 1ª | 20,3 (1992-1994) | 13,1 | **−7,2** (−36%) | −0,3 | −1,4 | +0,6 | −1,8 | 10 |
| Geada de 1994 | 1996 | 2ª | 20,3 (1992-1994) | 19,9 | −0,4 (−2%) | +1,5 | −0,5 | +1,7 | −0,5 | 9 |
| Seca de 2014 | 2014 | 1ª (enchimento) | 25,6 (2011-2013) | 26,9 | +1,4 | +0,1 | +1,0 | −0,1 | +1,6 | 2 |
| Seca de 2014 | 2015 | 2ª (florada) | 25,6 (2011-2013) | 26,6 | +1,1 | +0,6 | +0,1 | +0,7 | +0,3 | 3 |

* **Geada de 2021:** o dano continua na 2ª safra. Em 2023 o clima foi normal (−0,1) e quem ainda registra a geada é o NDVI (−1,1), que vê a lavoura se recompondo.
* **Geada de 1994:** o dano ficou concentrado em 1995; em 1996 a produção já tinha voltado ao nível anterior.
* **Seca de 2014:** o IBGE não registra queda nem na safra de 2014 (seca no enchimento dos grãos) nem na de 2015 (florada com 100 mm de déficit).
* A geada de 2000 fica de fora: na safra de 2001, 9 dos 10 municípios repetem o rendimento de 2000.

### Erros por município

![Mapa de erros](resultados/figuras/figura_mapa_erro_municipios.png)

O Random Forest errou menos que a persistência em 9 dos 10 municípios, de 14% (Juruaia) a 51% (Guaxupé). Os maiores erros foram de Itamogi (6,85) e Monte Belo (6,61), o único em que a persistência errou menos. O modelo comprime os extremos: superestima safras abaixo de 12 sacas/ha (em 7,6) e subestima as acima de 35 (em 13,1).

![Observado x estimado](resultados/figuras/figura_obs_vs_estimado.png)

### Tabela 4: previsores intrassafra (janela expansiva)

| Previsor | Safras de teste | RF | Só controles | Persistência M(t) | p (RF × persistência) |
|---|---|---|---|---|---|
| Seca | 2014 a 2017 (N = 40) | 4,16 | 4,27 | 4,21 | > 0,99 |
| Geada | 1994, 1995, 2000, 2001, 2021, 2022, 2023 (N = 70) | 6,90 | 7,28 | 5,91 | 0,98 |
| Geada, sem 2000 e 2001* | (N = 50) | 4,61 | 5,87 | 5,71 | 0,015 |

\* Exclusão definida depois de ver os resultados (safras com valores repetidos no IBGE); vale só como sensibilidade.

Os intervalos P10-P90 cobriram 75% (seca) e 70% (geada) dos valores reais, abaixo dos 80% esperados. Erro percentual médio do previsor de seca: 14,4%.

---

## 9. Experimentos e versões anteriores

A pasta [`experimentos/`](experimentos/README.md) guarda os testes que guiaram as escolhas do desenho final:

* **e1:** generalização para municípios inteiros fora do treino e para o bloco 2014-2023;
* **e2:** por que o K-Fold mede "completar a série" (só a tendência já erra 4,33; com a safra inteira fora e hiperparâmetros fixos, o RF erra 6,52; com a busca em grade da Tabela 1, 6,05);
* **e3:** residualização da tendência e da bienalidade antes do clima;
* **e4:** o balanço hídrico no modelo principal (reduz o erro do modelo completo, mas não o deixa melhor que o modelo só com controles).

Em `experimentos/versoes_anteriores/` estão os códigos de fases anteriores (cenários separados, Sentinel-2, CHIRPS, a primeira baseline com vazamento etc.), com uma tabela explicando o que cada um fazia e por que foi substituído.

---

## 10. Limitações

* Em 9 dos 10 municípios o IBGE repete o rendimento por 2 a 4 safras, sobretudo em 1999-2002; a PAM consolida estimativas de informantes locais.
* O K-Fold não mede a previsão de safras futuras; há poucos eventos extremos (uma seca e três geadas).
* Os municípios de uma mesma safra compartilham o clima, por isso o teste por safra é o mais honesto.
* O rendimento até 2012 é de café total; a partir de 2013, de café arábica.
* O ERA5-Land, com ~11 km, suaviza a temperatura mínima; a máscara de café do clima é fixa.
* O balanço hídrico usa um ponto na sede, e não a área de café.
* Os limites municipais da extração (FAO/GAUL 2015) diferem da malha do IBGE usada nos mapas.
* MDI e SHAP descrevem como o modelo **usou** as variáveis no ajuste, e não relações de causa.

O protocolo é replicável com dados públicos e tende a melhorar com dados mais precisos de produtividade e de clima, como os de cooperativas e de estações meteorológicas.

---

## 11. Reprodutibilidade

* Semente fixa (`random_state = 42`) em todos os modelos e divisões; os números se repetem a cada execução com as versões de `requirements.txt`.
* Correções de novembro de 2026: (i) a janela de enchimento passou a ser de dezembro de t−1 a abril de t (`1b2`); (ii) os pontos usados para achar o polígono de cada município no GAUL foram trocados por pontos bem dentro do município, e o script agora confere o nome do polígono achado (as coordenadas antigas caíam em municípios vizinhos em quatro casos); (iii) as sedes de Itamogi e Juruaia no balanço hídrico foram corrigidas. Os arquivos de `dados/brutos/` já estão corrigidos.
* A etapa 5a reproduz as faixas de estabilidade por fase com a ordem de colunas usada na checagem original (comentado no script).
* As etapas de coleta (1a a 1d) dependem de internet e do Earth Engine e foram rodadas no Colab; os arquivos que elas geram são os de `dados/brutos/`, usados no TCC.

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

**Como citar:** SILVA, L. F. V. D. *Sensoriamento remoto e aprendizado de máquina na estimativa da produtividade e do dano climático no café arábica*. Trabalho de Conclusão de Curso (Engenharia Agronômica). IFSULDEMINAS, Campus Muzambinho, 2026.
