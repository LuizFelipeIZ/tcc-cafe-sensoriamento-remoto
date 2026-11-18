# Experimentos e versões anteriores

Esta pasta guarda os testes que ajudaram a chegar ao desenho final do TCC. Eles não são necessários para reproduzir as tabelas do texto (isso está nas etapas 2 a 5), mas explicam **por que** algumas escolhas foram feitas.

## Testes com a base final (rodam com `python experimentos/<arquivo>.py`)

| Script | Pergunta | O que mostrou (MAE em sacas/ha) |
|---|---|---|
| `e1_validacao_espacial_e_temporal.py` | O modelo funciona num município que não viu? E num bloco de safras futuras? | Com 2 municípios inteiros fora do treino, 4,44 (perto do K-Fold, 4,69). No bloco 2014-2023, 4,28 contra 4,85 da persistência, mas com R² ≈ 0: acerta o nível do período, não a variação entre safras. |
| `e2_kfold_por_safra.py` | O que o K-Fold sorteado mede de verdade? | Só a tendência já erra 4,33, e a média dos vizinhos da mesma safra erra 4,34. Com a safra inteira fora do treino e hiperparâmetros fixos, o RF erra 6,52, mais que a persistência (6,26); com a busca em grade da Tabela 1, 6,05, sem diferença significativa dela. Ou seja, o K-Fold mede a capacidade de **completar a série**, não de prever o futuro. |
| `e3_residualizacao.py` | Tirando tendência e bienalidade por regressão, o clima explica o que sobra? | Clima e NDVI explicam cerca de 31% do resíduo (38% sem 1991-2002), com a florada no topo. Mas a combinação linear + RF erra mais (4,87) que o RF só com controles (4,54). |
| `e4_deficit_hidrico_no_modelo.py` | Vale incluir o balanço hídrico no modelo principal? | No modelo completo, 4,69 → 4,55 (p = 0,012 por safra), mas sem superar o modelo só com controles (4,54); nos controles, o déficit piora o erro (4,73). Ficou só no previsor de seca. |

Os resultados ficam em `resultados/e*.json`.

## `versoes_anteriores/`

Códigos de fases anteriores do projeto, guardados como registro. **Eles não rodam com os dados atuais** sem ajuste (esperam planilhas com outros nomes) e alguns usam fontes que foram trocadas. A tabela diz o que cada um fazia e o que mudou.

| Arquivo | O que fazia | Por que foi substituído |
|---|---|---|
| `v01_baseline_bienalidade_com_vazamento.py` | Baseline de média móvel e anomalia bienal | A média móvel incluía a própria safra (`rolling` sem `shift`): **vazamento** do alvo |
| `v02_baseline_bienalidade_sem_vazamento.py` | Mesma coisa, corrigida com `shift(1)` antes do `rolling` | Incorporado à etapa 3 |
| `v03_ndvi_landsat8_eventos_ponto_ssp.py` | NDVI do Landsat 8 num raio em torno de São Sebastião do Paraíso, só nos eventos de 2014 e 2021 | Passou a ser por município, toda a série e três sensores (etapa 1c) |
| `v04_simulador_previsao_pos_geada.py` | Primeira "previsão cega": treina com o passado e estima 2015, 2016, 2022 e 2023 | Virou a janela expansiva dos previsores (etapa 4b) |
| `v05_pipeline_dois_cenarios.py` | Dois modelos: só clima (1988-2023) e clima + NDVI (só anos recentes) | Com o NDVI do Landsat calibrado, um único modelo cobre 1989-2023 |
| `v06_thornthwaite_original.py` | Balanço hídrico, um pedido de 37 anos por município | A API pública recusava (erro 429) e alguns municípios ficavam sem dados |
| `v07_thornthwaite_download_em_blocos.py` | Download em blocos de 5 anos com novas tentativas | Incorporado à etapa 1d |
| `v08_thornthwaite_completa_blocos_faltantes.py` | Completa os blocos que falharam (Guaxupé 2002-2006, Monte Belo 2017-2021) | Incorporado à etapa 1d (as novas tentativas resolvem) |
| `v09_sidra_historico_1988_2012.py` | Produção de 1988 a 2012 pela API do SIDRA | Lia o ano pela posição errada da dimensão; corrigido na etapa 1a |
| `v10_clima_era5land_chirps_municipio.py` | Mínima do ERA5-Land e chuva do CHIRPS no polígono do município | Chuva e temperatura passaram a vir do ERA5-Land, só na área de café |
| `v11_gee_clima_e_ndvi_com_sentinel2.py` | Clima na área de café + NDVI Landsat calibrado com o Sentinel-2 | A parte de clima virou a etapa 1b; o Sentinel-2 só é consistente a partir de 2017 e saiu |
| `v12_ndvi_landsat_calibrado_sentinel2.py` | Calibração Landsat → Sentinel-2 por regressão comum | Trocada pela calibração entre Landsats pelo RMA (etapa 2b) |
| `v13_ndvi_landsat_corrigido_2_sensores.py` | NDVI com máscaras de nuvem e café, dois sensores | Ampliado para três sensores (etapa 1c) |
| `v14_pipeline_nested_cv_groupkfold_holdout.py` | K-Fold aninhado, GroupKFold, holdout temporal, SHAP e PDP | Organizado nas etapas 4a e 5 e no experimento e1 |
| `v15_pipeline_1988_2023_cenarios.py` | Mesma bateria de validação nos dois cenários | Idem |
| `v16_isolamento_tendencia_bienalidade.py` | Residualização em duas etapas | Virou o experimento e3 |
| `v17_alerta_seca_holdout_2014_2017.py` | Previsor de seca com 2014-2017 como bloco de teste | Trocado pela janela expansiva (etapa 4b) |
| `v18_previsores_seca_geada_walkforward.py` | Previsores de seca e de geada em janela expansiva | Completado com a comparação com o modelo só com controles (etapa 4b) |
| `v19_previsao_4_municipios_pos_evento.py` | Previsões pós-evento para Cabo Verde, Guaxupé, Muzambinho e Nova Resende | Análise de apoio, fora do texto final |
| `notebooks/` | Notebooks do Colab usados na extração (GEE) e na limpeza do IBGE | Registro do processo |
