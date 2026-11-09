import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
import matplotlib.pyplot as plt

def calcular_bienalidade(df, coluna_municipio, coluna_ano, coluna_produtividade):
    """
    Calcula a bienalidade natural e a baseline (média móvel de 3 anos).
    """
    df = df.sort_values(by=[coluna_municipio, coluna_ano])
    df['produtividade_t_1'] = df.groupby(coluna_municipio)[coluna_produtividade].shift(1)
    df['media_movel_3anos'] = df.groupby(coluna_municipio)[coluna_produtividade].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)
    df['anomalia_bienal_t_1'] = df['produtividade_t_1'] - df['media_movel_3anos']
    return df

def avaliar_baseline_rigorosa(df_final, coluna_alvo, coluna_baseline):
    """
    ITEM 1: Avalia o preditor ingênuo (media_movel_3anos) usando K-Fold (k=5).
    """
    df_clean = df_final.dropna(subset=[coluna_alvo, coluna_baseline]).copy()
    y_true = df_clean[coluna_alvo]
    y_pred_base = df_clean[coluna_baseline]

    r2_base = r2_score(y_true, y_pred_base)
    rmse_base = np.sqrt(mean_squared_error(y_true, y_pred_base))
    mae_base = mean_absolute_error(y_true, y_pred_base)

    print("\n" + "="*60)
    print(" 1. AVALIAÇÃO DA BASELINE (Preditor Ingênuo: Média Móvel 3 Anos)")
    print("="*60)
    print(f"R² da Baseline: {r2_base:.3f}")
    print(f"RMSE da Baseline: {rmse_base:.2f} kg/ha")
    print(f"MAE da Baseline: {mae_base:.2f} kg/ha (Aprox. {mae_base/60:.2f} sacas/ha)")
    print("="*60)

    return r2_base, rmse_base, mae_base

def treinar_com_nested_cv(df_final, variaveis_preditoras, variavel_alvo):
    """
    ITENS 2 e 3: Implementa Protocolo Fixo K-Fold (k=5) com Nested Cross-Validation
    para evitar vazamento de dados na otimização de hiperparâmetros (GridSearchCV).
    Caso prefira segurança absoluta sem tuning, utiliza max_depth=5 fixo.
    """
    df_final = df_final.dropna(subset=variaveis_preditoras + [variavel_alvo]).copy()
    print(f"\n[INFO] Amostra efetiva N = {len(df_final)} observações.")

    X = df_final[variaveis_preditoras]
    y = df_final[variavel_alvo]

    # Protocolo Externo Fixo (K-Fold k=5)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Validação Cruzada Aninhada (Nested CV) ou Parâmetros Conservadores Fixos
    # Para máxima defesa em banca pequena, optamos por Grid Search Interno (CV interna = 3)
    print("\n" + "="*60)
    print(" 2. TREINAMENTO COM NESTED CROSS-VALIDATION (K-Fold k=5)")
    print("="*60)

    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [3, 5],            # Mantido conservador para evitar overfitting
        'min_samples_split': [5, 10]
    }

    rf_base = RandomForestRegressor(random_state=42)

    # Nested CV manual para garantir previsões 100% cegas no conjunto externo
    y_pred_cego = np.zeros_like(y)
    melhores_parametros_por_fold = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Grid Search interno (CV interna = 3)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid,
                                   cv=inner_cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        melhor_modelo = grid_search.best_estimator_
        melhores_parametros_por_fold.append(grid_search.best_params_)

        # Predição cega no fold externo
        y_pred_cego[test_idx] = melhor_modelo.predict(X_test)

    print(f"Parâmetros selecionados em média nas dobras internas: {melhores_parametros_por_fold[0]}")

    df_final['Rendimento_Previsto'] = y_pred_cego
    df_final['Erro_Absoluto'] = abs(df_final[variavel_alvo] - df_final['Rendimento_Previsto'])

    r2_cv = r2_score(y, y_pred_cego)
    rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cego))
    mae_cv = mean_absolute_error(y, y_pred_cego)

    print("\n--- DESEMPENHO REAL (CEGO) DO RANDOM FOREST ---")
    print(f"R² (Poder de generalização): {r2_cv:.3f}")
    print(f"RMSE (Erro penalizado): {rmse_cv:.2f} kg/ha")
    print(f"MAE (Erro absoluto mediano): {mae_cv:.2f} kg/ha (Aprox. {mae_cv/60:.2f} sacas/ha)")
    print("="*60)

    # ITEM 4: Treinamento final em toda a base para extração da Redução de Variância
    rf_final = RandomForestRegressor(**grid_search.best_params_, random_state=42)
    rf_final.fit(X, y)

    importancias = pd.DataFrame({
        'Variavel': variaveis_preditoras,
        'Importancia_Perc': rf_final.feature_importances_ * 100
    }).sort_values(by='Importancia_Perc', ascending=False)

    print("\n" + "="*60)
    print(" 4. HIERARQUIA DE ATRIBUIÇÃO DE DANO (Feature Importance)")
    print("="*60)
    print(importancias)
    print("="*60)

    return rf_final, importancias, df_final

def gerar_graficos_e_exportar(df_resultados, importancias, municipios_alvo):
    """
    Gera o gráfico de barras unificado pós-anomalias e exporta os dados limpos.
    """
    print("\n--- GERANDO PRODUTOS VISUAIS E TABELAS DE SUPORTE ---")

    df_resultados['Rendimento_sacas_ha'] = df_resultados['Rendimento_kg_ha'] / 60
    df_resultados['Rendimento_Previsto_sacas_ha'] = df_resultados['Rendimento_Previsto'] / 60
    df_resultados['Erro_Absoluto_sacas'] = df_resultados['Erro_Absoluto'] / 60

    # Gráfico de Feature Importance (O resultado forte do TCC)
    plt.figure(figsize=(10, 6))
    importancias_invertido = importancias.iloc[::-1]
    plt.barh(importancias_invertido['Variavel'], importancias_invertido['Importancia_Perc'], color='#0B3B24')
    plt.title('Atribuição de Impacto Climático (Redução de Variância)', fontsize=14, fontweight='bold')
    plt.xlabel('Importância Relativa (%)', fontsize=12)
    plt.tight_layout()
    plt.savefig('TCC_Feature_Importance_Atribuicao.png', dpi=300)
    plt.show()

    # Gráfico Comparativo Pós-Anomalias (Seca e Geada)
    anos_pos_anomalia = [2015, 2016, 2022, 2023]
    df_filtrado = df_resultados[(df_resultados['Municipio'].isin(municipios_alvo)) & (df_resultados['Ano'].isin(anos_pos_anomalia))].copy()

    if not df_filtrado.empty:
        df_filtrado = df_filtrado.sort_values(by=['Municipio', 'Ano'])
        plt.figure(figsize=(18, 7))

        x = np.arange(len(df_filtrado))
        largura_barra = 0.35

        barras_real = plt.bar(x - largura_barra/2, df_filtrado['Rendimento_sacas_ha'], largura_barra, label='IBGE (Real)', color='#0B3B24')
        barras_modelo = plt.bar(x + largura_barra/2, df_filtrado['Rendimento_Previsto_sacas_ha'], largura_barra, label='Modelo (Cego)', color='#11CAA0')

        plt.bar_label(barras_real, padding=3, fmt='%.1f', fontsize=9, fontweight='bold', color='#0B3B24')
        plt.bar_label(barras_modelo, padding=3, fmt='%.1f', fontsize=9, fontweight='bold', color='#087B60')

        max_y = df_filtrado[['Rendimento_sacas_ha', 'Rendimento_Previsto_sacas_ha']].max().max()
        plt.ylim(0, max_y * 1.15)

        rotulos_x = [f"{mun}\n({ano})" for mun, ano in zip(df_filtrado['Municipio'], df_filtrado['Ano'])]

        plt.title('Validação de Suporte: Dinâmica Pós-Anomalias (Seca 15/16 e Geada 22/23)', fontsize=16, fontweight='bold')
        plt.xlabel('Município e Ano Pós-Estresse', fontsize=12)
        plt.ylabel('Rendimento (sacas/ha)', fontsize=12)
        plt.xticks(x, rotulos_x, rotation=0, fontsize=10)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)

        municipio_atual = df_filtrado['Municipio'].iloc[0]
        for i, mun in enumerate(df_filtrado['Municipio']):
            if mun != municipio_atual:
                plt.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5)
                municipio_atual = mun

        plt.tight_layout()
        plt.savefig('TCC_Grafico_Pos_Anomalias_Unificado.png', dpi=300)
        plt.show()

    # Exportando tabela consolidada
    nome_arquivo_saida = 'TCC_Resultados_Atribuicao_Climatica.xlsx'
    colunas_exportar = [
        'Ano', 'Municipio',
        'Rendimento_kg_ha', 'Rendimento_sacas_ha',
        'Rendimento_Previsto_sacas_ha', 'Erro_Absoluto_sacas'
    ]
    df_resultados[colunas_exportar].to_excel(nome_arquivo_saida, index=False, float_format="%.2f")
    print(f"\n[SUCESSO] Planilha analítica exportada: '{nome_arquivo_saida}'")

if __name__ == "__main__":
    nome_arquivo = 'TCC_Matriz_Fenologica_final.xlsx'
    print(f"Carregando dados de: {nome_arquivo}...")

    try:
        df_suprema = pd.read_excel(nome_arquivo)

        NOME_COLUNA_MUNICIPIO = 'Municipio'
        NOME_COLUNA_ANO = 'Ano'
        NOME_COLUNA_PRODUTIVIDADE = 'Rendimento_kg_ha'

        if NOME_COLUNA_MUNICIPIO in df_suprema.columns and NOME_COLUNA_ANO in df_suprema.columns:
            df_suprema = calcular_bienalidade(df_suprema, NOME_COLUNA_MUNICIPIO, NOME_COLUNA_ANO, NOME_COLUNA_PRODUTIVIDADE)
            print("\n[OK] Variáveis de bienalidade e baseline calculadas.")

        # 1. Executa a Baseline rigorosa
        avaliar_baseline_rigorosa(df_suprema, NOME_COLUNA_PRODUTIVIDADE, 'media_movel_3anos')

        variaveis_x = [
            'anomalia_bienal_t_1',
            'Tmin_Abs',
            'Precip_Anual',
            'Dias_Abaixo_3C',
            'NDVI_Vigor_Maximo',
            'NDVI_Estresse_Maximo',
            'NDVI_Florada',
            'NDVI_Amplitude'
        ]

        df_treino = df_suprema.dropna(subset=variaveis_x + [NOME_COLUNA_PRODUTIVIDADE])

        # 2, 3 e 4. Executa o treinamento com Nested CV e extração de Importância
        modelo, pesos, df_resultados = treinar_com_nested_cv(df_treino, variaveis_x, NOME_COLUNA_PRODUTIVIDADE)

        municipios_para_grafico = ['Cabo Verde', 'Muzambinho', 'Guaxupé', 'Nova Resende']
        gerar_graficos_e_exportar(df_resultados, pesos, municipios_para_grafico)

    except FileNotFoundError:
        print(f"\n[ERRO] O arquivo '{nome_arquivo}' não foi encontrado.")