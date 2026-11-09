"""
=====================================================================
TCC - Sensibilidade da Produção Cafeeira a Anomalias Climáticas
Pipeline Analítico Definitivo (1988-2023)
=====================================================================

Este script consome a matriz unificada de 35 anos.
Ele roda dois cenários de Machine Learning:
1. Modelo Histórico Longo (apenas clima, para capturar geadas antigas)
2. Modelo Fenológico Recente (clima + NDVI)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.inspection import PartialDependenceDisplay
from scipy import stats
import matplotlib.pyplot as plt
import shap
import warnings
import os

warnings.filterwarnings('ignore')
pd.set_option('display.width', 140)

# =====================================================================
# CONFIGURAÇÃO
# =====================================================================
ARQUIVO_ENTRADA = 'TCC_Matriz_Unificada_1988_2023_final.xlsx'
COL_MUNICIPIO = 'Municipio'
COL_ANO = 'Ano'
COL_ALVO = 'Rendimento_kg_ha'

# Anos de Choque Climático Severo para destaque
ANOS_CHOQUE = [1995, 2001, 2015, 2016, 2022, 2023] # Reflexos das geadas de 94, 00, 21 e seca de 14
MUNICIPIOS_GRAFICO = ['Cabo Verde', 'Muzambinho', 'Guaxupé', 'Nova Resende']

VARIAVEIS_CLIMA = [
    'anomalia_bienal_t_1', 'Tmin_Abs', 'SPI_Anual', 'Dias_Abaixo_3C', 'tendencia_temporal'
]

VARIAVEIS_NDVI = [
    'NDVI_Vigor_Maximo', 'NDVI_Estresse_Maximo', 'NDVI_Florada', 'NDVI_Amplitude'
]

PARAM_GRID = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'min_samples_split': [5, 10]
}

# =====================================================================
# 1. PREPARO DOS DADOS (isolamento da bienalidade)
# =====================================================================
def preparar_dados(caminho_arquivo):
    print("=========================================================")
    print(" 🛠️ INICIANDO PROCESSAMENTO DA MATRIZ UNIFICADA")
    print("=========================================================")

    df = pd.read_excel(caminho_arquivo)
    df = df.sort_values(by=[COL_MUNICIPIO, COL_ANO]).reset_index(drop=True)

    # Isolamento da bienalidade sem vazamento de dados (shift antes do rolling)
    df['produtividade_t_1'] = df.groupby(COL_MUNICIPIO)[COL_ALVO].shift(1)
    df['media_movel_3anos'] = df.groupby(COL_MUNICIPIO)[COL_ALVO].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['anomalia_bienal_t_1'] = df['produtividade_t_1'] - df['media_movel_3anos']

    # SPI (Standardized Precipitation Index)
    if 'Precip_Anual' in df.columns:
        precip = df['Precip_Anual'].fillna(df['Precip_Anual'].median()).values
        shape, loc, scale = stats.gamma.fit(precip, floc=0)
        percentis = stats.gamma.cdf(precip, shape, loc=loc, scale=scale)
        percentis = np.clip(percentis, 1e-6, 1 - 1e-6)
        df['SPI_Anual'] = stats.norm.ppf(percentis)
    else:
        df['SPI_Anual'] = 0

    # Tendência temporal (ajusta saltos tecnológicos entre 1988 e 2023)
    df['tendencia_temporal'] = df[COL_ANO] - df[COL_ANO].min()

    # Criando os dois recortes de dados
    df_clima = df.dropna(subset=VARIAVEIS_CLIMA + [COL_ALVO, 'media_movel_3anos']).copy()
    df_completo = df.dropna(subset=VARIAVEIS_CLIMA + VARIAVEIS_NDVI + [COL_ALVO, 'media_movel_3anos']).copy()

    print(f"✅ Cenário 1 (Histórico Clima): {len(df_clima)} registros válidos ({df_clima['Ano'].min()}-{df_clima['Ano'].max()})")
    print(f"✅ Cenário 2 (Moderno Clima+NDVI): {len(df_completo)} registros válidos ({df_completo['Ano'].min()}-{df_completo['Ano'].max()})")

    return df_clima, df_completo

# =====================================================================
# 2. MOTOR DE AVALIAÇÃO (Nested K-Fold)
# =====================================================================
def executar_pipeline_preditivo(df, vars_x, nome_cenario):
    print("\n" + "="*80)
    print(f" 🚀 EXECUTANDO: {nome_cenario}")
    print("="*80)

    X = df[vars_x]
    y = df[COL_ALVO]

    # Baseline Ingênua
    y_base = df['media_movel_3anos']
    r2_b = r2_score(y, y_base)
    mae_b = mean_absolute_error(y, y_base)
    print(f"📊 [BASELINE INGÊNUA] R²: {r2_b:.3f} | Erro Médio Absoluto: {mae_b/60:.2f} sacas/ha")

    # Nested CV
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_kfold = np.zeros(len(y))

    print("🤖 [RANDOM FOREST] Treinando árvores com Nested Cross-Validation (Cego)...")
    for train_idx, test_idx in outer_cv.split(X):
        gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_kfold[test_idx] = gs.predict(X.iloc[test_idx])

    r2_k = r2_score(y, y_pred_kfold)
    mae_k = mean_absolute_error(y, y_pred_kfold)
    df['Pred_RF'] = y_pred_kfold

    print(f"🎯 [MODELO IA CEGO] R²: {r2_k:.3f} | Erro Médio Absoluto: {mae_k/60:.2f} sacas/ha")

    # Treino Final para Extrair Feature Importance
    rf_final = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    rf_final.fit(X, y)
    best_rf = rf_final.best_estimator_

    importancias = pd.DataFrame({
        'Variavel': vars_x,
        'Importancia_Perc': best_rf.feature_importances_ * 100
    }).sort_values('Importancia_Perc', ascending=False)

    print("\n⚖️ HIERARQUIA DE ATRIBUIÇÃO DE DANO (Redução de Variância):")
    print(importancias.to_string(index=False))

    return best_rf, df, X

# =====================================================================
# 3. INTERPRETABILIDADE (SHAP e PDP)
# =====================================================================
def gerar_graficos_interpretabilidade(rf_model, X, sufixo):
    print(f"\n🖼️ Gerando gráficos de interpretabilidade ({sufixo})...")

    # PDP - Efeito Marginal da Temperatura Mínima
    if 'Tmin_Abs' in X.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        PartialDependenceDisplay.from_estimator(rf_model, X, ['Tmin_Abs'], ax=ax)
        ax.set_title(f'Ponto Crítico da Geada sobre a Safra ({sufixo})', fontweight='bold')
        ax.set_ylabel('Efeito no Rendimento (kg/ha)')
        plt.tight_layout()
        plt.savefig(f'TCC_PDP_Geada_{sufixo}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # SHAP - Impacto Global Agrupado
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'Impacto das Variáveis (SHAP Values) - {sufixo}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'TCC_SHAP_Impacto_{sufixo}.png', dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# 4. GRÁFICO COMPARATIVO PÓS-ANOMALIAS (Série Recente)
# =====================================================================
def gerar_grafico_comparativo(df):
    print("\n🖼️ Gerando gráfico de barras comparativo para o TCC...")
    df_f = df[(df[COL_MUNICIPIO].isin(MUNICIPIOS_GRAFICO)) &
              (df[COL_ANO].isin([2015, 2016, 2022, 2023]))].copy()

    if df_f.empty:
        print("⚠️ Dados recentes não encontrados para o gráfico.")
        return

    df_f = df_f.sort_values(by=[COL_MUNICIPIO, COL_ANO])

    df_f['Real_sacas'] = df_f[COL_ALVO] / 60
    df_f['RF_sacas'] = df_f['Pred_RF'] / 60
    df_f['Baseline_sacas'] = df_f['media_movel_3anos'] / 60

    x = np.arange(len(df_f))
    w = 0.27

    fig, ax = plt.subplots(figsize=(18, 7))
    b1 = ax.bar(x - w, df_f['Real_sacas'], w, label='Oficial IBGE (Real)', color='#0B3B24')
    b2 = ax.bar(x, df_f['RF_sacas'], w, label='Previsão Random Forest (IA)', color='#11CAA0')
    b3 = ax.bar(x + w, df_f['Baseline_sacas'], w, label='Preditor Inercial (Baseline)', color='#B0B0B0')

    for b, c in [(b1, '#0B3B24'), (b2, '#087B60'), (b3, '#707070')]:
        ax.bar_label(b, padding=2, fmt='%.1f', fontsize=9, fontweight='bold', color=c)

    max_y = df_f[['Real_sacas', 'RF_sacas', 'Baseline_sacas']].max().max()
    ax.set_ylim(0, max_y * 1.2)

    rotulos = [f"{m}\n({a})" for m, a in zip(df_f[COL_MUNICIPIO], df_f[COL_ANO])]
    ax.set_title('Recuperação Cafeeira Pós-Estresse: IA vs Inércia Histórica', fontsize=16, fontweight='bold')
    ax.set_xlabel('Município e Ano Pós-Estresse', fontsize=12)
    ax.set_ylabel('Rendimento Oficial (sacas/ha)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    mun_atual = df_f[COL_MUNICIPIO].iloc[0]
    for i, m in enumerate(df_f[COL_MUNICIPIO]):
        if m != mun_atual:
            ax.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5)
            mun_atual = m

    plt.tight_layout()
    plt.savefig('TCC_Grafico_Pós_Anomalias.png', dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    try:
        df_historico, df_moderno = preparar_dados(ARQUIVO_ENTRADA)

        # --- CENÁRIO 1: O TESTE DE FOGO HISTÓRICO (1988-2023) ---
        if not df_historico.empty:
            modelo_hist, df_res_hist, X_hist = executar_pipeline_preditivo(
                df_historico, VARIAVEIS_CLIMA, "CENÁRIO 1: MODELO CLIMÁTICO (35 ANOS - SEM NDVI)"
            )
            gerar_graficos_interpretabilidade(modelo_hist, X_hist, "Historico_Clima")

        # --- CENÁRIO 2: A PRECISÃO FENOLÓGICA (2013-2023) ---
        if not df_moderno.empty:
            modelo_mod, df_res_mod, X_mod = executar_pipeline_preditivo(
                df_moderno, VARIAVEIS_CLIMA + VARIAVEIS_NDVI, "CENÁRIO 2: MODELO MODERNO (CLIMA + SATÉLITE)"
            )
            gerar_graficos_interpretabilidade(modelo_mod, X_mod, "Moderno_Satelite")

            # Exportar os dados previstos do cenário moderno para o Excel do TCC
            exportar = df_res_mod[[COL_ANO, COL_MUNICIPIO, COL_ALVO, 'Pred_RF', 'media_movel_3anos']].copy()
            exportar['Real_Sacas'] = exportar[COL_ALVO] / 60
            exportar['IA_Sacas'] = exportar['Pred_RF'] / 60
            exportar.to_excel("TCC_Resultados_Previsoes_Sacas.xlsx", index=False, float_format="%.2f")

            gerar_grafico_comparativo(df_res_mod)

        print("\n=========================================================")
        print("🎉 PIPELINE FINALIZADO COM SUCESSO!")
        print("Verifique a aba esquerda para baixar os gráficos e planilhas.")
        print("=========================================================")

    except FileNotFoundError:
        print(f"\n❌ [ERRO] O arquivo '{ARQUIVO_ENTRADA}' não foi encontrado.")