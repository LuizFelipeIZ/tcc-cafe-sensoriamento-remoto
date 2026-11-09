"""
=====================================================================
TCC - Sensibilidade da Produção Cafeeira a Anomalias Climáticas
Pipeline Analítico Final (versão corrigida, sem vazamento de dados)
=====================================================================

O que este script faz, em ordem:
  1. Carrega a matriz fenológica e corrige o cálculo da bienalidade
     (shift(1) ANTES do rolling, para eliminar vazamento do alvo).
  2. Avalia a baseline ingênua (média móvel de 3 safras).
  3. Treina o Random Forest com Nested Cross-Validation (K-Fold k=5
     externo, GridSearchCV k=3 interno) para obter previsões 100% cegas.
  4. Roda duas validações adicionais mais rigorosas:
       - GroupKFold por município (generalização espacial)
       - Holdout temporal em bloco (treino < 2022, teste >= 2022)
  5. Extrai a Feature Importance (MDI) e valida com SHAP (agnóstico
     a multicolinearidade) + PDP para Tmin_Abs (efeito de limiar).
  6. Decompõe o desempenho por regime (anos normais vs. anos de choque
     climático: 2015/2016/2022/2023).
  7. Gera os gráficos finais e exporta a planilha consolidada.

Requisitos:
    pip install pandas numpy scikit-learn matplotlib shap openpyxl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.inspection import PartialDependenceDisplay
from scipy import stats
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.width', 140)

# =====================================================================
# CONFIGURAÇÃO
# =====================================================================
ARQUIVO_ENTRADA = 'TCC_Matriz_Fenologica_final.xlsx'
COL_MUNICIPIO = 'Municipio'
COL_ANO = 'Ano'
COL_ALVO = 'Rendimento_kg_ha'
ANOS_CHOQUE = [2015, 2016, 2022, 2023]          # seca 2014 -> impacto 15/16 | geada 2021 -> impacto 22/23
MUNICIPIOS_GRAFICO = ['Cabo Verde', 'Muzambinho', 'Guaxupe', 'Nova Resende']
ANO_CORTE_TEMPORAL = 2022                        # holdout: treino < corte, teste >= corte

VARIAVEIS_X = [
    'anomalia_bienal_t_1', 'Tmin_Abs', 'SPI_Anual', 'Dias_Abaixo_3C',
    'tendencia_temporal',            # tendência tecnológica/manejo ao longo dos anos
    'NDVI_Vigor_Maximo', 'NDVI_Estresse_Maximo', 'NDVI_Florada', 'NDVI_Amplitude'
]

# NOTA METODOLÓGICA: uma variável de interação Severidade_Geada
# (Tmin_Abs x Dias_Abaixo_3C) foi testada em rodada anterior e obteve
# importância desprezível (MDI = 0,02%; SHAP = 0,08%), sem ganho de
# desempenho preditivo. Foi removida do modelo final por parcimônia.
# Resultado nulo reportado na Metodologia/Discussão como teste de
# hipótese formalmente conduzido, não como omissão silenciosa.

PARAM_GRID = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],            # conservador, para evitar overfitting com N pequeno
    'min_samples_split': [5, 10]
}


# =====================================================================
# 1. PREPARO DOS DADOS (bienalidade sem vazamento)
# =====================================================================
def preparar_dados(caminho_arquivo):
    df = pd.read_excel(caminho_arquivo)
    df = df.sort_values(by=[COL_MUNICIPIO, COL_ANO]).reset_index(drop=True)

    df['produtividade_t_1'] = df.groupby(COL_MUNICIPIO)[COL_ALVO].shift(1)

    # CORREÇÃO CRÍTICA: shift(1) é aplicado ANTES do rolling.
    # Sem isso, a janela do rolling() inclui a própria observação do ano
    # corrente, vazando o alvo para dentro da "baseline ingênua" e para
    # dentro de anomalia_bienal_t_1 (usada como preditor no modelo).
    df['media_movel_3anos'] = df.groupby(COL_MUNICIPIO)[COL_ALVO].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['anomalia_bienal_t_1'] = df['produtividade_t_1'] - df['media_movel_3anos']

    # --- NOVA FEATURE 1: SPI (Standardized Precipitation Index) ---
    # Método McKee et al. (1993): ajusta uma distribuição Gama à precipitação
    # anual (pool regional, dado N pequeno por município) e transforma o
    # percentil de cada observação em um z-score da normal padrão. Mais
    # robusto que precipitação bruta porque normaliza para a variabilidade
    # histórica da série, não apenas o volume absoluto.
    precip = df['Precip_Anual'].values
    shape, loc, scale = stats.gamma.fit(precip, floc=0)
    percentis = stats.gamma.cdf(precip, shape, loc=loc, scale=scale)
    percentis = np.clip(percentis, 1e-6, 1 - 1e-6)  # evita +-inf no ppf
    df['SPI_Anual'] = stats.norm.ppf(percentis)

    # --- NOVA FEATURE 2: Tendência temporal (tecnologia/manejo) ---
    # Isola o ganho de produtividade por melhoria varietal e de manejo ao
    # longo do tempo (Schlenker & Roberts, 2009; Lobell & Burke, 2010),
    # evitando que o modelo confunda "safra mais recente" com "safra de
    # clima favorável".
    df['tendencia_temporal'] = df[COL_ANO] - df[COL_ANO].min()

    df_treino = df.dropna(subset=VARIAVEIS_X + [COL_ALVO, 'media_movel_3anos']).copy()
    df_treino = df_treino.reset_index(drop=True)

    print(f"[INFO] Amostra efetiva N = {len(df_treino)} observações "
          f"(de {len(df)} linhas brutas).")
    print(f"[INFO] Municípios: {df_treino[COL_MUNICIPIO].nunique()} | "
          f"Anos: {sorted(df_treino[COL_ANO].unique())}")

    return df_treino


# =====================================================================
# 2. BASELINE INGÊNUA
# =====================================================================
def avaliar_baseline(df):
    y = df[COL_ALVO]
    y_pred = df['media_movel_3anos']

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    print("\n" + "=" * 70)
    print(" 1. BASELINE INGÊNUA (Média Móvel de 3 Safras, sem vazamento)")
    print("=" * 70)
    print(f"R²:   {r2:.3f}")
    print(f"RMSE: {rmse:.2f} kg/ha")
    print(f"MAE:  {mae:.2f} kg/ha (~{mae/60:.2f} sacas/ha)")

    return {'r2': r2, 'rmse': rmse, 'mae': mae}


# =====================================================================
# 3. RANDOM FOREST COM NESTED CROSS-VALIDATION (K-Fold k=5)
# =====================================================================
def treinar_nested_cv(df):
    X = df[VARIAVEIS_X]
    y = df[COL_ALVO]

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros(len(y))

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                           cv=inner_cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs.fit(X_train, y_train)
        y_pred[test_idx] = gs.best_estimator_.predict(X_test)

    df['Pred_KFold'] = y_pred

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    print("\n" + "=" * 70)
    print(" 2. RANDOM FOREST - NESTED CV (K-Fold k=5 externo, GridSearch k=3 interno)")
    print("=" * 70)
    print(f"R²:   {r2:.3f}")
    print(f"RMSE: {rmse:.2f} kg/ha")
    print(f"MAE:  {mae:.2f} kg/ha (~{mae/60:.2f} sacas/ha)")

    return {'r2': r2, 'rmse': rmse, 'mae': mae}


# =====================================================================
# 4a. VALIDAÇÃO ADICIONAL: GroupKFold por Município (generalização espacial)
# =====================================================================
def validar_groupkfold(df):
    X = df[VARIAVEIS_X]
    y = df[COL_ALVO]
    groups = df[COL_MUNICIPIO]

    gkf = GroupKFold(n_splits=5)
    y_pred = np.zeros(len(y))

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                           cv=inner_cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs.fit(X_train, y_train)
        y_pred[test_idx] = gs.best_estimator_.predict(X_test)

    df['Pred_GroupKFold'] = y_pred

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print("\n" + "=" * 70)
    print(" 3. RANDOM FOREST - GroupKFold por Município (generalização espacial)")
    print("=" * 70)
    print("Municípios inteiros ficam fora do treino em cada dobra - nenhum dado")
    print("do município-teste é visto em nenhum ano. Teste mais rigoroso de")
    print("generalização espacial do que o K-Fold aleatório padrão.")
    print(f"R²:   {r2:.3f}")
    print(f"MAE:  {mae:.2f} kg/ha (~{mae/60:.2f} sacas/ha)")

    return {'r2': r2, 'mae': mae}


# =====================================================================
# 4b. VALIDAÇÃO ADICIONAL: Holdout Temporal em Bloco
# =====================================================================
def validar_holdout_temporal(df, ano_corte=ANO_CORTE_TEMPORAL):
    X = df[VARIAVEIS_X]
    y = df[COL_ALVO]

    mask_treino = df[COL_ANO] < ano_corte
    mask_teste = df[COL_ANO] >= ano_corte

    X_tr, y_tr = X[mask_treino], y[mask_treino]
    X_te, y_te = X[mask_teste], y[mask_teste]

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                       cv=inner_cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    gs.fit(X_tr, y_tr)
    y_pred = gs.best_estimator_.predict(X_te)

    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)

    print("\n" + "=" * 70)
    print(f" 4. RANDOM FOREST - Holdout Temporal em Bloco "
          f"(treino < {ano_corte}, teste >= {ano_corte})")
    print("=" * 70)
    print("Nenhum dado do período de teste é visto no treino - simula o cenário")
    print("real de prever a recuperação pós-choque usando apenas dados anteriores.")
    print(f"N treino: {len(y_tr)} | N teste: {len(y_te)}")
    print(f"R²:   {r2:.3f}")
    print(f"MAE:  {mae:.2f} kg/ha (~{mae/60:.2f} sacas/ha)")

    return {'r2': r2, 'mae': mae, 'n_treino': len(y_tr), 'n_teste': len(y_te)}


# =====================================================================
# 5. FEATURE IMPORTANCE (MDI) + SHAP + PDP
# =====================================================================
def analisar_atribuicao(df):
    X = df[VARIAVEIS_X]
    y = df[COL_ALVO]

    # Modelo final treinado na base completa (para extração de importância)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                       cv=inner_cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    gs.fit(X, y)
    rf_final = gs.best_estimator_
    rf_final.fit(X, y)

    # --- MDI (Mean Decrease in Impurity) ---
    importancias = pd.DataFrame({
        'Variavel': VARIAVEIS_X,
        'Importancia_Perc': rf_final.feature_importances_ * 100
    }).sort_values('Importancia_Perc', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print(" 5. FEATURE IMPORTANCE (MDI) - Hierarquia de Atribuição de Dano")
    print("=" * 70)
    print(f"Melhores parâmetros (base completa): {gs.best_params_}")
    print(importancias.to_string(index=False))

    # --- SHAP (validação agnóstica, robusta a multicolinearidade) ---
    explainer = shap.TreeExplainer(rf_final)
    shap_values = explainer.shap_values(X)

    shap_imp = pd.DataFrame({
        'Variavel': VARIAVEIS_X,
        'SHAP_mean_abs': np.abs(shap_values).mean(axis=0)
    }).sort_values('SHAP_mean_abs', ascending=False).reset_index(drop=True)
    shap_imp['SHAP_Perc'] = shap_imp['SHAP_mean_abs'] / shap_imp['SHAP_mean_abs'].sum() * 100

    comparativo = importancias.merge(shap_imp[['Variavel', 'SHAP_Perc']], on='Variavel')
    comparativo['Rank_MDI'] = comparativo['Importancia_Perc'].rank(ascending=False).astype(int)
    comparativo['Rank_SHAP'] = comparativo['SHAP_Perc'].rank(ascending=False).astype(int)
    comparativo = comparativo.sort_values('Rank_MDI')

    corr_rank = comparativo['Rank_MDI'].corr(comparativo['Rank_SHAP'], method='spearman')

    print("\n" + "=" * 70)
    print(" 5b. VALIDAÇÃO CONVERGENTE VIA SHAP (TreeExplainer)")
    print("=" * 70)
    print(comparativo.to_string(index=False))
    print(f"\nCorrelação de Spearman entre rankings MDI e SHAP: {corr_rank:.3f}")
    if corr_rank > 0.7:
        print("-> Alta convergência: hierarquia de atribuição confirmada por método")
        print("   agnóstico, não é artefato do MDI nem da multicolinearidade entre NDVIs.")

    # Gráfico SHAP summary
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # PDP para Tmin_Abs (efeito de limiar / não linearidade)
    fig, ax = plt.subplots(figsize=(8, 5))
    PartialDependenceDisplay.from_estimator(rf_final, X, ['Tmin_Abs'], ax=ax)
    ax.set_title('Efeito Marginal da Temperatura Mínima Absoluta\n'
                  'sobre o Rendimento Previsto (PDP)', fontweight='bold')
    ax.set_ylabel('Efeito parcial no Rendimento (kg/ha)')
    plt.tight_layout()
    plt.savefig('pdp_tmin_abs.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n[OK] Gráficos salvos: shap_summary.png, pdp_tmin_abs.png")

    return rf_final, importancias, comparativo


# =====================================================================
# 6. DECOMPOSIÇÃO POR REGIME (anos normais vs. anos de choque)
# =====================================================================
def decompor_por_regime(df):
    df['regime'] = np.where(df[COL_ANO].isin(ANOS_CHOQUE), 'Choque', 'Normal')

    print("\n" + "=" * 70)
    print(" 6. DECOMPOSIÇÃO POR REGIME (Baseline vs. RF, previsões K-Fold)")
    print("=" * 70)

    linhas = []
    for regime in ['Normal', 'Choque']:
        sub = df[df['regime'] == regime]
        r2_b = r2_score(sub[COL_ALVO], sub['media_movel_3anos'])
        mae_b = mean_absolute_error(sub[COL_ALVO], sub['media_movel_3anos'])
        r2_r = r2_score(sub[COL_ALVO], sub['Pred_KFold'])
        mae_r = mean_absolute_error(sub[COL_ALVO], sub['Pred_KFold'])
        linhas.append({
            'Regime': regime, 'N': len(sub),
            'R2_Baseline': r2_b, 'MAE_Baseline_sacas': mae_b / 60,
            'R2_RF': r2_r, 'MAE_RF_sacas': mae_r / 60,
        })

    df_regime = pd.DataFrame(linhas)
    print(df_regime.to_string(index=False))

    print("\n[INTERPRETAÇÃO]")
    for row in linhas:
        delta = row['MAE_Baseline_sacas'] - row['MAE_RF_sacas']
        vencedor = 'RF' if delta > 0 else 'Baseline'
        print(f" - Regime {row['Regime']}: {vencedor} tem menor erro "
              f"(diferença de {abs(delta):.2f} sacas/ha)")

    # --- Teste de Wilcoxon pareado: RF vs. Baseline, erros absolutos (geral) ---
    erros_rf = np.abs(df[COL_ALVO] - df['Pred_KFold'])
    erros_base = np.abs(df[COL_ALVO] - df['media_movel_3anos'])
    stat, p_valor = wilcoxon(erros_base, erros_rf)

    print("\n[TESTE DE SIGNIFICÂNCIA - Wilcoxon signed-rank pareado]")
    print(f"Comparação: erro absoluto Baseline vs. erro absoluto RF (N={len(df)})")
    print(f"Estatística W = {stat:.2f} | p-valor = {p_valor:.4f}")
    if p_valor < 0.05:
        print("-> O RF tem erro absoluto significativamente diferente da baseline (p<0.05).")
    else:
        print("-> Não há diferença estatisticamente significativa entre os erros (p>=0.05) -")
        print("   reportar com cautela, especialmente dado N=110.")

    return df_regime, {'wilcoxon_stat': stat, 'wilcoxon_p': p_valor}


# =====================================================================
# 7. GRÁFICO FINAL: Real x RF x Baseline nos anos pós-anomalia
# =====================================================================
def gerar_grafico_tres_series(df):
    df_f = df[(df[COL_MUNICIPIO].isin(MUNICIPIOS_GRAFICO)) &
              (df[COL_ANO].isin(ANOS_CHOQUE))].copy()
    df_f = df_f.sort_values(by=[COL_MUNICIPIO, COL_ANO])

    df_f['Real_sacas'] = df_f[COL_ALVO] / 60
    df_f['RF_sacas'] = df_f['Pred_KFold'] / 60
    df_f['Baseline_sacas'] = df_f['media_movel_3anos'] / 60

    x = np.arange(len(df_f))
    w = 0.27

    fig, ax = plt.subplots(figsize=(19, 7))
    b1 = ax.bar(x - w, df_f['Real_sacas'], w, label='IBGE (Real)', color='#0B3B24')
    b2 = ax.bar(x, df_f['RF_sacas'], w, label='Random Forest (Cego, K-Fold)', color='#11CAA0')
    b3 = ax.bar(x + w, df_f['Baseline_sacas'], w, label='Baseline (Média Móvel Ingênua)', color='#B0B0B0')

    for b, c in [(b1, '#0B3B24'), (b2, '#087B60'), (b3, '#707070')]:
        ax.bar_label(b, padding=2, fmt='%.1f', fontsize=8, fontweight='bold', color=c)

    max_y = df_f[['Real_sacas', 'RF_sacas', 'Baseline_sacas']].max().max()
    ax.set_ylim(0, max_y * 1.18)

    rotulos = [f"{m}\n({a})" for m, a in zip(df_f[COL_MUNICIPIO], df_f[COL_ANO])]
    ax.set_title('Real vs. Modelo vs. Baseline Ingênua nos Anos Pós-Anomalia\n'
                  '(Seca 2015/16 e Geada 2022/23)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Município e Ano Pós-Estresse', fontsize=12)
    ax.set_ylabel('Rendimento (sacas/ha)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    mun_atual = df_f[COL_MUNICIPIO].iloc[0]
    for i, m in enumerate(df_f[COL_MUNICIPIO]):
        if m != mun_atual:
            ax.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5)
            mun_atual = m

    plt.tight_layout()
    plt.savefig('grafico_tres_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n[OK] Gráfico de três séries salvo: grafico_tres_series.png")


# =====================================================================
# 8. EXPORTAÇÃO FINAL
# =====================================================================
def exportar_planilha(df):
    colunas = [COL_ANO, COL_MUNICIPIO, COL_ALVO, 'Pred_KFold', 'Pred_GroupKFold',
               'media_movel_3anos', 'regime']
    df_exp = df[colunas].copy()
    df_exp['Rendimento_sacas_ha'] = df_exp[COL_ALVO] / 60
    df_exp['RF_KFold_sacas'] = df_exp['Pred_KFold'] / 60
    df_exp['RF_GroupKFold_sacas'] = df_exp['Pred_GroupKFold'] / 60
    df_exp['Baseline_sacas'] = df_exp['media_movel_3anos'] / 60

    nome_saida = 'TCC_Resultados_Corrigidos_Completo.xlsx'
    df_exp.to_excel(nome_saida, index=False, float_format="%.2f")
    print(f"[OK] Planilha consolidada exportada: {nome_saida}")


# =====================================================================
# EXECUÇÃO
# =====================================================================
if __name__ == "__main__":
    try:
        df = preparar_dados(ARQUIVO_ENTRADA)

        resultado_baseline = avaliar_baseline(df)
        resultado_kfold = treinar_nested_cv(df)
        resultado_group = validar_groupkfold(df)
        resultado_temporal = validar_holdout_temporal(df)

        rf_final, importancias, comparativo_shap = analisar_atribuicao(df)
        df_regime, resultado_wilcoxon = decompor_por_regime(df)

        gerar_grafico_tres_series(df)
        exportar_planilha(df)

        print("\n" + "=" * 70)
        print(" RESUMO EXECUTIVO")
        print("=" * 70)
        print(f"Baseline ingênua:        R² = {resultado_baseline['r2']:.3f} | "
              f"MAE = {resultado_baseline['mae']/60:.2f} sacas/ha")
        print(f"RF (K-Fold padrão):      R² = {resultado_kfold['r2']:.3f} | "
              f"MAE = {resultado_kfold['mae']/60:.2f} sacas/ha")
        print(f"RF (GroupKFold espacial):R² = {resultado_group['r2']:.3f} | "
              f"MAE = {resultado_group['mae']/60:.2f} sacas/ha")
        print(f"RF (Holdout temporal):   R² = {resultado_temporal['r2']:.3f} | "
              f"MAE = {resultado_temporal['mae']/60:.2f} sacas/ha  "
              f"(N treino={resultado_temporal['n_treino']}, N teste={resultado_temporal['n_teste']})")
        print("=" * 70)

    except FileNotFoundError:
        print(f"\n[ERRO] O arquivo '{ARQUIVO_ENTRADA}' não foi encontrado.")
        print("Coloque a planilha na mesma pasta do script, ou ajuste ARQUIVO_ENTRADA.")
