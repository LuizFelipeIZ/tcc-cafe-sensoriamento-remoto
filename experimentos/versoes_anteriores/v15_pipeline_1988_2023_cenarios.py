"""
=====================================================================
TCC - Sensibilidade da Produção Cafeeira a Anomalias Climáticas
Pipeline Analítico Definitivo (1988-2023) - VERSÃO CORRIGIDA

Correção em relação à versão anterior: o Cenário 1 (clima puro,
35 anos) agora passa pela MESMA bateria de validação aplicada ao
Cenário 2 no pipeline original (K-Fold, GroupKFold, Holdout Temporal,
Wilcoxon, Decomposição por Regime). Sem isso, não é possível saber se
a extensão histórica realmente resolveu a fragilidade de generalização
temporal identificada anteriormente (R² = -1,313 no holdout 2022-2023).
=====================================================================
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
ARQUIVO_ENTRADA = 'TCC_Matriz_Mestra_FINAL_1988_2023.xlsx'
COL_MUNICIPIO = 'Municipio'
COL_ANO = 'Ano'
COL_ALVO = 'Rendimento_kg_ha'

# Anos de choque climático severo (safra do evento + 1, seguindo a mesma
# convenção usada para seca 2014 -> impacto 2015/16 e geada 2021 -> 2022/23)
ANOS_CHOQUE_HISTORICO = [1994, 1995, 2000, 2001]  # geadas de 94 e 00
ANOS_CHOQUE_RECENTE = [2015, 2016, 2022, 2023]     # seca 14 e geada 21
MUNICIPIOS_GRAFICO = ['Cabo Verde', 'Muzambinho', 'Guaxupe', 'Nova Resende']

# Corte para o holdout temporal do cenário histórico: testa exatamente
# nos anos da geada de 2000/2001, usando só o que era conhecido antes.
ANO_CORTE_HISTORICO = 2000
# Corte para o holdout temporal do cenário moderno (igual ao pipeline anterior)
ANO_CORTE_RECENTE = 2022

VARIAVEIS_CLIMA = [
    'anomalia_bienal_t_1', 'tendencia_temporal',
    'Tmin_Abs', 'SPI_Anual', 'Dias_Abaixo_3C',
    'Precip_Florada', 'Tmin_Abs_Florada',
    'Precip_Enchimento_Graos', 'Tmin_Abs_Enchimento_Graos',
    'Precip_Inverno_Risco_Geada', 'Tmin_Abs_Inverno_Risco_Geada',
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
# 1. PREPARO DOS DADOS
# =====================================================================
def preparar_dados(caminho_arquivo):
    print("=" * 70)
    print(" PROCESSAMENTO DA MATRIZ UNIFICADA")
    print("=" * 70)

    df = pd.read_excel(caminho_arquivo)
    df = df.sort_values(by=[COL_MUNICIPIO, COL_ANO]).reset_index(drop=True)

    df['produtividade_t_1'] = df.groupby(COL_MUNICIPIO)[COL_ALVO].shift(1)
    df['media_movel_3anos'] = df.groupby(COL_MUNICIPIO)[COL_ALVO].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['anomalia_bienal_t_1'] = df['produtividade_t_1'] - df['media_movel_3anos']

    # SPI ajustado sobre a série completa de precipitação. Isso é aceitável
    # aqui porque o SPI é definido pela prática meteorológica-padrão (McKee
    # et al., 1993) em relação a uma climatologia de referência fixa, e não
    # ao valor-alvo (y) que o modelo tenta prever — não há vazamento do
    # alvo. Ainda assim, para o holdout temporal, o ideal seria recalcular
    # o ajuste da gama apenas com o período de treino; ver nota na Seção
    # de limitações caso essa simplificação seja mantida no TCC.
    if 'Precip_Anual' in df.columns:
        precip = df['Precip_Anual'].fillna(df['Precip_Anual'].median()).values
        shape, loc, scale = stats.gamma.fit(precip, floc=0)
        percentis = stats.gamma.cdf(precip, shape, loc=loc, scale=scale)
        percentis = np.clip(percentis, 1e-6, 1 - 1e-6)
        df['SPI_Anual'] = stats.norm.ppf(percentis)
    else:
        df['SPI_Anual'] = 0

    df['tendencia_temporal'] = df[COL_ANO] - df[COL_ANO].min()

    df_clima = df.dropna(subset=VARIAVEIS_CLIMA + [COL_ALVO, 'media_movel_3anos']).copy().reset_index(drop=True)
    df_completo = df.dropna(subset=VARIAVEIS_CLIMA + VARIAVEIS_NDVI + [COL_ALVO, 'media_movel_3anos']).copy().reset_index(drop=True)

    print(f"Cenário 1 (Histórico, só Clima): {len(df_clima)} registros "
          f"({df_clima[COL_ANO].min()}-{df_clima[COL_ANO].max()})")
    print(f"Cenário 2 (Moderno, Clima+NDVI): {len(df_completo)} registros "
          f"({df_completo[COL_ANO].min()}-{df_completo[COL_ANO].max()})")

    return df_clima, df_completo


# =====================================================================
# 2. BATERIA COMPLETA DE VALIDAÇÃO (reutilizável para os dois cenários)
# =====================================================================
def rodar_bateria_validacao(df, vars_x, nome_cenario, ano_corte_temporal, anos_choque):
    print("\n" + "=" * 70)
    print(f" {nome_cenario}")
    print("=" * 70)

    X = df[vars_x]
    y = df[COL_ALVO]

    # --- Baseline ---
    y_base = df['media_movel_3anos']
    r2_b = r2_score(y, y_base)
    mae_b = mean_absolute_error(y, y_base)
    print(f"[BASELINE INGÊNUA]        R² = {r2_b:.3f} | MAE = {mae_b/60:.2f} sacas/ha")

    # --- Nested CV (K-Fold k=5) ---
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_kfold = np.zeros(len(y))
    for train_idx, test_idx in outer_cv.split(X):
        gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                           cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_kfold[test_idx] = gs.predict(X.iloc[test_idx])
    df['Pred_KFold'] = y_pred_kfold
    r2_k = r2_score(y, y_pred_kfold)
    mae_k = mean_absolute_error(y, y_pred_kfold)
    print(f"[RF - K-Fold padrão]      R² = {r2_k:.3f} | MAE = {mae_k/60:.2f} sacas/ha")

    # --- GroupKFold por município ---
    groups = df[COL_MUNICIPIO]
    gkf = GroupKFold(n_splits=5)
    y_pred_group = np.zeros(len(y))
    for train_idx, test_idx in gkf.split(X, y, groups):
        gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                           cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_group[test_idx] = gs.predict(X.iloc[test_idx])
    r2_g = r2_score(y, y_pred_group)
    mae_g = mean_absolute_error(y, y_pred_group)
    print(f"[RF - GroupKFold espacial] R² = {r2_g:.3f} | MAE = {mae_g/60:.2f} sacas/ha")

    # --- Holdout Temporal em Bloco (o teste que mais importa) ---
    mask_treino = df[COL_ANO] < ano_corte_temporal
    mask_teste = df[COL_ANO] >= ano_corte_temporal
    if mask_treino.sum() > 10 and mask_teste.sum() > 3:
        X_tr, y_tr = X[mask_treino], y[mask_treino]
        X_te, y_te = X[mask_teste], y[mask_teste]
        gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                           cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs.fit(X_tr, y_tr)
        y_pred_temp = gs.predict(X_te)
        r2_t = r2_score(y_te, y_pred_temp)
        mae_t = mean_absolute_error(y_te, y_pred_temp)
        print(f"[RF - Holdout Temporal]   R² = {r2_t:.3f} | MAE = {mae_t/60:.2f} sacas/ha "
              f"(treino<{ano_corte_temporal}, N treino={len(y_tr)}, N teste={len(y_te)})")
    else:
        print(f"[RF - Holdout Temporal]   [AVISO] dados insuficientes para o corte em {ano_corte_temporal}")
        r2_t, mae_t = None, None

    # --- Wilcoxon pareado ---
    erros_rf = np.abs(y - y_pred_kfold)
    erros_base = np.abs(y - y_base)
    try:
        stat, p_valor = wilcoxon(erros_base, erros_rf)
        print(f"[Wilcoxon RF vs Baseline] W = {stat:.2f} | p = {p_valor:.4f}")
    except ValueError as e:
        print(f"[Wilcoxon RF vs Baseline] Não foi possível calcular: {e}")

    # --- Decomposição por regime ---
    df['regime'] = np.where(df[COL_ANO].isin(anos_choque), 'Choque', 'Normal')
    print("\n[Decomposição por regime]")
    for regime in ['Normal', 'Choque']:
        sub = df[df['regime'] == regime]
        if len(sub) < 5:
            print(f"  Regime {regime}: N insuficiente ({len(sub)})")
            continue
        r2_br = r2_score(sub[COL_ALVO], sub['media_movel_3anos'])
        mae_br = mean_absolute_error(sub[COL_ALVO], sub['media_movel_3anos'])
        r2_rr = r2_score(sub[COL_ALVO], sub['Pred_KFold'])
        mae_rr = mean_absolute_error(sub[COL_ALVO], sub['Pred_KFold'])
        print(f"  Regime {regime} (N={len(sub)}): Baseline R²={r2_br:.3f}, MAE={mae_br/60:.2f} sacas | "
              f"RF R²={r2_rr:.3f}, MAE={mae_rr/60:.2f} sacas")

    # --- Feature Importance final + SHAP ---
    gs_full = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                            cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    gs_full.fit(X, y)
    rf_final = gs_full.best_estimator_
    rf_final.fit(X, y)

    importancias = pd.DataFrame({
        'Variavel': vars_x,
        'Importancia_Perc': rf_final.feature_importances_ * 100
    }).sort_values('Importancia_Perc', ascending=False)
    print("\n[Feature Importance - MDI]")
    print(importancias.to_string(index=False))

    resultados = {
        'baseline': {'r2': r2_b, 'mae': mae_b},
        'kfold': {'r2': r2_k, 'mae': mae_k},
        'groupkfold': {'r2': r2_g, 'mae': mae_g},
        'temporal': {'r2': r2_t, 'mae': mae_t},
    }

    return rf_final, importancias, df, X, resultados


# =====================================================================
# 3. INTERPRETABILIDADE (SHAP + PDP)
# =====================================================================
def gerar_graficos_interpretabilidade(rf_model, X, sufixo):
    print(f"\n[Gerando gráficos de interpretabilidade: {sufixo}]")

    if 'Tmin_Abs' in X.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        PartialDependenceDisplay.from_estimator(rf_model, X, ['Tmin_Abs'], ax=ax)
        ax.set_title(f'Efeito Marginal da Temperatura Mínima ({sufixo})', fontweight='bold')
        ax.set_ylabel('Efeito no Rendimento (kg/ha)')
        plt.tight_layout()
        plt.savefig(f'TCC_PDP_Geada_{sufixo}.png', dpi=300, bbox_inches='tight')
        plt.close()

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'SHAP Values - {sufixo}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'TCC_SHAP_Impacto_{sufixo}.png', dpi=300, bbox_inches='tight')
    plt.close()


# =====================================================================
# 4. GRÁFICO COMPARATIVO (Real x RF x Baseline)
# =====================================================================
def gerar_grafico_comparativo(df, anos_alvo, sufixo):
    df_f = df[(df[COL_MUNICIPIO].isin(MUNICIPIOS_GRAFICO)) & (df[COL_ANO].isin(anos_alvo))].copy()
    if df_f.empty:
        print(f"[AVISO] Sem dados para o gráfico comparativo ({sufixo}).")
        return

    df_f = df_f.sort_values(by=[COL_MUNICIPIO, COL_ANO])
    df_f['Real_sacas'] = df_f[COL_ALVO] / 60
    df_f['RF_sacas'] = df_f['Pred_KFold'] / 60
    df_f['Baseline_sacas'] = df_f['media_movel_3anos'] / 60

    x = np.arange(len(df_f))
    w = 0.27
    fig, ax = plt.subplots(figsize=(18, 7))
    b1 = ax.bar(x - w, df_f['Real_sacas'], w, label='IBGE (Real)', color='#0B3B24')
    b2 = ax.bar(x, df_f['RF_sacas'], w, label='Random Forest (Cego)', color='#11CAA0')
    b3 = ax.bar(x + w, df_f['Baseline_sacas'], w, label='Baseline Inercial', color='#B0B0B0')
    for b, c in [(b1, '#0B3B24'), (b2, '#087B60'), (b3, '#707070')]:
        ax.bar_label(b, padding=2, fmt='%.1f', fontsize=9, fontweight='bold', color=c)
    max_y = df_f[['Real_sacas', 'RF_sacas', 'Baseline_sacas']].max().max()
    ax.set_ylim(0, max_y * 1.2)
    rotulos = [f"{m}\n({a})" for m, a in zip(df_f[COL_MUNICIPIO], df_f[COL_ANO])]
    ax.set_title(f'Real vs. Modelo vs. Baseline ({sufixo})', fontsize=15, fontweight='bold')
    ax.set_xlabel('Município e Ano')
    ax.set_ylabel('Rendimento (sacas/ha)')
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=9)
    ax.legend()
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'TCC_Grafico_Comparativo_{sufixo}.png', dpi=300, bbox_inches='tight')
    plt.close()


# =====================================================================
# EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    try:
        df_historico, df_moderno = preparar_dados(ARQUIVO_ENTRADA)

        resumo = {}

        # --- CENÁRIO 1: MODELO CLIMÁTICO DE LONGO PRAZO (1988-2023) ---
        if not df_historico.empty and df_historico[COL_ANO].min() < 2010:
            modelo_hist, imp_hist, df_res_hist, X_hist, res_hist = rodar_bateria_validacao(
                df_historico, VARIAVEIS_CLIMA,
                "CENÁRIO 1: MODELO CLIMÁTICO (LONGO PRAZO, SEM NDVI)",
                ano_corte_temporal=ANO_CORTE_HISTORICO,
                anos_choque=ANOS_CHOQUE_HISTORICO + ANOS_CHOQUE_RECENTE
            )
            gerar_graficos_interpretabilidade(modelo_hist, X_hist, "Historico_Clima")
            gerar_grafico_comparativo(df_res_hist, ANOS_CHOQUE_HISTORICO, "Historico")
            resumo['cenario_1_historico'] = res_hist

            exportar_hist = df_res_hist[[COL_ANO, COL_MUNICIPIO, COL_ALVO, 'Pred_KFold', 'media_movel_3anos', 'regime']].copy()
            exportar_hist.to_csv('TCC_Previsoes_Completas_Cenario1_Historico.csv', index=False)
        else:
            print("\n[AVISO] Base histórica não contém anos anteriores a 2010 - "
                  "verifique se a extensão 1988-2012 foi corretamente incorporada.")

        # --- CENÁRIO 2: MODELO MODERNO (CLIMA + NDVI, 2013-2023) ---
        if not df_moderno.empty:
            modelo_mod, imp_mod, df_res_mod, X_mod, res_mod = rodar_bateria_validacao(
                df_moderno, VARIAVEIS_CLIMA + VARIAVEIS_NDVI,
                "CENÁRIO 2: MODELO MODERNO (CLIMA + SATÉLITE)",
                ano_corte_temporal=ANO_CORTE_RECENTE,
                anos_choque=ANOS_CHOQUE_RECENTE
            )
            gerar_graficos_interpretabilidade(modelo_mod, X_mod, "Moderno_Satelite")
            gerar_grafico_comparativo(df_res_mod, ANOS_CHOQUE_RECENTE, "Moderno")
            resumo['cenario_2_moderno'] = res_mod

            exportar = df_res_mod[[COL_ANO, COL_MUNICIPIO, COL_ALVO, 'Pred_KFold', 'media_movel_3anos']].copy()
            exportar['Real_Sacas'] = exportar[COL_ALVO] / 60
            exportar['IA_Sacas'] = exportar['Pred_KFold'] / 60
            exportar.to_excel("TCC_Resultados_Previsoes_Sacas.xlsx", index=False, float_format="%.2f")
            exportar.to_csv('TCC_Previsoes_Completas_Cenario2_Moderno.csv', index=False)

        # --- RESUMO COMPARATIVO ENTRE OS DOIS CENÁRIOS ---
        print("\n" + "=" * 70)
        print(" RESUMO: A EXTENSÃO HISTÓRICA RESOLVEU O PROBLEMA DE EXTRAPOLAÇÃO?")
        print("=" * 70)
        for cenario, res in resumo.items():
            r2_temp = res['temporal']['r2']
            r2_temp_str = f"{r2_temp:.3f}" if r2_temp is not None else "N/A"
            print(f"{cenario}: R² Holdout Temporal = {r2_temp_str} "
                  f"(compare com o R² = -1.313 do modelo moderno original, sem extensão)")
        print("=" * 70)

    except FileNotFoundError:
        print(f"\n[ERRO] O arquivo '{ARQUIVO_ENTRADA}' não foi encontrado.")
