"""
=====================================================================
TCC - Atribuição de Dano Climático com Isolamento de Tendência
Temporal e Bienalidade (Residualização em Duas Etapas)
=====================================================================

PROBLEMA QUE ISSO RESOLVE: nos modelos anteriores, 'tendencia_temporal'
e 'anomalia_bienal_t_1' competem pela mesma "fatia" de importância que
as variáveis climáticas/fisiológicas genuínas (Tmin_Abs, NDVI, etc.),
o que pode mascarar ou inflar artificialmente o peso relativo do clima.

SOLUÇÃO (residualização / partialling-out, técnica econométrica padrão):
  ETAPA 1: Regressão linear simples de Rendimento_kg_ha em função APENAS
           de tendencia_temporal e anomalia_bienal_t_1. Extrai os
           RESÍDUOS - a parte do rendimento que tendência tecnológica e
           ciclo bienal NÃO conseguem explicar.
  ETAPA 2: Random Forest (Nested CV) treinado para prever esses
           RESÍDUOS usando SOMENTE as variáveis climáticas/fisiológicas
           genuínas (sem tendencia_temporal, sem anomalia_bienal_t_1).
           A Feature Importance (MDI) e o SHAP dessa etapa revelam o
           peso real de cada fator climático sobre a parcela do
           rendimento que sobra depois de "descontar" tecnologia e
           bienalidade - isolamento estatístico de verdade, não só
           remoção de variável.

Roda para os dois cenários: Cenário 1 (clima longo, 1988-2023) e
Cenário 2 (clima+NDVI moderno, 2013-2023).
=====================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from scipy import stats
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.width', 140)

ARQUIVO_ENTRADA = 'TCC_Matriz_Mestra_FINAL_1988_2023.xlsx'
COL_MUNICIPIO = 'Municipio'
COL_ANO = 'Ano'
COL_ALVO = 'Rendimento_kg_ha'

VARIAVEIS_ISOLAR = ['tendencia_temporal', 'anomalia_bienal_t_1']

VARIAVEIS_CLIMA_PURO = [
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
# 1. PREPARO (idêntico ao pipeline principal, para consistência)
# =====================================================================
def preparar_dados(caminho_arquivo):
    df = pd.read_excel(caminho_arquivo)
    df = df.sort_values(by=[COL_MUNICIPIO, COL_ANO]).reset_index(drop=True)

    df['produtividade_t_1'] = df.groupby(COL_MUNICIPIO)[COL_ALVO].shift(1)
    df['media_movel_3anos'] = df.groupby(COL_MUNICIPIO)[COL_ALVO].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['anomalia_bienal_t_1'] = df['produtividade_t_1'] - df['media_movel_3anos']

    if 'Precip_Anual' in df.columns:
        precip = df['Precip_Anual'].fillna(df['Precip_Anual'].median()).values
        shape, loc, scale = stats.gamma.fit(precip, floc=0)
        percentis = stats.gamma.cdf(precip, shape, loc=loc, scale=scale)
        percentis = np.clip(percentis, 1e-6, 1 - 1e-6)
        df['SPI_Anual'] = stats.norm.ppf(percentis)
    else:
        df['SPI_Anual'] = 0

    df['tendencia_temporal'] = df[COL_ANO] - df[COL_ANO].min()

    vars_clima_completas = VARIAVEIS_CLIMA_PURO + VARIAVEIS_ISOLAR
    df_clima = df.dropna(subset=vars_clima_completas + [COL_ALVO]).copy().reset_index(drop=True)
    df_completo = df.dropna(subset=vars_clima_completas + VARIAVEIS_NDVI + [COL_ALVO]).copy().reset_index(drop=True)

    return df_clima, df_completo


# =====================================================================
# 2. ETAPA 1: REGRESSÃO DE CONTROLE (tendência + bienalidade) E RESÍDUOS
# =====================================================================
def calcular_residuos(df):
    X_controle = df[VARIAVEIS_ISOLAR]
    y = df[COL_ALVO]

    modelo_controle = LinearRegression()
    modelo_controle.fit(X_controle, y)
    y_predito_controle = modelo_controle.predict(X_controle)

    r2_controle = r2_score(y, y_predito_controle)
    df['Predito_Tendencia_Bienalidade'] = y_predito_controle
    df['Residuo_Rendimento'] = y - y_predito_controle

    print(f"  [Etapa 1 - Regressão de controle] R² (tendência+bienalidade sozinhas) = {r2_controle:.3f}")
    print(f"    Coeficiente tendencia_temporal:  {modelo_controle.coef_[0]:.3f} kg/ha por ano")
    print(f"    Coeficiente anomalia_bienal_t_1: {modelo_controle.coef_[1]:.3f}")
    print(f"    Isso significa que tendência+bienalidade explicam {r2_controle*100:.1f}% da variância")
    print(f"    total do rendimento - o restante ({(1-r2_controle)*100:.1f}%) é o que resta para o clima explicar.")

    return df, r2_controle


# =====================================================================
# 3. ETAPA 2: RANDOM FOREST SOBRE OS RESÍDUOS (só clima/NDVI)
# =====================================================================
def treinar_sobre_residuos(df, vars_x, nome_cenario):
    print(f"\n[INFO] {nome_cenario}")
    X = df[vars_x]
    y_residuo = df['Residuo_Rendimento']

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros(len(y_residuo))
    for train_idx, test_idx in outer_cv.split(X):
        gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                           cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs.fit(X.iloc[train_idx], y_residuo.iloc[train_idx])
        y_pred[test_idx] = gs.predict(X.iloc[test_idx])

    r2_residuo = r2_score(y_residuo, y_pred)
    mae_residuo = mean_absolute_error(y_residuo, y_pred)
    print(f"  [Etapa 2 - RF sobre resíduos] R² = {r2_residuo:.3f} | MAE = {mae_residuo/60:.2f} sacas/ha")
    print(f"    (R² aqui mede quanto do resíduo - a parte NÃO explicada por tendência/bienalidade -")
    print(f"     o clima/NDVI consegue explicar. Não é comparável ao R² do modelo original.)")

    # Modelo final para extração de importância
    gs_full = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                            cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    gs_full.fit(X, y_residuo)
    rf_final = gs_full.best_estimator_
    rf_final.fit(X, y_residuo)

    importancias = pd.DataFrame({
        'Variavel': vars_x,
        'Importancia_Perc': rf_final.feature_importances_ * 100
    }).sort_values('Importancia_Perc', ascending=False).reset_index(drop=True)

    explainer = shap.TreeExplainer(rf_final)
    shap_values = explainer.shap_values(X)
    shap_imp = pd.DataFrame({
        'Variavel': vars_x,
        'SHAP_mean_abs': np.abs(shap_values).mean(axis=0)
    }).sort_values('SHAP_mean_abs', ascending=False).reset_index(drop=True)
    shap_imp['SHAP_Perc'] = shap_imp['SHAP_mean_abs'] / shap_imp['SHAP_mean_abs'].sum() * 100

    comparativo = importancias.merge(shap_imp[['Variavel', 'SHAP_Perc']], on='Variavel')
    comparativo['Rank_MDI'] = comparativo['Importancia_Perc'].rank(ascending=False).astype(int)
    comparativo['Rank_SHAP'] = comparativo['SHAP_Perc'].rank(ascending=False).astype(int)
    comparativo = comparativo.sort_values('Rank_MDI')
    corr_rank = comparativo['Rank_MDI'].corr(comparativo['Rank_SHAP'], method='spearman')

    print(f"\n  Hierarquia de atribuição CLIMÁTICA PURA (tendência/bienalidade já removidas):")
    print(comparativo.to_string(index=False))
    print(f"\n  Correlação de Spearman MDI x SHAP: {corr_rank:.3f}")

    return rf_final, comparativo, X, r2_residuo, shap_values


# =====================================================================
# 4. GRÁFICOS COMPARATIVOS (antes/depois do isolamento)
# =====================================================================
def gerar_grafico_comparacao(importancia_com, importancia_sem, nome_cenario, caminho_saida):
    """Compara a hierarquia de importância COM tendência/bienalidade
    competindo (modelo original) vs. SEM elas (modelo isolado/residual)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, imp, titulo in [(axes[0], importancia_com, 'COM tendência/bienalidade\n(modelo original)'),
                             (axes[1], importancia_sem, 'SEM tendência/bienalidade\n(isolado, sobre resíduos)')]:
        imp_sorted = imp.sort_values('Importancia_Perc' if 'Importancia_Perc' in imp.columns else 'SHAP_Perc')
        col_valor = 'Importancia_Perc' if 'Importancia_Perc' in imp.columns else 'SHAP_Perc'
        cores = ['#0B3B24' if v in VARIAVEIS_ISOLAR else '#11CAA0' for v in imp_sorted['Variavel']]
        ax.barh(imp_sorted['Variavel'], imp_sorted[col_valor], color=cores)
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.set_xlabel('Importância (%)', fontsize=10)

    plt.suptitle(f'Efeito do Isolamento de Tendência Temporal e Bienalidade - {nome_cenario}',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()


def gerar_shap_residuo(shap_values, X, nome_cenario, caminho_saida):
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'SHAP sobre Resíduos (Clima Puro) - {nome_cenario}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()


# =====================================================================
# EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    df_clima, df_completo = preparar_dados(ARQUIVO_ENTRADA)

    resultados = {}

    for nome_cenario, df, vars_x_completas, vars_x_puras in [
        ("CENÁRIO 1 (Clima, 1988-2023)", df_clima,
         VARIAVEIS_CLIMA_PURO + VARIAVEIS_ISOLAR, VARIAVEIS_CLIMA_PURO),
        ("CENÁRIO 2 (Clima+NDVI, 2013-2023)", df_completo,
         VARIAVEIS_CLIMA_PURO + VARIAVEIS_NDVI + VARIAVEIS_ISOLAR, VARIAVEIS_CLIMA_PURO + VARIAVEIS_NDVI),
    ]:
        print("\n" + "=" * 70)
        print(f" {nome_cenario}")
        print("=" * 70)

        # --- Modelo ORIGINAL (com tendência/bienalidade competindo) ---
        X_original = df[vars_x_completas]
        y_original = df[COL_ALVO]
        gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                           cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs.fit(X_original, y_original)
        rf_original = gs.best_estimator_
        rf_original.fit(X_original, y_original)
        importancia_original = pd.DataFrame({
            'Variavel': vars_x_completas,
            'Importancia_Perc': rf_original.feature_importances_ * 100
        }).sort_values('Importancia_Perc', ascending=False)

        print("\n  Hierarquia ORIGINAL (tendência/bienalidade competindo):")
        print(importancia_original.to_string(index=False))

        # --- Etapa 1: resíduos ---
        df, r2_controle = calcular_residuos(df)

        # --- Etapa 2: RF sobre resíduos, só clima/NDVI ---
        rf_residuo, comparativo_residuo, X_residuo, r2_residuo, shap_values = treinar_sobre_residuos(
            df, vars_x_puras, nome_cenario
        )

        sufixo = 'Historico' if '1988' in nome_cenario else 'Moderno'
        gerar_grafico_comparacao(importancia_original, comparativo_residuo, nome_cenario,
                                  f'TCC_Isolamento_Comparacao_{sufixo}.png')
        gerar_shap_residuo(shap_values, X_residuo, nome_cenario, f'TCC_Isolamento_SHAP_Residuo_{sufixo}.png')

        resultados[nome_cenario] = {
            'r2_controle': r2_controle,
            'r2_residuo': r2_residuo,
            'importancia_original': importancia_original,
            'importancia_isolada': comparativo_residuo,
        }

    print("\n" + "=" * 70)
    print(" RESUMO FINAL")
    print("=" * 70)
    for nome, res in resultados.items():
        print(f"\n{nome}:")
        print(f"  Tendência+Bienalidade explicam sozinhas: {res['r2_controle']*100:.1f}% da variância do rendimento")
        print(f"  Clima/NDVI explicam do restante (resíduo): {res['r2_residuo']*100:.1f}%")
        top3 = res['importancia_isolada'].nlargest(3, 'Importancia_Perc')[['Variavel', 'Importancia_Perc']]
        print(f"  Top 3 variáveis climáticas puras: {top3.to_dict('records')}")
