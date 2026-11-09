"""
=====================================================================
TCC - Cenário 3: Alerta Intra-Safra de Seca Localizada
=====================================================================
Usa exclusivamente variáveis conhecíveis ANTES da colheita (maio-agosto
na região): déficit hídrico de florada e de expansão de grãos, mais
temperatura máxima de expansão. Exclui deliberadamente as variáveis de
"Inverno_Risco_Geada" (jun-ago), que se sobrepõem à própria colheita.

Evento de teste: seca de 2014-2017 (evento plurianual, não apenas 2014),
usado como bloco único de holdout temporal - o único evento de seca
severa documentável na série.
=====================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from scipy import stats
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.width', 140)

ARQUIVO_ENTRADA = 'TCC_Matriz_Mestra_THORNTHWAITE.xlsx'
COL_MUNICIPIO = 'Municipio'
COL_ANO = 'Ano'
COL_ALVO = 'Rendimento_kg_ha'

SECA_ANOS_CHOQUE = [2014, 2015, 2016, 2017]  # evento plurianual corrigido

VARIAVEIS_INTRA_SAFRA = [
    'anomalia_bienal_t_1', 'tendencia_temporal',
    'DEF_Florada', 'DEF_Expansao_Graos', 'Tmax_Expansao_Graos',
]

PARAM_GRID = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'min_samples_split': [5, 10]
}


def preparar_dados(caminho_arquivo):
    df = pd.read_excel(caminho_arquivo)
    df[COL_MUNICIPIO] = df[COL_MUNICIPIO].str.strip()
    df = df.sort_values(by=[COL_MUNICIPIO, COL_ANO]).reset_index(drop=True)

    df['produtividade_t_1'] = df.groupby(COL_MUNICIPIO)[COL_ALVO].shift(1)
    df['media_movel_3anos'] = df.groupby(COL_MUNICIPIO)[COL_ALVO].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['anomalia_bienal_t_1'] = df['produtividade_t_1'] - df['media_movel_3anos']
    df['tendencia_temporal'] = df[COL_ANO] - df[COL_ANO].min()

    df_final = df.dropna(subset=VARIAVEIS_INTRA_SAFRA + [COL_ALVO, 'media_movel_3anos']).copy().reset_index(drop=True)
    print(f"[INFO] Amostra efetiva N = {len(df_final)} (de {len(df)} linhas brutas).")
    print(f"[INFO] Municípios com dado completo: {sorted(df_final[COL_MUNICIPIO].unique())}")
    return df_final


def rodar_bateria_validacao(df):
    X = df[VARIAVEIS_INTRA_SAFRA]
    y = df[COL_ALVO]

    y_base = df['media_movel_3anos']
    r2_b = r2_score(y, y_base)
    mae_b = mean_absolute_error(y, y_base)
    print(f"\n[BASELINE INGÊNUA]      R² = {r2_b:.3f} | MAE = {mae_b/60:.2f} sacas/ha")

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
    print(f"[RF - K-Fold padrão]    R² = {r2_k:.3f} | MAE = {mae_k/60:.2f} sacas/ha")

    groups = df[COL_MUNICIPIO]
    if groups.nunique() >= 5:
        gkf = GroupKFold(n_splits=min(5, groups.nunique()))
        y_pred_group = np.zeros(len(y))
        for train_idx, test_idx in gkf.split(X, y, groups):
            gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                               cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            gs.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred_group[test_idx] = gs.predict(X.iloc[test_idx])
        r2_g = r2_score(y, y_pred_group)
        mae_g = mean_absolute_error(y, y_pred_group)
        print(f"[RF - GroupKFold]       R² = {r2_g:.3f} | MAE = {mae_g/60:.2f} sacas/ha")
    else:
        print("[AVISO] Municípios insuficientes para GroupKFold de 5 dobras.")

    mask_treino = ~df[COL_ANO].isin(SECA_ANOS_CHOQUE)
    mask_teste = df[COL_ANO].isin(SECA_ANOS_CHOQUE)
    if mask_treino.sum() > 10 and mask_teste.sum() > 3:
        X_tr, y_tr = X[mask_treino], y[mask_treino]
        X_te, y_te = X[mask_teste], y[mask_teste]
        gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                           cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs.fit(X_tr, y_tr)
        y_pred_seca = gs.predict(X_te)
        r2_seca = r2_score(y_te, y_pred_seca)
        mae_seca = mean_absolute_error(y_te, y_pred_seca)
        print(f"\n[RF - HOLDOUT DO EVENTO DE SECA 2014-2017] (o número mais importante desta seção)")
        print(f"  R² = {r2_seca:.3f} | MAE = {mae_seca/60:.2f} sacas/ha "
              f"(N treino={len(y_tr)}, N teste={len(y_te)})")
        df.loc[mask_teste, 'Pred_Holdout_Seca'] = y_pred_seca
    else:
        print(f"\n[AVISO] Dados insuficientes para o holdout do evento de seca "
              f"(treino={mask_treino.sum()}, teste={mask_teste.sum()}).")
        r2_seca, mae_seca = None, None

    erros_rf = np.abs(y - y_pred_kfold)
    erros_base = np.abs(y - y_base)
    try:
        stat, p_valor = wilcoxon(erros_base, erros_rf)
        print(f"\n[Wilcoxon RF vs Baseline] W = {stat:.2f} | p = {p_valor:.4f}")
    except ValueError as e:
        print(f"\n[Wilcoxon] Não foi possível calcular: {e}")

    gs_full = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                            cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    gs_full.fit(X, y)
    rf_final = gs_full.best_estimator_
    rf_final.fit(X, y)
    importancias = pd.DataFrame({
        'Variavel': VARIAVEIS_INTRA_SAFRA,
        'Importancia_Perc': rf_final.feature_importances_ * 100
    }).sort_values('Importancia_Perc', ascending=False)
    print("\n[Feature Importance - MDI]")
    print(importancias.to_string(index=False))

    return df, importancias, {'r2_seca': r2_seca, 'mae_seca': mae_seca}


def gerar_grafico_evento_seca(df, caminho_saida):
    sub = df[df[COL_ANO].isin(SECA_ANOS_CHOQUE) & df['Pred_Holdout_Seca'].notna()].copy()
    if sub.empty:
        print("[AVISO] Sem dados suficientes para o gráfico do evento de seca.")
        return
    sub = sub.sort_values([COL_MUNICIPIO, COL_ANO])
    sub['Real_sacas'] = sub[COL_ALVO] / 60
    sub['RF_sacas'] = sub['Pred_Holdout_Seca'] / 60
    sub['Baseline_sacas'] = sub['media_movel_3anos'] / 60

    x = np.arange(len(sub))
    w = 0.27
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.bar(x - w, sub['Real_sacas'], w, label='IBGE (Real)', color='#0B3B24')
    ax.bar(x, sub['RF_sacas'], w, label='RF (Cego, sem ver a seca)', color='#11CAA0')
    ax.bar(x + w, sub['Baseline_sacas'], w, label='Baseline', color='#B0B0B0')
    rotulos = [f"{m}\n({a})" for m, a in zip(sub[COL_MUNICIPIO], sub[COL_ANO])]
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=7, rotation=45, ha='right')
    ax.set_title('Holdout do Evento de Seca 2014-2017: Real vs. Modelo (nunca viu a seca) vs. Baseline',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('sacas/ha')
    ax.legend()
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráfico salvo: {caminho_saida}")


if __name__ == "__main__":
    try:
        df = preparar_dados(ARQUIVO_ENTRADA)
        df, importancias, resultado_seca = rodar_bateria_validacao(df)
        gerar_grafico_evento_seca(df, 'TCC_Cenario3_Holdout_Seca.png')
        df.to_csv('TCC_Cenario3_Previsoes_Completas.csv', index=False)
        print("\n[OK] Cenário 3 concluído.")
    except FileNotFoundError:
        print(f"[ERRO] Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
