"""
=====================================================================
TCC - Previsão de Rendimento Cafeeiro Pós-Anomalias Climáticas
Script Final: Cabo Verde, Guaxupé, Muzambinho e Nova Resende
=====================================================================

DESIGN: o modelo é TREINADO com os 10 municípios da microrregião (mais
dados = mais poder estatístico e melhor generalização, conforme toda a
validação de robustez já feita no TCC). Os resultados finais (tabelas e
gráficos) são então FILTRADOS para os 4 municípios de interesse. Isso
evita o erro de treinar com N muito pequeno, que enfraqueceria o modelo
sem necessidade - a limitação é só na apresentação, não no aprendizado.

Roda os dois cenários já validados no TCC:
  Cenário 1: Clima puro, 1988-2023 (350 obs, sem NDVI)
  Cenário 2: Clima + NDVI via satélite, 2013-2023 (110 obs)

E gera as previsões pós-evento para os 4 municípios, nos 4 eventos
climáticos documentados: Geada 1994, Geada 2000, Seca 2014, Geada 2021.

Requisitos: pip install pandas numpy scikit-learn matplotlib shap scipy openpyxl
Entrada: TCC_Matriz_Mestra_FINAL_1988_2023.xlsx (mesma pasta do script)
=====================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from scipy import stats
import matplotlib.pyplot as plt
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

# Os 4 municípios de interesse para a apresentação final
MUNICIPIOS_ALVO = ['Cabo Verde', 'Guaxupe', 'Muzambinho', 'Nova Resende']

# Os quatro eventos climáticos documentados e seus dois anos de resposta
EVENTOS = {
    'Geada 1994': [1994, 1995],
    'Geada 2000': [2000, 2001],
    'Seca 2014':  [2015, 2016],
    'Geada 2021': [2022, 2023],
}

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

    df_clima = df.dropna(subset=VARIAVEIS_CLIMA + [COL_ALVO, 'media_movel_3anos']).copy().reset_index(drop=True)
    df_completo = df.dropna(subset=VARIAVEIS_CLIMA + VARIAVEIS_NDVI + [COL_ALVO, 'media_movel_3anos']).copy().reset_index(drop=True)

    print(f"[INFO] Cenário 1 (Clima, 1988-2023): {len(df_clima)} observações, 10 municípios")
    print(f"[INFO] Cenário 2 (Clima+NDVI, 2013-2023): {len(df_completo)} observações, 10 municípios")

    return df_clima, df_completo


# =====================================================================
# 2. TREINO COM NESTED CROSS-VALIDATION (K-Fold k=5)
# =====================================================================
def treinar_modelo(df, vars_x, nome_cenario):
    print(f"\n[INFO] Treinando {nome_cenario}...")
    X = df[vars_x]
    y = df[COL_ALVO]

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros(len(y))
    for train_idx, test_idx in outer_cv.split(X):
        gs = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID,
                           cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred[test_idx] = gs.predict(X.iloc[test_idx])

    df['Pred_KFold'] = y_pred
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"  R² = {r2:.3f} | MAE = {mae/60:.2f} sacas/ha (N={len(df)}, todos os 10 municípios)")

    return df


# =====================================================================
# 3. FILTRAR PARA OS 4 MUNICÍPIOS E MONTAR A TABELA PÓS-EVENTO
# =====================================================================
def montar_tabela_pos_evento(df_hist, df_mod, municipios_alvo, eventos):
    linhas = []
    for evento, anos in eventos.items():
        fonte = df_hist if anos[0] < 2013 else df_mod
        sub = fonte[(fonte[COL_MUNICIPIO].isin(municipios_alvo)) & (fonte[COL_ANO].isin(anos))].copy()
        sub['Evento'] = evento
        sub['Real_sacas'] = sub[COL_ALVO] / 60
        sub['RF_sacas'] = sub['Pred_KFold'] / 60
        sub['Baseline_sacas'] = sub['media_movel_3anos'] / 60
        sub['Erro_RF_sacas'] = (sub['Real_sacas'] - sub['RF_sacas']).abs()
        sub['Erro_Baseline_sacas'] = (sub['Real_sacas'] - sub['Baseline_sacas']).abs()
        linhas.append(sub[[COL_ANO, COL_MUNICIPIO, 'Evento', 'Real_sacas', 'RF_sacas',
                            'Baseline_sacas', 'Erro_RF_sacas', 'Erro_Baseline_sacas']])

    tabela = pd.concat(linhas, ignore_index=True)
    tabela = tabela.sort_values(['Evento', COL_MUNICIPIO, COL_ANO]).reset_index(drop=True)
    return tabela


# =====================================================================
# 4. GRÁFICOS
# =====================================================================
def gerar_grafico_resumo(tabela, caminho_saida):
    resumo = tabela.groupby('Evento').agg(
        MAE_RF=('Erro_RF_sacas', 'mean'),
        MAE_Baseline=('Erro_Baseline_sacas', 'mean')
    ).reindex(['Geada 1994', 'Geada 2000', 'Seca 2014', 'Geada 2021'])

    x = np.arange(len(resumo))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, resumo['MAE_RF'], w, label='Random Forest', color='#11CAA0')
    b2 = ax.bar(x + w/2, resumo['MAE_Baseline'], w, label='Baseline Inercial', color='#B0B0B0')
    ax.bar_label(b1, fmt='%.2f', padding=3, fontweight='bold', color='#087B60')
    ax.bar_label(b2, fmt='%.2f', padding=3, fontweight='bold', color='#707070')
    ax.set_xticks(x)
    ax.set_xticklabels(resumo.index, fontsize=11)
    ax.set_ylabel('Erro Médio Absoluto (sacas/ha)', fontsize=12)
    ax.set_title('Erro de Previsão Pós-Evento\nCabo Verde, Guaxupé, Muzambinho e Nova Resende',
                  fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    return resumo


def gerar_grafico_detalhado(tabela, caminho_saida):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()
    eventos_ordem = ['Geada 1994', 'Geada 2000', 'Seca 2014', 'Geada 2021']

    for i, evento in enumerate(eventos_ordem):
        ax = axes[i]
        sub = tabela[tabela['Evento'] == evento].sort_values([COL_MUNICIPIO, COL_ANO])
        xx = np.arange(len(sub))
        ww = 0.27
        ax.bar(xx - ww, sub['Real_sacas'], ww, label='IBGE (Real)', color='#0B3B24')
        ax.bar(xx, sub['RF_sacas'], ww, label='Random Forest', color='#11CAA0')
        ax.bar(xx + ww, sub['Baseline_sacas'], ww, label='Baseline', color='#B0B0B0')
        rotulos = [f"{m}\n({a})" for m, a in zip(sub[COL_MUNICIPIO], sub[COL_ANO])]
        ax.set_xticks(xx)
        ax.set_xticklabels(rotulos, fontsize=8, rotation=45, ha='right')
        ax.set_title(evento, fontsize=13, fontweight='bold')
        ax.set_ylabel('sacas/ha', fontsize=10)
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)
        if i == 0:
            ax.legend(fontsize=9)

    plt.suptitle('Real vs. Modelo vs. Baseline nos Anos Pós-Evento\n'
                  'Cabo Verde, Guaxupé, Muzambinho e Nova Resende',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=250, bbox_inches='tight')
    plt.close()


# =====================================================================
# EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    df_clima, df_completo = preparar_dados(ARQUIVO_ENTRADA)

    df_hist = treinar_modelo(df_clima, VARIAVEIS_CLIMA, "Cenário 1 (Clima, 1988-2023)")
    df_mod = treinar_modelo(df_completo, VARIAVEIS_CLIMA + VARIAVEIS_NDVI, "Cenário 2 (Clima+NDVI, 2013-2023)")

    tabela = montar_tabela_pos_evento(df_hist, df_mod, MUNICIPIOS_ALVO, EVENTOS)
    tabela.to_excel('TCC_Previsoes_4_Municipios_Pos_Evento.xlsx', index=False)

    print("\n" + "=" * 70)
    print(" RESUMO - ERRO MÉDIO POR EVENTO (Cabo Verde, Guaxupé, Muzambinho, Nova Resende)")
    print("=" * 70)
    resumo = gerar_grafico_resumo(tabela, 'TCC_MAE_4_Municipios_Resumo.png')
    print(resumo)

    gerar_grafico_detalhado(tabela, 'TCC_Detalhado_4_Municipios.png')

    print("\n[OK] Arquivos gerados:")
    print("  - TCC_Previsoes_4_Municipios_Pos_Evento.xlsx (tabela completa)")
    print("  - TCC_MAE_4_Municipios_Resumo.png (gráfico-resumo)")
    print("  - TCC_Detalhado_4_Municipios.png (gráfico detalhado por evento)")
