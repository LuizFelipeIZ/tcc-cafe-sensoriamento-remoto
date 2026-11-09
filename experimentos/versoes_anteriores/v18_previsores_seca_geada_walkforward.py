"""
=====================================================================
TCC - Previsores Operacionais: Seca (intra-safra) e Geada (defasado)
=====================================================================
Dois modelos, com lógicas temporais diferentes e deliberadas:

  MODELO SECA: usa deficit hidrico da PROPRIA safra (Florada/Expansao),
  porque a seca se instala e afeta o mesmo ciclo de crescimento -
  eh um alerta "dentro da safra em curso".

  MODELO GEADA: usa as variaveis de inverno DEFASADAS em 1 ano
  (Tmin_Abs_Inverno_Risco_Geada_t_1 etc.), porque a geada de
  junho-agosto do ano X danifica a florada/safra de X+1, nao a
  propria safra de X (convencao ja usada no projeto: "geada 2021"
  impacta as safras de 2022/2023).

Ambos validados por janela expansiva (walk-forward): o treino de cada
ano de teste usa SOMENTE safras estritamente anteriores aquele ano.
=====================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.width', 140)

ARQUIVO_ENTRADA = 'TCC_Matriz_Mestra_THORNTHWAITE.xlsx'
COL_MUNICIPIO = 'Municipio'
COL_ANO = 'Ano'
COL_ALVO = 'Rendimento_kg_ha'

RF_PARAMS = dict(n_estimators=200, max_depth=5, min_samples_split=5, random_state=42)

ANOS_TESTE_SECA = [2014, 2015, 2016, 2017]
VARIAVEIS_SECA = [
    'anomalia_bienal_t_1', 'tendencia_temporal',
    'DEF_Florada', 'DEF_Expansao_Graos', 'Tmax_Expansao_Graos',
]

ANOS_TESTE_GEADA = [1994, 1995, 2000, 2001, 2021, 2022, 2023]
VARIAVEIS_GEADA = [
    'anomalia_bienal_t_1', 'tendencia_temporal',
    'Tmin_Abs_Inverno_Risco_Geada_t_1',
    'Dias_Abaixo_3C_t_1',
    'Precip_Inverno_Risco_Geada_t_1',
]


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

    # Defasagem das variaveis de inverno em 1 ano, para o modelo de geada
    for col in ['Tmin_Abs_Inverno_Risco_Geada', 'Dias_Abaixo_3C', 'Precip_Inverno_Risco_Geada']:
        if col in df.columns:
            df[f'{col}_t_1'] = df.groupby(COL_MUNICIPIO)[col].shift(1)

    return df


def prever_com_intervalo(rf_model, X_linha):
    previsoes_arvores = np.array([arvore.predict(X_linha)[0] for arvore in rf_model.estimators_])
    return {
        'previsto': previsoes_arvores.mean(),
        'p10': np.percentile(previsoes_arvores, 10),
        'p90': np.percentile(previsoes_arvores, 90),
    }


def rodar_previsor(df_bruto, variaveis, anos_teste, nome_modelo):
    df = df_bruto.dropna(subset=variaveis + [COL_ALVO, 'media_movel_3anos']).copy().reset_index(drop=True)
    print(f"\n[INFO] {nome_modelo}: N = {len(df)} observações válidas.")

    resultados = []
    for ano_previsao in anos_teste:
        df_treino = df[df[COL_ANO] < ano_previsao]
        df_teste = df[df[COL_ANO] == ano_previsao]
        if df_treino.empty or df_teste.empty:
            print(f"  [AVISO] Dados insuficientes para prever {ano_previsao}.")
            continue

        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(df_treino[variaveis], df_treino[COL_ALVO])

        for _, linha in df_teste.iterrows():
            X_linha = linha[variaveis].to_frame().T
            pred = prever_com_intervalo(rf, X_linha)
            real_sacas = linha[COL_ALVO] / 60
            resultados.append({
                'Municipio': linha[COL_MUNICIPIO], 'Ano': ano_previsao,
                'Real_sacas': round(real_sacas, 1),
                'Previsto_sacas': round(pred['previsto'] / 60, 1),
                'P10_sacas': round(pred['p10'] / 60, 1),
                'P90_sacas': round(pred['p90'] / 60, 1),
                'Erro_Absoluto_sacas': round(abs(real_sacas - pred['previsto'] / 60), 1),
                'Dentro_do_Intervalo': (pred['p10']/60 <= real_sacas <= pred['p90']/60),
            })
        print(f"  [OK] Safra {ano_previsao}: treino N={len(df_treino)}, previsão para {len(df_teste)} municípios.")

    resultado_df = pd.DataFrame(resultados)
    if resultado_df.empty:
        print(f"  [ERRO] Nenhuma previsão gerada para {nome_modelo}.")
        return resultado_df

    mae = resultado_df['Erro_Absoluto_sacas'].mean()
    cobertura = resultado_df['Dentro_do_Intervalo'].mean() * 100
    print(f"\n  MAE geral ({nome_modelo}): {mae:.2f} sacas/ha")
    print(f"  Cobertura do intervalo P10-P90: {cobertura:.1f}% (ideal: ~80%)")
    print(f"\n  MAE por safra:")
    print(resultado_df.groupby('Ano')['Erro_Absoluto_sacas'].mean().to_string())

    return resultado_df


if __name__ == "__main__":
    df_bruto = preparar_dados(ARQUIVO_ENTRADA)

    print("#" * 100)
    print("# MODELO SECA (intra-safra, sem defasagem)")
    print("#" * 100)
    resultados_seca = rodar_previsor(df_bruto, VARIAVEIS_SECA, ANOS_TESTE_SECA, "SECA")

    print("\n" + "#" * 100)
    print("# MODELO GEADA (defasado 1 ano - inverno do ano anterior)")
    print("#" * 100)
    resultados_geada = rodar_previsor(df_bruto, VARIAVEIS_GEADA, ANOS_TESTE_GEADA, "GEADA")

    if not resultados_seca.empty:
        resultados_seca.to_excel('TCC_Previsor_Operacional_Seca.xlsx', index=False)
    if not resultados_geada.empty:
        resultados_geada.to_excel('TCC_Previsor_Operacional_Geada.xlsx', index=False)
    print("\n[OK] Planilhas salvas.")
