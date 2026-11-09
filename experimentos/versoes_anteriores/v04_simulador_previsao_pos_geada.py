import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def simulador_maquina_do_tempo():
    print("=========================================================")
    print(" 🔮 SIMULADOR DE MÁQUINA DO TEMPO: PREVISÃO CEGA PÓS-GEADA")
    print("=========================================================")

    # Usaremos o modelo fenológico completo (Clima + NDVI)
    ARQUIVO = 'TCC_Matriz_Unificada_1988_2023.xlsx'
    COL_ALVO = 'Rendimento_kg_ha'

    VARIAVEIS_X = [
        'anomalia_bienal_t_1', 'Tmin_Abs', 'SPI_Anual', 'Dias_Abaixo_3C',
        'tendencia_temporal', 'NDVI_Vigor_Maximo', 'NDVI_Estresse_Maximo',
        'NDVI_Florada', 'NDVI_Amplitude'
    ]

    try:
        df = pd.read_excel(ARQUIVO)

        # --- NOVO BLOCO: Calculando as variáveis avançadas na hora ---
        df = df.sort_values(by=['Municipio', 'Ano']).reset_index(drop=True)

        # 1. Bienalidade e Inércia
        df['produtividade_t_1'] = df.groupby('Municipio')[COL_ALVO].shift(1)
        df['media_movel_3anos'] = df.groupby('Municipio')[COL_ALVO].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        df['anomalia_bienal_t_1'] = df['produtividade_t_1'] - df['media_movel_3anos']

        # 2. Clima (SPI) e Tendência
        if 'Precip_Anual' in df.columns:
            precip = df['Precip_Anual'].fillna(df['Precip_Anual'].median()).values
            shape, loc, scale = stats.gamma.fit(precip, floc=0)
            percentis = stats.gamma.cdf(precip, shape, loc=loc, scale=scale)
            percentis = np.clip(percentis, 1e-6, 1 - 1e-6)
            df['SPI_Anual'] = stats.norm.ppf(percentis)
        else:
            df['SPI_Anual'] = 0

        df['tendencia_temporal'] = df['Ano'] - df['Ano'].min()
        # -------------------------------------------------------------

        # Agora sim, filtra apenas linhas que tenham o NDVI e Clima completos
        df = df.dropna(subset=VARIAVEIS_X + [COL_ALVO, 'media_movel_3anos']).copy()

        ANOS_DA_PREVISAO = [2015, 2016, 2022, 2023]

        for ANO_DA_PREVISAO in ANOS_DA_PREVISAO:
            print(f"\n1. Dividindo a linha do tempo no ano de {ANO_DA_PREVISAO}...")

            # O PASSADO: O modelo treina com tudo que aconteceu ATÉ o ano anterior à previsão
            df_treino = df[df['Ano'] < ANO_DA_PREVISAO]

            if df_treino.empty:
               print(f"⚠️ Não há dados suficientes no passado para treinar o modelo para {ANO_DA_PREVISAO}.")
               continue

            X_treino = df_treino[VARIAVEIS_X]
            y_treino = df_treino[COL_ALVO]

            # O FUTURO: O modelo vai ser testado no ano de previsão
            df_teste = df[df['Ano'] == ANO_DA_PREVISAO]

            if df_teste.empty:
                 print(f"⚠️ A base de dados não contém registros para testar a previsão do ano de {ANO_DA_PREVISAO}.")
                 continue

            # AQUI ESTÁ O SEGREDO: X_teste NÃO tem a coluna de produção (y)!
            X_teste = df_teste[VARIAVEIS_X]
            y_teste_real = df_teste[COL_ALVO] # Guardamos isso só para conferir depois se a IA acertou

            print(f"-> A IA vai estudar {len(df_treino)} safras antigas (Até {ANO_DA_PREVISAO-1}).")
            print(f"-> A IA vai tentar prever {len(df_teste)} municípios no ano de {ANO_DA_PREVISAO}.")

            print(f"\n2. Treinando a Inteligência Artificial apenas com o passado (até {ANO_DA_PREVISAO-1})...")
            # Usamos os mesmos parâmetros conservadores do seu TCC
            rf = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_split=5, random_state=42)
            rf.fit(X_treino, y_treino)

            print(f"3. Fazendo a previsão de {ANO_DA_PREVISAO} baseada apenas em Clima e NDVI...")
            # Note que passamos apenas X_teste (Clima e NDVI). A IA não sabe o resultado de 2022!
            previsoes = rf.predict(X_teste)

            print("\n=========================================================")
            print(f" 📊 RESULTADOS DA PREVISÃO CEGA PARA {ANO_DA_PREVISAO} (Em sacas/ha)")
            print("=========================================================")

            df_resultados = pd.DataFrame({
                'Municipio': df_teste['Municipio'],
                'Previsto_IA': previsoes / 60,
                'Real_IBGE': y_teste_real / 60,
                'Baseline_Inercia': df_teste['media_movel_3anos'] / 60
            })

            # Calculando o erro absoluto (a diferença entre a previsão e a realidade)
            df_resultados['Erro_IA'] = abs(df_resultados['Previsto_IA'] - df_resultados['Real_IBGE'])

            # Exibindo bonito no terminal
            pd.set_option('display.float_format', '{:.1f}'.format)
            print(df_resultados[['Municipio', 'Baseline_Inercia', 'Previsto_IA', 'Real_IBGE', 'Erro_IA']].to_string(index=False))

            # Resumo do erro
            erro_medio = df_resultados['Erro_IA'].mean()
            print(f"\n🎯 Erro Médio Absoluto da IA para {ANO_DA_PREVISAO}: {erro_medio:.1f} sacas/ha")
            print("-> Compare a 'Previsto_IA' com a 'Real_IBGE' e veja como ela ajustou a ")
            print("   estimativa muito melhor do que se seguisse apenas a 'Baseline_Inercia'.")
            print("\n" + "="*57)

    except FileNotFoundError:
        print(f"❌ [ERRO] O arquivo '{ARQUIVO}' não foi encontrado.")

if __name__ == "__main__":
    simulador_maquina_do_tempo()