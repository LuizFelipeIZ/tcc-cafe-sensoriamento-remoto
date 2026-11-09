"""
ETAPA 2b: NDVI de três sensores na mesma escala (calibração pelo RMA)
=====================================================================

O problema
----------
Landsat 5, 7 e 8 não medem o NDVI exatamente igual (bandas e calibração
diferentes). Se a série juntasse os três sem ajuste, o modelo poderia
"aprender" a troca de satélite como se fosse mudança na lavoura.

A solução
---------
Cada sensor foi extraído em todos os anos em que tinha imagens (etapa 1c).
Nos anos em que dois sensores se sobrepõem, ajusta-se uma reta que leva um
sensor para a escala do outro:
    * Landsat 7 -> Landsat 8, nas safras de 2014 a 2022
    * Landsat 5 -> Landsat 7, nas safras de 2000 a 2011 (sem 2003, ano da
      falha do corretor de linhas do Landsat 7)
A série final usa o Landsat 5 em 1989-1999, o Landsat 7 em 2000-2013 e o
Landsat 8 em 2014-2023, tudo na escala do Landsat 8.

Por que RMA e não a regressão comum (OLS)
----------------------------------------
Os dois sensores têm erro. A regressão comum supõe erro só no y e "achata"
a reta, o que encolhe a variabilidade da série corrigida. A regressão do eixo
maior reduzido (RMA; Smith, 2009) usa b = sinal(r) * dp(y) / dp(x) e preserva
a variabilidade. A parte 1 deste script mostra isso numa validação deixando
uma safra de fora (leave-one-year-out): o RMA preserva de 100% a 107% do
desvio padrão, contra 77% a 87% da OLS, com viés < 0,002.

Como rodar
----------
    python etapa2_preprocessamento/2b_calibracao_ndvi.py

Saídas:
    dados/processados/ndvi_harmonizado_rma.csv  (série final, escala do L8)
    resultados/2b_calibracao_ndvi.json          (coeficientes e validação)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DADOS_BRUTOS, DADOS_PROC, salvar_json  # noqa: E402

INDICES = ["NDVI_Vigor_Maximo", "NDVI_Estresse_Maximo", "NDVI_Florada"]
PARES = {
    "L7->L8": ("L7", "L8", list(range(2014, 2023))),
    "L5->L7": ("L5", "L7", [a for a in range(2000, 2012) if a != 2003]),
}

d = pd.read_csv(DADOS_BRUTOS / "ndvi_landsat_por_sensor_1989_2023.csv")
# Florada sem imagem limpa: média do mesmo sensor e safra nos outros municípios
d["NDVI_Florada"] = d.groupby(["Sensor", "Ano"]).NDVI_Florada.transform(lambda s: s.fillna(s.mean()))
largo = d.pivot_table(index=["Ano", "Municipio"], columns="Sensor", values=INDICES + ["N_obs_medio_pixel"])


def ols(x, y):
    b = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    return y.mean() - b * x.mean(), b


def rma(x, y):
    r = np.corrcoef(x, y)[0, 1]
    b = np.sign(r) * y.std(ddof=1) / x.std(ddof=1)
    return y.mean() - b * x.mean(), b


def pares_sobrepostos(indice, origem, destino, anos):
    sub = largo.loc[largo.index.get_level_values(0).isin(anos), [(indice, origem), (indice, destino)]].dropna()
    sub.columns = ["x", "y"]
    return sub


# ---------------------------------------------------------------------------
# Parte 1: validação OLS x RMA, deixando uma safra de fora
# ---------------------------------------------------------------------------
validacao = {}
for indice in INDICES:
    for par, (origem, destino, anos) in PARES.items():
        sub = pares_sobrepostos(indice, origem, destino, anos)
        safra = sub.index.get_level_values(0)
        res = {"N": len(sub), "R2": round(float(np.corrcoef(sub.x, sub.y)[0, 1] ** 2), 3)}
        for nome, ajuste in [("OLS", ols), ("RMA", rma)]:
            a, b = ajuste(sub.x, sub.y)
            previsto = []
            for s in sorted(set(safra)):
                fora = safra == s
                aa, bb = ajuste(sub.x[~fora], sub.y[~fora])
                previsto.append(aa + bb * sub.x[fora])
            p = pd.concat(previsto).reindex(sub.index)
            res[nome] = {
                "a": round(float(a), 3), "b": round(float(b), 3),
                "MAE_safra_fora": round(float(np.abs(p - sub.y).mean()), 4),
                "vies": round(float((p - sub.y).mean()), 4),
                "razao_desvio_padrao": round(float(p.std() / sub.y.std()), 3),
            }
        validacao[f"{indice} | {par}"] = res

# ---------------------------------------------------------------------------
# Parte 2: aplica o RMA e monta a série final na escala do Landsat 8
# ---------------------------------------------------------------------------
coef = {}
for indice in INDICES:
    for par, (origem, destino, anos) in PARES.items():
        sub = pares_sobrepostos(indice, origem, destino, anos)
        coef[(indice, par)] = rma(sub.x, sub.y)

linhas = []
for (ano, mun), r in largo.iterrows():
    sensor = "L8" if ano >= 2014 else ("L7" if ano >= 2000 else "L5")
    linha = {"Ano": ano, "Municipio": mun, "Sensor_base": sensor, "N_obs": r[("N_obs_medio_pixel", sensor)]}
    for indice in INDICES:
        v = r[(indice, sensor)]
        if sensor == "L5":                      # L5 -> L7
            a, b = coef[(indice, "L5->L7")]
            v = a + b * v
        if sensor in ("L5", "L7"):              # L7 -> L8
            a, b = coef[(indice, "L7->L8")]
            v = a + b * v
        linha[indice] = v
    linhas.append(linha)
serie = pd.DataFrame(linhas)
serie["NDVI_Amplitude"] = serie.NDVI_Vigor_Maximo - serie.NDVI_Estresse_Maximo
serie.to_csv(DADOS_PROC / "ndvi_harmonizado_rma.csv", index=False)

salvar_json({"validacao_safra_fora": validacao,
             "coeficientes_rma": {f"{k[0]} | {k[1]}": {"a": float(v[0]), "b": float(v[1])} for k, v in coef.items()}},
            "2b_calibracao_ndvi.json")

print("Validação deixando uma safra de fora (razão dp = variabilidade preservada):")
for k, v in validacao.items():
    print(f"  {k:32s} R²={v['R2']:.2f}  OLS dp {v['OLS']['razao_desvio_padrao']:.2f}  "
          f"RMA dp {v['RMA']['razao_desvio_padrao']:.2f}  RMA viés {v['RMA']['vies']:+.4f}  "
          f"RMA MAE {v['RMA']['MAE_safra_fora']:.3f}")
print(f"\nSérie harmonizada: {len(serie)} linhas, safras {serie.Ano.min()}-{serie.Ano.max()}")
print(serie.groupby("Sensor_base").size().to_dict())
