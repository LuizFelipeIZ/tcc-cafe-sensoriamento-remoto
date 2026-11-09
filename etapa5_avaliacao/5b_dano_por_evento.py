"""
ETAPA 5b: dano climático em sacas/ha nas safras de evento (Tabela 3 do TCC)
==========================================================================

Ideia
-----
O valor SHAP de uma variável, numa observação, é quanto ela desloca a
estimativa daquela observação em relação à média do modelo. Somando os
valores SHAP das 6 variáveis climáticas, tem-se o efeito do CLIMA sobre o
rendimento estimado, em sacas/ha; somando os 4 de NDVI, o efeito do NDVI.
A média dos 10 municípios em cada safra de evento é o dano atribuído.

Para comparar, a tabela mostra também o rendimento observado e a média das
três safras anteriores, M(t). A diferença entre eles é a queda observada,
que inclui a bienalidade e não só o clima.

Eventos
-------
geadas de 1994, 2000 e 2021 (a safra afetada é a do ano seguinte) e a seca
de 2014 (com a safra de 2015).

Como rodar
----------
    python etapa5_avaliacao/5b_dano_por_evento.py   (~10 s)

Saídas: resultados/tabela3_dano_por_evento.csv, resultados/5b_dano.json
"""
import sys
from pathlib import Path

import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CLIMA, KG_POR_SACA, NDVI, RESULTADOS, VARIAVEIS,  # noqa: E402
                    busca_em_grade, carregar_base, salvar_json)

EVENTOS = [
    (1994, "Geada de 1994"), (1995, "Após a geada de 1994"),
    (2000, "Geada de 2000"), (2001, "Após a geada de 2000"),
    (2014, "Seca de 2014"), (2015, "Após a seca de 2014"),
    (2021, "Geada de 2021"), (2022, "Após a geada de 2021"),
    (2023, "Segunda safra após a geada de 2021"),
]

base = carregar_base()
modelo = busca_em_grade().fit(base[VARIAVEIS], base[ALVO]).best_estimator_
sv = pd.DataFrame(shap.TreeExplainer(modelo).shap_values(base[VARIAVEIS]) / KG_POR_SACA, columns=VARIAVEIS)
sv[ANO] = base[ANO]
sv["clima"] = sv[CLIMA].sum(axis=1)
sv["ndvi"] = sv[NDVI].sum(axis=1)
sv["observado"] = base[ALVO] / KG_POR_SACA
sv["M"] = base.media_movel_3anos / KG_POR_SACA
por_safra = sv.groupby(ANO)

linhas = []
for safra, evento in EVENTOS:
    s = por_safra.get_group(safra)
    pior = s[CLIMA].mean()
    linhas.append({
        "Safra": safra, "Evento": evento,
        "Observado": round(s.observado.mean(), 1),
        "M(t)": round(s.M.mean(), 1),
        "Observado - M(t)": round((s.observado - s.M).mean(), 1),
        "Efeito do clima": round(s.clima.mean(), 1),
        "Efeito do NDVI": round(s.ndvi.mean(), 1),
        "Variável climática de maior efeito negativo": pior.idxmin(),
        "Efeito dela": round(pior.min(), 1),
    })
tabela = pd.DataFrame(linhas)
tabela.to_csv(RESULTADOS / "tabela3_dano_por_evento.csv", index=False)

clima_por_safra = por_safra.clima.mean()
resumo = {
    "efeito_medio_absoluto_do_clima_em_uma_safra": round(float(clima_por_safra.abs().mean()), 2),
    "faixa_do_efeito_do_clima": [round(float(clima_por_safra.min()), 2), round(float(clima_por_safra.max()), 2)],
    "safras_com_inverno_mais_seco_SHAP": por_safra.Precip_Inverno_Risco_Geada_t1.mean().nsmallest(4).round(2).to_dict(),
    "safra_2003_SHAP_tmin_florada": round(float(por_safra.get_group(2003).Tmin_Abs_Florada_t1.mean()), 2),
}
salvar_json({"tabela": linhas, **resumo}, "5b_dano.json")

print(tabela.drop(columns=["Variável climática de maior efeito negativo"]).to_string(index=False))
print("\nEfeito médio do clima numa safra qualquer (|sacas/ha|):", resumo["efeito_medio_absoluto_do_clima_em_uma_safra"])
print("Safras em que o inverno seco mais reduziu a estimativa:", resumo["safras_com_inverno_mais_seco_SHAP"])
