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

Parte 2: a 1ª e a 2ª safra afetada, com referência fixa
-------------------------------------------------------
Na Tabela 3, cada safra é comparada com a média das TRÊS SAFRAS ANTERIORES A
ELA. Na segunda safra após um evento, essa média já inclui as safras
danificadas, e a queda some: em 2023, M(t) é a média de 2020, 2021 e 2022, e
a Tabela 3 mostra queda de 0,0.

Por isso a parte 2 fixa a referência nas três safras ANTES DO EVENTO (a M(t)
da primeira safra afetada, em cada município) e acompanha duas safras:
    * geada de 2021 (julho, com a colheita quase terminada): 2022 e 2023,
      referência nas safras de 2019 a 2021
    * geada de 1994 (junho e julho): 1995 e 1996, referência em 1992-1994
    * seca de 2014: 2014 (seca no enchimento dos grãos) e 2015 (florada de
      set-nov de 2014, também seca), referência em 2011-2013
A geada de 2000 fica de fora: na safra de 2001, 9 dos 10 municípios repetem
o rendimento de 2000 (ver etapa 2a).
"Não explicado" é a queda menos os efeitos do clima e do NDVI.

Como rodar
----------
    python etapa5_avaliacao/5b_dano_por_evento.py   (~10 s)

Saídas: resultados/tabela3_dano_por_evento.csv,
        resultados/tabela3b_dano_safras_seguintes.csv (parte 2),
        resultados/5b_dano.json
"""
import sys
from pathlib import Path

import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CLIMA, KG_POR_SACA, NDVI, RESULTADOS, VARIAVEIS,  # noqa: E402
                    busca_em_grade, carregar_base, kfold_externo, salvar_json)

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

# SHAP fora da partição (sugestão do orientador): cada observação é explicada pelo
# modelo do K-Fold que NÃO a viu no treino, inclusive as safras dos eventos.
oof = pd.DataFrame(0.0, index=base.index, columns=VARIAVEIS)
for treino, teste in kfold_externo().split(base):
    m = busca_em_grade().fit(base.iloc[treino][VARIAVEIS], base.iloc[treino][ALVO]).best_estimator_
    oof.iloc[teste] = shap.TreeExplainer(m).shap_values(base.iloc[teste][VARIAVEIS]) / KG_POR_SACA
sv["clima_oof"] = oof[CLIMA].sum(axis=1).values
sv["ndvi_oof"] = oof[NDVI].sum(axis=1).values
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

# ---------------------------------------------------------------------------
# Parte 2: 1ª e 2ª safra afetada, com a referência fixa antes do evento
# ---------------------------------------------------------------------------
SAFRAS_SEGUINTES = [
    ("Geada de 2021", [2022, 2023]),
    ("Geada de 1994", [1995, 1996]),
    ("Seca de 2014", [2014, 2015]),
]
sv["Municipio"] = base.Municipio.values
linhas_b = []
for evento, safras in SAFRAS_SEGUINTES:
    # referência de cada município: média das 3 safras antes do evento
    ref = sv[sv[ANO] == safras[0]].set_index("Municipio").M
    for ordem, safra in enumerate(safras, start=1):
        s = sv[sv[ANO] == safra].set_index("Municipio")
        queda = s.observado - ref
        linhas_b.append({
            "Evento": evento, "Safra": safra, "Safra afetada": f"{ordem}ª",
            "Referência (3 safras antes)": round(ref.mean(), 1),
            "Safras da referência": f"{safras[0] - 3}-{safras[0] - 1}",
            "Observado": round(s.observado.mean(), 1),
            "Queda": round(queda.mean(), 1),
            "Queda (%)": round(100 * queda.mean() / ref.mean()),
            "Efeito do clima": round(s.clima.mean(), 1),
            "Efeito do NDVI": round(s.ndvi.mean(), 1),
            "Clima (fora da partição)": round(s.clima_oof.mean(), 1),
            "NDVI (fora da partição)": round(s.ndvi_oof.mean(), 1),
            "Não explicado": round(queda.mean() - s.clima.mean() - s.ndvi.mean(), 1),
            "Municípios com queda": int((queda < 0).sum()),
        })
tabela_b = pd.DataFrame(linhas_b)
tabela_b.to_csv(RESULTADOS / "tabela3b_dano_safras_seguintes.csv", index=False)

clima_por_safra = por_safra.clima.mean()
resumo = {
    "efeito_medio_absoluto_do_clima_em_uma_safra": round(float(clima_por_safra.abs().mean()), 2),
    "faixa_do_efeito_do_clima": [round(float(clima_por_safra.min()), 2), round(float(clima_por_safra.max()), 2)],
    "safras_com_inverno_mais_seco_SHAP": por_safra.Precip_Inverno_Risco_Geada_t1.mean().nsmallest(4).round(2).to_dict(),
    "safra_2003_SHAP_tmin_florada": round(float(por_safra.get_group(2003).Tmin_Abs_Florada_t1.mean()), 2),
}
salvar_json({"tabela": linhas, "safras_seguintes_referencia_fixa": linhas_b, **resumo}, "5b_dano.json")

print(tabela.drop(columns=["Variável climática de maior efeito negativo"]).to_string(index=False))
print("\nEfeito médio do clima numa safra qualquer (|sacas/ha|):", resumo["efeito_medio_absoluto_do_clima_em_uma_safra"])
print("Safras em que o inverno seco mais reduziu a estimativa:", resumo["safras_com_inverno_mais_seco_SHAP"])
print("\nParte 2: 1ª e 2ª safra afetada, frente às 3 safras antes do evento (sacas/ha):")
print(tabela_b.drop(columns=["Safras da referência"]).to_string(index=False))
