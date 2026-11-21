"""
EXPERIMENTO 6: o problema está no modelo ou no dado do IBGE?
=============================================================

Dois testes de reserva para a banca (não entram no texto do TCC).

Teste A: dados limpos
    Retira da base as observações suspeitas do IBGE e repete os cenários 1 a 4
    (K-Fold aninhado e safra inteira fora):
      * rendimento igual ao da safra anterior no mesmo município (repetição);
      * rendimento praticamente igual (diferença de até 2 kg/ha) ao de pelo
        menos outros 3 municípios na mesma safra (valor regional, como os
        1.141 kg/ha de 1996).

Teste B: sinal plantado
    Soma à produtividade real um efeito climático CONHECIDO e verifica se o
    modelo com clima (Cenário 2) o detecta em relação à base (Cenário 1):
        efeito = s × z,  z = média padronizada de [mínima do inverno (t−1),
                                                 chuva na florada (t−1)]
    com s = 0, 1, 2 e 4 sacas/ha (desvio padrão do efeito). As variáveis de
    defasagem (Y(t−1), M(t), A(t), Y(t−2)) são recalculadas com a nova série.
    Se o Cenário 2 passa a errar menos que o 1 quando há sinal, o método
    funciona e a falta de ganho no TCC vem do dado, e não do modelo.

Como rodar
----------
    python experimentos/e6_dados_limpos_e_sinal_plantado.py   (~1 h, 2 núcleos)

Saída: resultados/e6_dados_limpos_e_sinal_plantado.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ALVO, ANO, CLIMA, CONTROLES, DADOS_BRUTOS, KG_POR_SACA, MUN, NDVI, RESULTADOS,  # noqa: E402
                    busca_em_grade, carregar_base, holm, metricas, validacao_aninhada, wilcoxon_obs_e_safra)

CENARIOS = {"C1": CONTROLES, "C2": CONTROLES + CLIMA, "C3": CONTROLES + NDVI, "C4": CONTROLES + CLIMA + NDVI}
SAIDA = RESULTADOS / "e6_dados_limpos_e_sinal_plantado.json"
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def safra_fora(d, variaveis):
    X, y = d[variaveis], d[ALVO]
    pred = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, d[ANO]):
        pred[te] = busca_em_grade().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    return pred


def avaliar(d, cenarios, validacoes=("kfold", "safra_fora")):
    """MAE de cada cenário, contra a persistência M(t) e contra o Cenário 1."""
    y = d[ALVO].values
    erro_M = np.abs(y - d.media_movel_3anos.values)
    r = {"N": len(d), "safras": int(d[ANO].nunique()), "persistencia_M": metricas(y, d.media_movel_3anos)}
    for val in validacoes:
        rv, erros = {}, {}
        for c in cenarios:
            p = validacao_aninhada(d, CENARIOS[c])[0] if val == "kfold" else safra_fora(d, CENARIOS[c])
            erros[c] = np.abs(y - p)
            rv[c] = {**metricas(y, p), "wilcoxon_vs_M": wilcoxon_obs_e_safra(erro_M, erros[c], d[ANO])}
            log(f"  {val:10s} {c}: MAE {rv[c]['MAE']:.2f}")
        outros = [c for c in cenarios if c != "C1"]
        brutos = []
        for c in outros:
            rv[c]["wilcoxon_vs_C1"] = wilcoxon_obs_e_safra(erros["C1"], erros[c], d[ANO])
            brutos.append(rv[c]["wilcoxon_vs_C1"][1])
        for c, p in zip(outros, holm(brutos)):
            rv[c]["holm_safra_vs_C1"] = round(float(p), 4)
        r[val] = rv
    return r


def salvar(res):
    SAIDA.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


res = json.loads(SAIDA.read_text(encoding="utf-8")) if SAIDA.exists() else {}
base = carregar_base()

# ------------------------------------------------------------------ Teste A
if "A_dados_limpos" not in res:
    bruto = pd.read_csv(DADOS_BRUTOS / "producao_ibge_pam_1988_2023.csv").sort_values([MUN, ANO])
    bruto["repete"] = bruto.groupby(MUN)[ALVO].diff().eq(0)
    rep = bruto.set_index([MUN, ANO])["repete"]
    chave = list(zip(base[MUN], base[ANO]))
    f_rep = np.array([bool(rep.get(k, False)) for k in chave])
    f_reg = np.zeros(len(base), bool)
    for ano, g in base.groupby(ANO):
        v = g[ALVO].values
        iguais = (np.abs(v[:, None] - v[None, :]) <= 2).sum(1) - 1
        f_reg[g.index[iguais >= 3]] = True
    suspeita = f_rep | f_reg
    limpa = base[~suspeita].reset_index(drop=True)
    log(f"Teste A: {f_rep.sum()} repetidos, {f_reg.sum()} regionais, {suspeita.sum()} retirados; N = {len(limpa)}")
    res["A_dados_limpos"] = {
        "retirados": {"repeticao_safra_anterior": int(f_rep.sum()), "valor_regional": int(f_reg.sum()),
                      "total": int(suspeita.sum()),
                      "por_safra": {int(a): int(n) for a, n in base[suspeita].groupby(ANO).size().items()}},
        **avaliar(limpa, list(CENARIOS)),
    }
    salvar(res)

# ------------------------------------------------------------------ Teste B
bruto = pd.read_csv(DADOS_BRUTOS / "producao_ibge_pam_1988_2023.csv")[[MUN, ANO, ALVO]]
zs = lambda s: (s - s.mean()) / s.std()
z = (zs(base["Tmin_Abs_Inverno_Risco_Geada_t1"]) + zs(base["Precip_Florada_t1"]))
z = (z / z.std()).values
var_entre = base.assign(z=z).groupby(ANO).z.mean().var() / np.var(z, ddof=1)
res.setdefault("B_sinal_plantado", {"variaveis": ["Tmin_Abs_Inverno_Risco_Geada_t1", "Precip_Florada_t1"],
                                    "fracao_da_variancia_do_efeito_entre_safras": round(float(var_entre), 2)})
for s in [0, 1, 2, 4]:
    k = f"s={s}"
    if k in res["B_sinal_plantado"]:
        continue
    log(f"Teste B, efeito de {s} sacas/ha")
    efeito = pd.Series(s * KG_POR_SACA * z, index=pd.MultiIndex.from_frame(base[[MUN, ANO]]))
    serie = bruto.copy()
    serie[ALVO] = serie[ALVO] + [efeito.get((m, a), 0.0) for m, a in zip(serie[MUN], serie[ANO])]
    serie = serie.sort_values([MUN, ANO])
    g = serie.groupby(MUN)[ALVO]
    serie["produtividade_t_1"] = g.shift(1)
    serie["media_movel_3anos"] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    serie["anomalia_bienal_t_1"] = serie.produtividade_t_1 - serie.media_movel_3anos
    serie["y_t2"] = g.shift(2)
    novas = [ALVO, "produtividade_t_1", "media_movel_3anos", "anomalia_bienal_t_1", "y_t2"]
    d = base.drop(columns=novas).merge(serie[[MUN, ANO] + novas], on=[MUN, ANO], how="left")
    d = d[base.columns]
    res["B_sinal_plantado"][k] = avaliar(d, ["C1", "C2"])
    salvar(res)
log("fim")
