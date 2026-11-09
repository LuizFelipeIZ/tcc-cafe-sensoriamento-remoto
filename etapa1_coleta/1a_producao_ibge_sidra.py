"""
ETAPA 1a: produção de café por município (IBGE, PAM, Tabela 1613)
====================================================================

O que faz
---------
Baixa da API pública do SIDRA a área colhida (ha), a quantidade produzida (t)
e o rendimento médio (kg/ha) do café nos 10 municípios, de 1988 a 2023.

Por que dois produtos
---------------------
A Tabela 1613 só separa o café arábica do canéfora a partir de 2013. Por isso:
  * 1988 a 2012: "Café (em grão) Total"
  * 2013 a 2023: "Café (em grão) Arábica"
Na região estudada praticamente todo o café é arábica, então a emenda é
coerente, mas ela é citada como limitação no TCC.

Como rodar
----------
Precisa de internet (rode no seu computador ou no Colab):
    pip install pandas requests
    python etapa1_coleta/1a_producao_ibge_sidra.py

Saída: dados/brutos/producao_ibge_pam_1988_2023.csv
(o arquivo já está no repositório, então esta etapa é opcional)

Observação: a primeira versão deste script lia o ano pela posição da
dimensão (D2N) e falhava com "invalid literal for int(): 'Área colhida'".
Agora o ano é localizado pelo cabeçalho que a própria API devolve.
"""
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DADOS_BRUTOS  # noqa: E402

MUNICIPIOS = [
    "Arceburgo", "Cabo Verde", "Guaranésia", "Guaxupé", "Itamogi",
    "Jacuí", "Juruaia", "Monte Belo", "Muzambinho", "Nova Resende",
]
UF = "MG"
TABELA = "1613"
PERIODOS = [
    # (produto procurado no descritor da tabela, ano inicial, ano final)
    ("café (em grão) total", 1988, 2012),
    ("café (em grão) arábica", 2013, 2023),
]
VARIAVEIS = {
    "Area_Colhida_ha": "Área colhida",
    "Producao_t": "Quantidade produzida",
    "Rendimento_kg_ha": "Rendimento médio da produção",
}
SAIDA = DADOS_BRUTOS / "producao_ibge_pam_1988_2023.csv"


def sem_acento(texto):
    return unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII").lower().strip()


def codigos_municipios():
    """Resolve o código IBGE de cada município pelo nome (API de Localidades)."""
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{UF}/municipios"
    todos = requests.get(url, timeout=30).json()
    mapa = {sem_acento(m["nome"]): m["id"] for m in todos}
    return {nome: mapa[sem_acento(nome)] for nome in MUNICIPIOS}


def descritor():
    url = f"https://apisidra.ibge.gov.br/DescritoresTabela/t/{TABELA}"
    return requests.get(url, timeout=30).json()


def codigo_produto(desc, produto):
    """Acha a classificação e a categoria do produto (ex.: café arábica)."""
    for classe in desc.get("classificacoes", desc.get("Classificacoes", [])):
        for cat in classe.get("categorias", classe.get("Categorias", [])):
            nome = cat.get("nome", cat.get("Nome", ""))
            if sem_acento(produto) in sem_acento(nome):
                return classe.get("id", classe.get("Id")), cat.get("id", cat.get("Id"))
    raise ValueError(f"Produto '{produto}' não encontrado no descritor da Tabela {TABELA}.")


def codigos_variaveis(desc):
    saida = {}
    for coluna, nome in VARIAVEIS.items():
        for v in desc.get("variaveis", desc.get("Variaveis", [])):
            if sem_acento(nome) in sem_acento(v.get("nome", v.get("Nome", ""))):
                saida[coluna] = v.get("id", v.get("Id"))
                break
    return saida


def baixar(cod_mun, cod_var, cod_classe, cod_cat, ano_ini, ano_fim):
    url = (f"https://apisidra.ibge.gov.br/values/t/{TABELA}/n6/{cod_mun}"
           f"/p/{ano_ini}-{ano_fim}/v/{cod_var}/c{cod_classe}/{cod_cat}")
    resposta = requests.get(url, timeout=60).json()
    cabecalho, linhas = resposta[0], resposta[1:]
    # a coluna do ano é a que o cabeçalho chama de "Ano" (ex.: D3N)
    chave_ano = next(k for k, v in cabecalho.items() if k.endswith("N") and sem_acento(v).startswith("ano"))
    registros = []
    for linha in linhas:
        valor = linha["V"]
        valor = pd.to_numeric(valor, errors="coerce")   # "...", "-" e "X" viram vazio
        registros.append({"Ano": int(linha[chave_ano]), "Valor": valor})
    return registros


def main():
    municipios = codigos_municipios()
    desc = descritor()
    variaveis = codigos_variaveis(desc)
    print("Municípios:", municipios)
    print("Variáveis:", variaveis)

    tabelas = []
    for produto, ano_ini, ano_fim in PERIODOS:
        cod_classe, cod_cat = codigo_produto(desc, produto)
        print(f"\n{produto}: classificação {cod_classe}, categoria {cod_cat}, {ano_ini}-{ano_fim}")
        for nome, cod_mun in municipios.items():
            partes = []
            for coluna, cod_var in variaveis.items():
                reg = pd.DataFrame(baixar(cod_mun, cod_var, cod_classe, cod_cat, ano_ini, ano_fim))
                partes.append(reg.rename(columns={"Valor": coluna}).set_index("Ano"))
                time.sleep(0.3)   # gentileza com a API pública
            bloco = pd.concat(partes, axis=1).reset_index()
            # mesma grafia sem acento usada nas outras bases (Guaranesia, Guaxupe, Jacui)
            bloco.insert(0, "Municipio", sem_acento(nome).title())
            bloco["Produto"] = produto
            tabelas.append(bloco)
            print(f"  {nome}: {len(bloco)} safras")

    df = pd.concat(tabelas, ignore_index=True)
    df = df.sort_values(["Municipio", "Ano"]).reset_index(drop=True)
    df.drop(columns="Produto").to_csv(SAIDA, index=False)
    print(f"\nSalvo: {SAIDA} ({len(df)} linhas)")


if __name__ == "__main__":
    main()
