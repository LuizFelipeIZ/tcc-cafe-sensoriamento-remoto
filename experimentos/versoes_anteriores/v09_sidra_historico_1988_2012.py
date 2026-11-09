"""
=====================================================================
TCC - Extensão Histórica da Série de Produção Cafeeira via SIDRA/IBGE
=====================================================================

Este script busca, diretamente da API pública do IBGE, os dados de
produção de café (arábica) para os 10 municípios da microrregião de
São Sebastião do Paraíso, cobrindo o período ANTERIOR a 2013 (o início
da sua base atual), permitindo estender a série histórica e incluir
mais eventos climáticos extremos (geadas de 1994 e 2000, por exemplo).

O script NÃO exige que você descubra códigos manualmente:
  1. Resolve os códigos IBGE dos 10 municípios automaticamente, por nome,
     via API de Localidades do IBGE.
  2. Descobre automaticamente o código de classificação de "café (em
     grão) Arábica" dentro da Tabela 1613, consultando o descritor
     oficial da tabela em tempo real.
  3. Faz o download dos dados de Área Colhida, Quantidade Produzida e
     Rendimento Médio para o intervalo de anos definido abaixo.
  4. Salva um CSV pronto para você mesclar com a sua matriz atual
     (TCC_Matriz_Fenologica_final.xlsx), usando os mesmos nomes de
     coluna já usados no seu pipeline.

IMPORTANTE: este script precisa de acesso à internet e deve ser rodado
no SEU computador (não em ambientes com rede restrita). Rode com:

    pip install pandas requests openpyxl
    python3 extrair_dados_sidra_historico.py

Fonte dos dados: IBGE, Sistema IBGE de Recuperação Automática - SIDRA,
Tabela 1613 (Produção Agrícola Municipal, Lavouras Permanentes).
"""

import requests
import pandas as pd
import time

# =====================================================================
# CONFIGURAÇÃO
# =====================================================================
MUNICIPIOS_ALVO = [
    "Arceburgo", "Cabo Verde", "Guaranésia", "Guaxupé", "Itamogi",
    "Jacuí", "Juruaia", "Monte Belo", "Muzambinho", "Nova Resende"
]
UF = "MG"

TABELA_SIDRA = "1613"          # Lavouras permanentes, série 1974-2023
ANO_INICIO = 1988              # ano inicial da extensão (ajuste se quiser ir mais longe: 1974)
ANO_FIM = 2012                 # até o ano anterior ao início da sua base atual (2013)

# Variáveis da Tabela 1613 (nomes oficiais, resolvidos automaticamente
# a partir do descritor - não são hardcoded por número aqui)
NOMES_VARIAVEIS_DESEJADAS = {
    "area_colhida": "Área colhida",
    "quantidade_produzida": "Quantidade produzida",
    "rendimento_medio": "Rendimento médio da produção",
}

PRODUTO_DESEJADO = "café (em grão) Arábica"  # busca por correspondência parcial, case-insensitive

ARQUIVO_SAIDA = "TCC_SIDRA_Historico_1988_2012.csv"


# =====================================================================
# 1. RESOLVER CÓDIGOS IBGE DOS MUNICÍPIOS AUTOMATICAMENTE
# =====================================================================
def obter_codigos_municipios(municipios, uf):
    print(f"[INFO] Buscando códigos IBGE para {len(municipios)} municípios de {uf}...")
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    todos = resp.json()

    def normalizar(s):
        import unicodedata
        s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')
        return s.lower().strip()

    mapa_nomes = {normalizar(m['nome']): m['id'] for m in todos}

    codigos = {}
    faltantes = []
    for nome in municipios:
        chave = normalizar(nome)
        if chave in mapa_nomes:
            codigos[nome] = mapa_nomes[chave]
        else:
            faltantes.append(nome)

    if faltantes:
        print(f"[AVISO] Não foi possível resolver automaticamente: {faltantes}")
        print("        Verifique a grafia exata (com acentos) na lista MUNICIPIOS_ALVO.")

    print(f"[OK] {len(codigos)} códigos resolvidos: {codigos}")
    return codigos


# =====================================================================
# 2. DESCOBRIR O CÓDIGO DE CLASSIFICAÇÃO DO CAFÉ NA TABELA 1613
# =====================================================================
def obter_codigo_classificacao_cafe(tabela, produto_desejado):
    print(f"\n[INFO] Consultando descritor da Tabela {tabela} para localizar '{produto_desejado}'...")
    url = f"https://apisidra.ibge.gov.br/DescritoresTabela/t/{tabela}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    descritor = resp.json()

    # O descritor traz uma lista de classificações, cada uma com suas categorias.
    # Procuramos a classificação de "Produto das lavouras permanentes" e, dentro
    # dela, a categoria cujo nome corresponde ao café arábica.
    classificacoes = descritor.get("classificacoes", descritor.get("Classificacoes", []))

    codigo_classificacao = None
    codigo_categoria = None

    for c in classificacoes:
        categorias = c.get("categorias", c.get("Categorias", []))
        for cat in categorias:
            nome_cat = cat.get("nome", cat.get("Nome", ""))
            if produto_desejado.lower() in nome_cat.lower():
                codigo_classificacao = c.get("id", c.get("Id"))
                codigo_categoria = cat.get("id", cat.get("Id"))
                print(f"[OK] Encontrado: classificação={codigo_classificacao}, "
                      f"categoria={codigo_categoria} ('{nome_cat}')")
                return codigo_classificacao, codigo_categoria

    print("[ERRO] Não foi possível localizar o produto no descritor automaticamente.")
    print("       Consulte manualmente em: https://apisidra.ibge.gov.br/DescritoresTabela/t/1613")
    return None, None


# =====================================================================
# 3. DESCOBRIR OS CÓDIGOS DAS VARIÁVEIS DESEJADAS
# =====================================================================
def obter_codigos_variaveis(tabela, nomes_desejados):
    url = f"https://apisidra.ibge.gov.br/DescritoresTabela/t/{tabela}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    descritor = resp.json()

    variaveis = descritor.get("variaveis", descritor.get("Variaveis", []))
    codigos = {}
    for chave, nome_busca in nomes_desejados.items():
        for v in variaveis:
            nome_v = v.get("nome", v.get("Nome", ""))
            if nome_busca.lower() in nome_v.lower():
                codigos[chave] = v.get("id", v.get("Id"))
                break

    print(f"[OK] Códigos de variáveis resolvidos: {codigos}")
    return codigos


# =====================================================================
# 4. BUSCAR OS DADOS PARA CADA MUNICÍPIO
# =====================================================================
def buscar_dados_municipio(codigo_municipio, nome_municipio, tabela,
                            codigo_var, codigo_classificacao, codigo_categoria,
                            ano_inicio, ano_fim):
    periodo = f"{ano_inicio}-{ano_fim}"
    url = (
        f"https://apisidra.ibge.gov.br/values/"
        f"t/{tabela}/n6/{codigo_municipio}/p/{periodo}/"
        f"v/{codigo_var}/c{codigo_classificacao}/{codigo_categoria}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dados = resp.json()
        # A primeira linha do retorno da API Sidra é sempre o cabeçalho
        dados = dados[1:]
        registros = []
        for d in dados:
            registros.append({
                "Municipio": nome_municipio,
                # A ordem das dimensões D1/D2/D3... segue a ordem dos parâmetros
                # na URL (n6, p, v, c82) -> D1=Município, D2=Ano, D3=Variável, D4=Produto.
                "Ano": int(d.get("D2N")),
                "Valor": d.get("V"),
            })
        return registros
    except Exception as e:
        print(f"[ERRO] Falha ao buscar dados de {nome_municipio}: {e}")
        return []


# =====================================================================
# EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    codigos_municipios = obter_codigos_municipios(MUNICIPIOS_ALVO, UF)
    codigo_classificacao, codigo_categoria = obter_codigo_classificacao_cafe(
        TABELA_SIDRA, PRODUTO_DESEJADO
    )
    codigos_variaveis = obter_codigos_variaveis(TABELA_SIDRA, NOMES_VARIAVEIS_DESEJADAS)

    if not codigo_classificacao or not codigos_variaveis:
        print("\n[ERRO FATAL] Não foi possível resolver os parâmetros da API automaticamente.")
        print("Alternativa manual: acesse https://sidra.ibge.gov.br/tabela/1613, selecione")
        print("'café (em grão) Arábica', os 10 municípios, o período 1988-2012 e as variáveis")
        print("Área colhida / Quantidade produzida / Rendimento médio, e exporte em .csv/.xlsx.")
        exit(1)

    todos_registros = {"area_colhida": [], "quantidade_produzida": [], "rendimento_medio": []}

    for nome_mun, codigo_mun in codigos_municipios.items():
        print(f"\n[INFO] Buscando dados de {nome_mun} ({codigo_mun})...")
        for chave, codigo_var in codigos_variaveis.items():
            registros = buscar_dados_municipio(
                codigo_mun, nome_mun, TABELA_SIDRA, codigo_var,
                codigo_classificacao, codigo_categoria, ANO_INICIO, ANO_FIM
            )
            todos_registros[chave].extend(registros)
            time.sleep(0.3)  # gentileza com a API pública

    # Monta os DataFrames de cada variável e faz merge
    dfs = {}
    for chave, registros in todos_registros.items():
        if registros:
            df = pd.DataFrame(registros).rename(columns={"Valor": chave})
            dfs[chave] = df

    if not dfs:
        print("\n[ERRO] Nenhum dado retornado pela API. Verifique conexão e parâmetros.")
        exit(1)

    df_final = None
    for chave, df in dfs.items():
        if df_final is None:
            df_final = df
        else:
            df_final = df_final.merge(df, on=["Municipio", "Ano"], how="outer")

    # Renomeia para o padrão já usado no seu pipeline
    df_final = df_final.rename(columns={
        "area_colhida": "Area_Colhida_ha",
        "quantidade_produzida": "Producao_t",
        "rendimento_medio": "Rendimento_kg_ha",
    })

    df_final = df_final.sort_values(by=["Municipio", "Ano"]).reset_index(drop=True)
    df_final.to_csv(ARQUIVO_SAIDA, index=False)

    print(f"\n[SUCESSO] Dados salvos em: {ARQUIVO_SAIDA}")
    print(f"Linhas obtidas: {len(df_final)}")
    print(df_final.head(15).to_string(index=False))
    print("\nPRÓXIMO PASSO: mescle este CSV com sua matriz atual (que já tem NDVI/clima")
    print("a partir de 2013), lembrando que os anos < 2015 não terão NDVI (Sentinel-2 só")
    print("existe a partir de 2015). Use esses anos apenas no modelo climático estendido")
    print("(sem NDVI), conforme discutido na Metodologia.")
