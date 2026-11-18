"""
Roda, em ordem, todas as etapas que funcionam sem internet (2 a 5) e os
experimentos. A etapa 1 (coleta) precisa de internet e do Google Earth
Engine; os arquivos que ela gera já estão em dados/brutos/.

    python executar_tudo.py              # etapas 2 a 5 (~4 min)
    python executar_tudo.py --com-experimentos   # + experimentos (~6 min a mais)
"""
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
ETAPAS = [
    "etapa2_preprocessamento/2a_auditoria_e_matriz.py",
    "etapa2_preprocessamento/2b_calibracao_ndvi.py",
    "etapa3_base/3_montar_base.py",
    "etapa4_modelagem/4a_random_forest_kfold_aninhado.py",
    "etapa4_modelagem/4b_previsores_intrasafra.py",
    "etapa5_avaliacao/5a_importancia_mdi_shap.py",
    "etapa5_avaliacao/5b_dano_por_evento.py",
    "etapa5_avaliacao/5c_erros_e_diagnosticos.py",
    "etapa5_avaliacao/5d_figuras_metodologia.py",
    "etapa5_avaliacao/5e_numeros_do_texto.py",
]
EXPERIMENTOS = [
    "experimentos/e1_validacao_espacial_e_temporal.py",
    "experimentos/e2_kfold_por_safra.py",
    "experimentos/e3_residualizacao.py",
    "experimentos/e4_deficit_hidrico_no_modelo.py",
]

lista = ETAPAS + (EXPERIMENTOS if "--com-experimentos" in sys.argv else [])
for script in lista:
    print(f"\n{'=' * 78}\n>>> {script}\n{'=' * 78}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, str(RAIZ / script)], check=True, cwd=RAIZ)
    print(f"[{time.time() - t0:.0f} s]")
print("\nPronto. Tabelas e números em resultados/, figuras em resultados/figuras/.")
