# Contribuição para o Trabalho 1 — Random Forest (cenário binário)

Autor da etapa de modelagem: Daniel Sousa de Sena.
Dados: cenário binário pré-processado pelo grupo, y=1 Dengue e y=0 Descartado/Inconclusivo.

## O que foi executado
O script 05_random_forest_binario.py utiliza os arquivos data/processed/binario/{train,val,test}_tree.parquet, gerados pela etapa 02 do repositório do grupo. A amostragem aleatória foi reproduzível (seed 42): 99.535 linhas de treino, 29.845 de validação e 79.817 de teste. Foram excluídas as três colunas hosp_* para o cenário de triagem; restaram 55 preditores. Não foram usadas variáveis laboratoriais nem os desfechos finais como preditores.

Floresta com 200 árvores, max_features=sqrt, class_weight=balanced e 2 ou 10 amostras mínimas por folha. Escolha pela AUC na validação: folha 2 = 0,7210; folha 10 = 0,7264. A opção folha 10 foi treinada novamente usando as amostras de treino e validação e avaliada no teste com limiar 0,5. O DummyClassifier(strategy=prior) é a referência, avaliada no MESMO teste.

## Resultado no teste
Modelo         n      Acurácia  Bal.Acc  Precisão dengue  Recall dengue  F1 dengue  F1 macro  AUC ROC  AP
Dummy          79817  0,6574    0,5000   0,6574           1,0000         0,7933     0,3966    0,5000   0,6574
Random Forest  79817  0,6694    0,6641   0,7874           0,6810         0,7303     0,6516    0,7230   0,8170

Matriz de confusão (Random Forest): VN=17.695, FP=9.651, FN=16.737, VP=35.734. Cerca de 31,9% dos casos rotulados como Dengue não foram identificados ao limiar 0,5. A classe 0 inclui inconclusivos, portanto não deve ser descrita como ausência comprovada de dengue. O classificador Dummy prevê a classe majoritária para todos, o que explica recall=1 e F1 positivo alto apesar de não identificar a classe 0.

## Arquivos
05_random_forest_binario.py: código para reproduzir a etapa de modelagem.
resultados/validacao_rf.csv: comparação das duas configurações na validação.
resultados/metricas_teste.csv: métricas no teste para RF e Dummy.
resultados/matriz_confusao_rf.png, curva_roc.png, curva_pr.png: figuras para o relatório.
resultados/importancia_atributos_rf.csv: importância exploratória por impureza (não é causalidade).
resultados/protocolo_rf.json: configurações e tamanhos amostrados.

## Como reproduzir
Colocar 05_random_forest_binario.py em src/ da pasta arbovirus-brasil-eda-main e manter data/processed/binario/{train,val,test}_tree.parquet nas posições esperadas. Na raiz do projeto, no PowerShell:
.\.venv\Scripts\python.exe .\src\05_random_forest_binario.py
Saídas: reports/random_forest_binario/. Os Parquets não estão incluídos neste ZIP devido ao volume; a pessoa que preparou os dados pode compartilhá-los à parte.

## Pontos para o relatório
A comparação válida entre modelos deste pacote é RF vs Dummy, no mesmo conjunto de teste. O baseline 03_sanity_baseline.py usa outra amostra e inclui hospitalização, portanto não é comparação direta. A divisão original foi aleatória e estratificada (70/15/15), com mistura dos anos; resultados futuros podem diferir. Importâncias elevadas de idade, mês de notificação, dor retro-orbital e região não provam relação causal. No código original de pré-processamento, mes_sin é calculado a partir de DT_NOTIFIC, isto é, mês de notificação.
