# Datathon FIAP - Passos Magicos

Projeto de analytics e machine learning para o case da Associacao Passos Magicos. O objetivo e analisar os indicadores educacionais de 2022 a 2024 e construir uma solucao preditiva para identificar alunos em risco de defasagem.

## Entregas do Projeto

- Analise das 11 perguntas de negocio do enunciado.
- Pipeline Python para leitura, tratamento, feature engineering, treino e avaliacao do modelo.
- Notebook com as etapas tecnicas do modelo preditivo.
- Modelo treinado salvo em arquivo `.joblib`.
- Aplicacao Streamlit para consulta da probabilidade de risco.
- Material de apoio em `.docx`.

## Estrutura

```text
Datathon - FIAP/
  app.py
  requirements.txt
  README.md

  data/
    BASE DE DADOS PEDE 2024 - DATATHON.xlsx
    Dicionario Dados Datathon.pdf
    exemplo_previsao_lote.csv

  models/
    modelo_risco_defasagem.joblib

  notebooks/
    modelo_risco_defasagem.ipynb

  src/
    __init__.py
    datathon_pipeline.py

  Docx/
    Analise Educacional 2022-2024.docx
    Dores de Negocio (Codigo).docx
    Modelo Preditivo de Risco de Defasagem.docx
```

## Base de Dados

A base e o dicionario foram obtidos pelo link informado no PDF do Datathon.

- Base: `data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx`
- Dicionario: `data/Dicionario Dados Datathon.pdf`
- Arquivo de teste em lote: `data/exemplo_previsao_lote.csv`

A base contem as abas `PEDE2022`, `PEDE2023` e `PEDE2024`.

## Modelo Preditivo

O modelo estima a probabilidade de um aluno estar em risco de defasagem. A variavel alvo foi definida a partir do indicador `IAN`:

```text
Risco_Defasagem = 1 quando IAN < 7
```

Essa regra segue a interpretacao de que o IAN mede adequacao do nivel: quanto menor o valor, maior a defasagem.

Para evitar vazamento de informacao, o `IAN` e usado apenas para criar o alvo de treino. Ele nao entra como variavel de entrada do modelo.

Features utilizadas:

- `IDA`
- `IEG`
- `IPS`
- `IPP`
- `IAA`
- `IPV`
- `Saude_Academica`
- `Bem_Estar_Psico`
- `Risco_Composto`
- `Gap_Expectativa_Realidade`
- `Coerencia_Autoavaliacao`

Resultado com a base real:

- Modelo escolhido: Random Forest
- Acuracia: 0.627
- Precisao: 0.627
- Recall: 0.778
- F1-score: 0.694
- ROC-AUC: 0.631

O modelo treinado esta salvo em:

```text
models/modelo_risco_defasagem.joblib
```

## Aplicacao Streamlit

O app permite:

- previsao individual usando os indicadores `IDA`, `IEG`, `IPS`, `IPP`, `IAA` e `IPV`;
- previsao em lote por arquivo `.csv` ou `.xlsx`;
- classificacao do aluno em baixo risco, risco moderado ou alto risco;
- download do resultado da previsao em lote.

O arquivo de exemplo para teste em lote fica em:

```text
data/exemplo_previsao_lote.csv
```

## Como Executar

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Treine novamente o modelo:

```bash
python src/datathon_pipeline.py
```

Execute o app Streamlit:

```bash
python -m streamlit run app.py
```

O app local abre normalmente em:

```text
http://localhost:8501
```

## Arquivos Principais

- `src/datathon_pipeline.py`: pipeline completo de dados e modelagem.
- `notebooks/modelo_risco_defasagem.ipynb`: notebook da entrega tecnica.
- `app.py`: aplicacao Streamlit com previsao individual e previsao por arquivo.
- `models/modelo_risco_defasagem.joblib`: modelo treinado.
- `Docx/Analise Educacional 2022-2024.docx`: respostas das 11 perguntas.
- `Docx/Modelo Preditivo de Risco de Defasagem.docx`: rascunho do codigo-base do modelo.

## Pendencias Para Entrega Final

- Executar e salvar o notebook com as saidas.
- Subir o projeto no GitHub.
- Publicar o app no Streamlit Community Cloud.
- Gravar o video de ate 5 minutos.

