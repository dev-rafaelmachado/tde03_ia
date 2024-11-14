## Requisitos

- Python 3.9 ou superior
- Dependências (instalar via `pip`):

```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

- **data**: Contém o módulo `dataset` que carrega os dados de entrada.
- **utils**: Contém funções auxiliares para avaliação dos modelos, salvamento dos resultados, etc.
- **ga**: Contém o algoritmo genético e suas operações (seleção, crossover, etc.).
- **main.py**: O arquivo principal para rodar o projeto.

## Como Rodar o Projeto

Obs: Lembre de adicionar a pasta `static/input` com os arquivos de dados de treinamento e teste antes de rodar o projeto.

Para rodar o projeto, execute o arquivo `main.py` diretamente da linha de comando.

### Execução Básica

```bash
python src/main.py
```

Isso irá rodar todos os modelos disponíveis (`baseline`, `wrapper`, e `ga`) com os parâmetros padrão.

### Execução com Parâmetros

Você pode controlar quais modelos devem ser executados e o número de features a serem usadas passando argumentos na linha de comando.

A sintaxe dos parâmetros é a seguinte:

```bash
python src/main.py <exec_models> <num_features>
```

- **<exec_models>**: Um número de 3 dígitos onde cada dígito controla a execução de um modelo. Cada posição controla a execução de um modelo conforme abaixo:
  - 1ª posição (centenas) → **Baseline**: Se for 1, executa o modelo baseline.
  - 2ª posição (dezenas) → **Wrapper**: Se for 1, executa o modelo com wrapper.
  - 3ª posição (unidades) → **GA (Algoritmo Genético)**: Se for 1, executa o modelo com algoritmo genético.

  Exemplo: `python src/main.py 111` vai executar todos os modelos.

  Exemplo 2: `python src/main.py 100` vai executar apenas o modelo baseline.

- **<num_features>**: Um número inteiro que define a quantidade de features a ser utilizada no treinamento do modelo. Caso não seja passado, o valor será `None`.

  Exemplo: `python src/main.py 111 10` vai executar todos os modelos com 10 features.

### Exemplos de Execução

1. **Executar todos os modelos com todas as features**:
   ```bash
   python src/main.py 111
   ```

2. **Executar apenas o modelo baseline com todas as features**:
   ```bash
   python src/main.py 100
   ```

3. **Executar o modelo GA (algoritmo genético) com 10 features**:
   ```bash
   python src/main.py 001 10
   ```

4. **Executar apenas o modelo wrapper com 15 features**:
   ```bash
   python src/main.py 010 15
   ```

### Resultados

Os resultados serão salvos automaticamente no diretório `/static/output`.

## Funções Principais

- **`load_data()`**: Carrega os dados de treinamento e teste.
- **`evaluate_models()`**: Avalia os modelos de acordo com os parâmetros passados.
- **`save_results()`**: Salva os resultados da avaliação no diretório especificado.

## Bye bye!

Após a execução, o programa imprime "Bye bye!" indicando que o processo foi concluído.
