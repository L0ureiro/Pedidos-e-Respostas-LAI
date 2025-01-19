# Pedidos-e-Respostas-LAI

Classificador de texto com FastAPI que utiliza um modelo BERT para análise de pedidos e respostas LAI (Lei de Acesso à Informação).

## Descrição

Esta aplicação FastAPI implementa um modelo de classificação de texto treinado com BERT. A API recebe textos como entrada e retorna previsões de classificação, incluindo as probabilidades para cada classe possível.

## Requisitos

- Python 3.7 ou superior
- pip (gerenciador de pacotes do Python)

## Instalação

### 1. Criar e Ativar Ambiente Virtual

Primeiro, crie um ambiente virtual para isolar as dependências do projeto:

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
## Windows:
.\venv\Scripts\activate

## Linux/macOS:
source venv/bin/activate
```

### 2. Instalar Dependências

Com o ambiente virtual ativado, instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

## Executando a Aplicação

### 1. Iniciar o Servidor

Execute a aplicação usando o Uvicorn:

```bash
uvicorn main:app --reload
```

### 2. Acessar a API

Após iniciar o servidor, você pode:

- Acessar a documentação interativa (Swagger UI) em: http://127.0.0.1:8000/docs
- Testar a API diretamente através da interface Swagger
- Fazer requisições para a API usando seu cliente HTTP preferido

## Testando a API

A interface Swagger UI permite testar todas as funcionalidades da API diretamente pelo navegador. Acesse http://127.0.0.1:8000/docs para:

- Visualizar todos os endpoints disponíveis
- Testar as operações da API
- Ver exemplos de requisições e respostas
