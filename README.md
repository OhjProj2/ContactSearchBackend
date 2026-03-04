
# ContactSearch

An app for searching contact information from websites.

The progam uses LLM to find contact information from list of websites. For example you can provide list of schools and program tries to find principals email, phone number etc.

## Authors

- [@Prshkv](https://www.github.com/Prshkv)
- [@VeeraElo](https://www.github.com/VeeraElo)
- [@EnergyJoe](https://www.github.com/Energyjoe)
- [@Matimane](https://www.github.com/Matimane)
- [@Eetuhellberg](https://www.github.com/eetuhellberg)

## Tech Stack

**Client:** JavaScript/Typescript React

**Server:** Python, FastAPI

Database: MongoDB

## Backend

### Initialization

1. Install uv package manager.
1. Run uv sync.
1. Run uv run playwright install.

### Running the backend

1. uv run fastapi dev app/main.py

### External documentation

[Langchain Ollama documentation](https://reference.langchain.com/python/integrations/langchain_ollama/?_gl=1*1ggao8h*_gcl_au*ODU1NTg1NTc2LjE3Njk3NjE0NDQ.*_ga*Mzg5Mzc0NTU4LjE3Njk3NjE0NDQ.*_ga_47WX3HKKY2*czE3NzAxMTQyOTUkbzIkZzEkdDE3NzAxMTQ0NzEkajUkbDAkaDA.#langchain_ollama.ChatOllama.base_url)

## Database

We're using MongoDB because program's data schema is evolving constantly and there is lots of variation in contact details naturally.

### Installing in Docker container

Before installing MongoDB, install MongoDB Shell (mongosh). Instructions for installation:

[Install mongosh](https://www.mongodb.com/docs/mongodb-shell/install/?operating-system=windows&windows-installation-method=msiexec)

### [Install MongoDB Community with Docker](https://www.mongodb.com/docs/v7.0/tutorial/install-mongodb-community-with-docker/)

1. Pull the docker image.
   1. docker pull mongodb/mongodb-community-server:latest
2. Run the docker image.
   1. docker run --name mongodb -p 27017:27017 -d mongodb/mongodb-community-server:latest

### Using PyMongo library

PyMongo is the recommended way to work with MongoDB from Python. Check the newest documentation from PyMongo's homepage: [PyMongo documentation](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/).
