
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

### Local setup: Installing in Docker container

Before installing MongoDB, install MongoDB Shell (mongosh). Instructions for installation:

[Install mongosh](https://www.mongodb.com/docs/mongodb-shell/install/?operating-system=windows&windows-installation-method=msiexec)

### [Install MongoDB Community with Docker](https://www.mongodb.com/docs/v7.0/tutorial/install-mongodb-community-with-docker/)

1. Pull the docker image.
   1. docker pull mongodb/mongodb-community-server:latest
2. Run the docker image.
   1. docker run --name mongodb -p 27017:27017 -d mongodb/mongodb-community-server:latest

### MongoDB Atlas cloud service

Create MongoDB cloud service at [MongoDB Atlas](https://account.mongodb.com/account/login?nds=true). After creating a cluster:

1. Check the current Rahti outgoing customer traffic IP address from [Docs CSC Security Guide](https://docs.csc.fi/cloud/rahti/security-guide/).
2. Go to Network Access -> IP Access List.
3. Whitelist the Rahti IP address.

### Using PyMongo library

PyMongo is the recommended way to work with MongoDB from Python. Check the newest documentation from PyMongo's homepage: [PyMongo documentation](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/).

## Deployment to Rahti

### Using other than main branch

1. Go to Builds -> BuildConfigs -> YAML
2. Add/edit following line.

```yaml
  source:
    type: Git
    git:
      uri: 'https://github.com/OhjProj2/ContactSearchBackend'
      ref: sprint2mongodb <-- INSERT BRANCH HERE
```

### Timeout setting

Rahti default timeout is 60 s. Queries can take lot longer than that. To set timeout, edit route YAML file. Add or edit the line shown:

```yaml
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: my-route
  annotations:
    haproxy.router.openshift.io/timeout: 60s <-- TIMEOUT SETTING ON THIS LINE
```

### Dockerfile

A ready playwright image is used in the Dockerfile. Playwrights [Docker page](https://playwright.dev/docs/docker) has up-to-date information on the image.

When Rahti deploys a pod, it creates a random user identifier (UID). Containers aren't allowed to run as root.

For an image to support running as an arbitrary user, directories and files that are written to by processes in the image must be owned by the root group and be read/writable by that group. Files to be executed must also have group execute permissions.

[Redhat's documentation on Openshift platform images](https://docs.redhat.com/en/documentation/openshift_container_platform/4.13/html/images/creating-images#use-uid_create-images) has detailed instructions on how to configure containers to work with this restriction.

## Tests and Continuous Integration
This project includes unit and integration tests for the FastAPI backend, ensuring the application works correctly. Tests are written using pytest and cover the following areas:

**Unit-tests:** Ensures correct string output and validation

**Integration-tests:** Mocks external dependencies (LLM, web fetch, MongoDB). Verifies correct HTTP status codes and response structure

### Running tests locally

1. Run 'uv run pytest'.

## GitHub Actions CI
The project uses **GitHub Actions** for continuous integration. All tests run automatically on every push and pull request to the main branch.
- Workflow is located at .github/workflows/ci.yml
- Sets up Python and a virtual environment
- Installs dependencies and runs unit and integration tests
- Test results are reported directly in GitHub Actions
