# API schemas

## What we seek?

name (first name, last name)
occupation
role
email
telephone
address
social media (list)
company registry

## What we send to our own API

content of web page

- markdown
- full html content

max answer length
temperature
top_p

thinking: low, medium, high

system message

model

## Ollama API

# Workflow

UI -> backend

backend

- formulates system message
- processes web page
- maps the call made to backend to ollama api call

backend -> Ollama

- maps the response to frontend (JSON)

# frontend

## Parameters

- url
- checkbox
  - occupation
  - name (first name, last name)
  - email
  - telephone
  - address
  - social media
  - company reg number
  -
- list of roles/occupations

## Architecture

- react
- material ui
-

# TODO

- method that creates API call to Ollama
- method that parses frontend input to
