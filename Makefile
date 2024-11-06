.PHONY: up down

# Name of the Docker Compose file
COMPOSE_FILE ?= docker-compose.yml

# Name of the Docker Compose project
COMPOSE_PROJECT_NAME ?= document-scanner

up:
	docker-compose -f $(COMPOSE_FILE) -p $(COMPOSE_PROJECT_NAME) up -d --build

down:
	docker-compose -f $(COMPOSE_FILE) -p $(COMPOSE_PROJECT_NAME) down