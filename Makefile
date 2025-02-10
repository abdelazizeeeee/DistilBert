.PHONY: up down logs restart

COMPOSE_FILE ?= docker-compose.yml
COMPOSE_PROJECT_NAME ?= sentimentanalysis

up:
	docker-compose -f $(COMPOSE_FILE) -p $(COMPOSE_PROJECT_NAME) up -d --build

down:
	docker-compose -f $(COMPOSE_FILE) -p $(COMPOSE_PROJECT_NAME) down

logs:
	docker-compose -f $(COMPOSE_FILE) -p $(COMPOSE_PROJECT_NAME) logs -f

restart: down up