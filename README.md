# IUP — платформа прокторинга

Полноценная система прокторинга: **агент студента**, **сервер**, **панель проктора**.

## Архитектура

```
IUP/
├── agent/          # клиент на машине студента
├── server/         # FastAPI backend + WebSocket
├── dashboard/      # веб-панель проктора
├── shared/         # общие типы и константы
├── scripts/        # запуск и демо-данные
└── docker-compose.yml
```

## Работа без консоли

### Преподаватель — http://localhost:8000

1. Войти (`admin@iup.local` / `admin123`)
2. **Экзамены** → создать экзамен → **Управление**
3. Ввести email студента → **Создать сессию**
4. Студент увидит экзамен в своём кабинете

### Участник — два способа

**A. Приложение IUP Student (рекомендуется)**

```bash
chmod +x scripts/run_student.sh
./scripts/run_student.sh
```

Или установите ярлык на рабочий стол:

```bash
cp iup-student.desktop ~/Desktop/
# отредактируйте путь Exec= если проект не в ~/IUP
```

В приложении: войти → выбрать экзамен → **Начать экзамен**.

**B. Веб-кабинет** — http://localhost:8000/student

Войти как студент → выбрать экзамен → скачать скрипт или использовать приложение.

**C. Windows — нативное приложение**

```powershell
# В PowerShell из папки проекта:
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
scripts\run_student.bat
```

Или соберите один `.exe` со всем функционалом (на Windows):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_student_windows.ps1
# Результат: dist\IUP Student.exe (~300–600 MB)
```

Студенту достаточно одного файла `IUP Student.exe`: вход, список экзаменов, запись и прокторинг (камера, экран, микрофон). Настройки сохраняются в `%APPDATA%\iup\`.

Демо-студент: `student@iup.local` / `student123`

---

## Быстрый старт (локально)

> **Fish shell:** не используйте `source .venv/bin/activate` — это bash-скрипт.
> Вместо этого вызывайте `.venv/bin/python` и `.venv/bin/pip` напрямую
> или готовые скрипты `./scripts/run_server.sh` / `./scripts/run_agent.sh`.

```bash
# Установка (bash)
chmod +x scripts/*.sh
./scripts/install.sh

# Или в fish:
# fish scripts/install.fish

cp .env.example .env

# 1. Сервер (терминал 1)
./scripts/run_server.sh

# 2. Демо-сессия (терминал 2)
PYTHONPATH=. .venv/bin/python scripts/seed_demo.py

# 3. Агент студента (терминал 2)
./scripts/run_agent.sh --token <TOKEN_ИЗ_SEED>
```

Без скриптов (любая оболочка):

```bash
cd /home/impulse3e/IUP
.venv/bin/pip install -r agent/requirements.txt
PYTHONPATH=. .venv/bin/python -m agent.main --token <TOKEN>
```

Откройте http://localhost:8000 — панель проктора.

**Логин по умолчанию:** `admin@example.com` / `admin123`

## Docker (PostgreSQL + MinIO)

```bash
docker compose up --build
```

## Роли

| Роль | Возможности |
|------|-------------|
| student | проходит экзамен через агент |
| proctor | live-мониторинг, review |
| teacher | создание экзаменов, экспорт |
| admin | webhooks, пользователи |

Демо-пользователи (`scripts/seed_demo.py`):
- `student@example.com` / `student123`
- `teacher@example.com` / `teacher123`

## Этапы реализации

### 1. Монолит → платформа
- `agent/` — захват и анализ
- `server/` — API и WebSocket
- `shared/` — общие схемы

### 2. Сессии и роли
- JWT-авторизация
- Exam → ExamSession → Violation → Review
- RBAC по ролям

### 3. Anti-cheat
- pre-check (камера, микрофон, экран)
- identity verification (face embedding)
- process monitor (Discord, OBS, TeamViewer…)
- second monitor detection
- tamper guard (debugger)
- heartbeat

### 4. Видео и доказательства
- upload video chunks на сервер
- evidence clips при нарушениях
- local/S3 storage

### 5. Dashboard
- список сессий
- live-лента нарушений (WebSocket)
- review и экспорт CSV

### 6. Интеграции
- webhooks (`violation.created`, `session.ended`)
- LTI launch stub (`POST /api/lti/launch`)

### 7. Безопасность
- согласие перед стартом
- audit log
- `.env` для секретов
- retention_days на экзаменах

### 8. Клиент
- кроссплатформенный агент (Windows/Linux)
- `scripts/run_agent.sh`
- Docker для сервера

## API (основное)

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/auth/login` | Вход |
| POST | `/api/exams` | Создать экзамен |
| POST | `/api/exams/{id}/sessions` | Создать сессию |
| GET | `/api/sessions/token/{token}` | Данные сессии |
| POST | `/api/sessions/token/{token}/start` | Старт |
| POST | `/api/sessions/token/{token}/events` | Нарушение |
| POST | `/api/sessions/token/{token}/evidence` | Доказательство |
| WS | `/api/ws/sessions/{id}` | Live |

## Запуск агента

```bash
python -m agent.main --token <ACCESS_TOKEN> --server http://localhost:8000
```

Или через `.env`:
```env
IUP_SESSION_TOKEN=...
IUP_SERVER_URL=http://localhost:8000
```

## Legacy

Старый standalone-режим с Telegram заменён платформенной архитектурой. `app.py` перенаправляет на `agent.main`.

## Следующие шаги для production

- WebRTC live preview
- Kiosk / browser lockdown
- Code signing агента (.exe / AppImage)
- Полноценный LTI 1.3
- GPU inference и phone detection
- Horizontal scaling (Redis pub/sub для WS)
