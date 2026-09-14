# IUP — платформа прокторинга

Локальная система прокторинга для аудитории: **сервер + панель преподавателя** и **приложение участника** с камерой, экраном и микрофоном.

Репозиторий: https://github.com/Impulse3e/IUP

```
IUP/
├── agent/          # IUP Student: лаунчер и прокторинг
├── server/         # FastAPI, SQLite, WebSocket
├── dashboard/      # панель преподавателя (/) и кабинет (/student)
├── shared/         # общие типы и константы
├── scripts/        # установка, запуск, ярлыки, бэкап
└── docker-compose.yml
```

## Демо-аккаунты

Один набор логинов для панели, кабинета и лаунчера (`scripts/seed_demo.py`).

| Роль | Email | Пароль |
|------|-------|--------|
| admin | `admin@iup.local` | `admin123` |
| teacher | `teacher@iup.local` | `teacher123` |
| proctor | `proctor@iup.local` | `proctor123` |
| student | `student@iup.local` | `student123` |

Новому студенту, которого преподаватель создаёт по email, пароль показывается **один раз**. Участник может сам зарегистрироваться: кабинет `/student` → «Создать аккаунт», или кнопка в IUP Student. Смена пароля — в кабинете, в панели преподавателя и в приложении (кнопка «Пароль»).

Пароль не короче 8 символов.

## Windows — обычный сценарий

Нужен Python 3.11+ с Tcl/Tk.

### 1. Сервер и панель преподавателя

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
scripts\create_teacher_shortcut.bat
```

Дальше ярлык **IUP Teacher** на рабочем столе (или `scripts\run_teacher.bat`):

- поднимает сервер на http://127.0.0.1:8000
- открывает панель в браузере
- окно консоли не закрывать, пока идёт экзамен

Вход: `admin@iup.local` / `admin123`

Только сервер, без браузера: `scripts\run_server.bat`.

Повторно заполнить демо-данные: `scripts\run_seed.bat`.

### 2. Участник (камера)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
scripts\create_student_shortcut.bat
```

Ярлык **IUP Student** (или `scripts\run_student.bat`) → войти тем же email → **Начать экзамен**.

Сборка одного `.exe` (без `.venv` на машине студента):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_student_windows.ps1
```

Результат: `dist\IUP Student.exe`. Настройки: `%APPDATA%\iup\`.

Готовые сборки CI: https://github.com/Impulse3e/IUP/releases

### 3. Экзамен в одном кабинете (LAN)

На компьютере-сервере узнайте IPv4 (`ipconfig`) и разрешите входящие на порт **8000**.

- преподаватель: http://192.168.x.x:8000
- студент в IUP Student: «Показать адрес сервера» → `http://192.168.x.x:8000`

Копия базы: `scripts\backup_db.bat` → `data\backups\`.

## Что умеет панель

- экзамен с окном времени (нельзя войти слишком рано; после закрытия события не пишутся)
- Live: стена камер, красная рамка при критичных событиях
- проверка: зачёт / аннулирование, комментарий, экспорт CSV / JSON / HTML
- повторный запуск агента не требует заново снимать лицо, если сессия уже подтверждена

Веб-кабинет участника (без камеры): http://127.0.0.1:8000/student

## Linux / macOS

Fish: не вызывайте `source .venv/bin/activate`. Используйте `.venv/bin/python` или скрипты.

```bash
chmod +x scripts/*.sh
./scripts/install.sh
cp -n .env.example .env

./scripts/run_server.sh
PYTHONPATH=. .venv/bin/python scripts/seed_demo.py
./scripts/run_student.sh
```

Ярлык Linux: `iup-student.desktop` (поправьте путь `Exec=`, если проект не в `~/IUP`).

## Docker (PostgreSQL + MinIO)

```bash
docker compose up --build
```

## Роли

| Роль | Возможности |
|------|-------------|
| student | экзамен через IUP Student |
| proctor | Live, проверка |
| teacher | экзамены, сессии, экспорт |
| admin | webhooks, копия БД |

## Важно

- `.env`, `.venv`, `data/` и видео в git не входят
- секрет JWT берётся из `.env` или `data/secret.key`
- LTI-заглушка выключена, пока не заданы `LTI_CLIENT_ID` и `LTI_LAUNCH_SECRET`
