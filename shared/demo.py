"""Единые демо-учётные записи для локального запуска."""

DEMO_USERS = (
    {
        "email": "admin@iup.local",
        "password": "admin123",
        "full_name": "IUP Admin",
        "role": "admin",
    },
    {
        "email": "teacher@iup.local",
        "password": "teacher123",
        "full_name": "Demo Teacher",
        "role": "teacher",
    },
    {
        "email": "proctor@iup.local",
        "password": "proctor123",
        "full_name": "Demo Proctor",
        "role": "proctor",
    },
    {
        "email": "student@iup.local",
        "password": "student123",
        "full_name": "Demo Student",
        "role": "student",
    },
)

DEMO_EXAM_TITLE = "Демо-экзамен"
DEMO_EXAM_DESCRIPTION = "Тестовый запуск IUP. Студент может записаться сам или открыть уже назначенную сессию."
