const API = "/api";
const state = {
  token: localStorage.getItem("iup_token") || "",
  user: null,
  sessions: [],
  exams: [],
  students: [],
  selectedSession: null,
  selectedExam: null,
  violations: [],
  ws: null,
};

const app = document.getElementById("app");
const userInfo = document.getElementById("user-info");
const nav = document.getElementById("main-nav");

document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => showView(btn.dataset.view));
});

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(`${API}${path}`, { ...options, headers });
  if (!response.ok) throw new Error(await response.text());
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

function setupNav() {
  const role = state.user?.role || "";
  const items = {
    login: true,
    exams: ["teacher", "admin"].includes(role),
    sessions: ["proctor", "teacher", "admin"].includes(role),
    live: ["proctor", "teacher", "admin"].includes(role),
    review: ["proctor", "teacher", "admin"].includes(role),
    admin: role === "admin",
  };
  nav.querySelectorAll(".nav-btn").forEach((btn) => {
    const view = btn.dataset.view;
    btn.style.display = items[view] ? "block" : "none";
  });
}

function defaultView() {
  if (state.user?.role === "student") {
    window.location.href = "/student";
    return "login";
  }
  if (["teacher", "admin"].includes(state.user?.role)) return "exams";
  return "sessions";
}

async function loginView() {
  app.innerHTML = `
    <div class="card" style="max-width:420px">
      <h2>Вход</h2>
      <label>Email<input id="email" value="admin@iup.local"></label>
      <label>Пароль<input id="password" type="password" value="admin123"></label>
      <button class="primary" id="login-btn">Войти</button>
      <p class="muted" style="margin-top:16px">
        Участник экзамена? <a href="/student">Кабинет студента →</a>
      </p>
    </div>`;
  document.getElementById("login-btn").onclick = async () => {
    const body = new URLSearchParams({
      username: document.getElementById("email").value,
      password: document.getElementById("password").value,
    });
    const response = await fetch(`${API}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
    });
    if (!response.ok) return alert("Ошибка входа");
    state.token = (await response.json()).access_token;
    localStorage.setItem("iup_token", state.token);
    state.user = await api("/users/me");
    userInfo.textContent = `${state.user.full_name} (${state.user.role})`;
    setupNav();
    showView(defaultView());
  };
}

async function examsView() {
  state.exams = await api("/exams");
  app.innerHTML = `
    <h2>Экзамены</h2>
    <div class="card">
      <h3>Создать экзамен</h3>
      <input id="exam-title" placeholder="Название экзамена">
      <input id="exam-desc" placeholder="Описание (необязательно)">
      <input id="exam-student-email" placeholder="Email студента (сразу назначить)">
      <input id="exam-student-name" placeholder="Имя студента (для нового)">
      <button class="primary" id="create-exam">Создать</button>
      <p class="muted">Если email не указан — студент сможет записаться сам в кабинете /student</p>
    </div>
    <div class="grid" style="margin-top:16px">
      ${state.exams.map((e) => `
        <div class="card">
          <h3>${e.title}</h3>
          <p class="muted">${e.description || "Без описания"}</p>
          <button class="primary" data-exam="${e.id}">Управление</button>
        </div>`).join("") || "<p class='muted'>Экзаменов пока нет</p>"}
    </div>`;
  document.getElementById("create-exam").onclick = async () => {
    const title = document.getElementById("exam-title").value.trim();
    if (!title) return alert("Введите название");
    await api("/exams", {
      method: "POST",
      body: JSON.stringify({
        title,
        description: document.getElementById("exam-desc").value,
        student_email: document.getElementById("exam-student-email").value.trim(),
        student_name: document.getElementById("exam-student-name").value.trim(),
      }),
    });
    examsView();
  };
  app.querySelectorAll("button[data-exam]").forEach((btn) => {
    btn.onclick = () => {
      state.selectedExam = state.exams.find((e) => e.id === btn.dataset.exam);
      showView("exam-detail");
    };
  });
}

async function examDetailView() {
  if (!state.selectedExam) return showView("exams");
  const sessions = await api(`/exams/${state.selectedExam.id}/sessions`);
  app.innerHTML = `
    <h2>${state.selectedExam.title}</h2>
    <div class="card">
      <h3>Назначить участнику</h3>
      <input id="student-email" placeholder="email студента, например student@mail.ru">
      <input id="student-name" placeholder="Имя (для нового студента)">
      <button class="primary" id="assign-btn">Создать сессию</button>
      <p class="muted">Новому студенту будет задан пароль <code>student123</code></p>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>Сессии (${sessions.length})</h3>
      <table>
        <thead><tr><th>Студент</th><th>Статус</th><th>Risk</th><th></th></tr></thead>
        <tbody>
          ${sessions.map((s) => `
            <tr>
              <td>${s.student_name || s.student_email}</td>
              <td><span class="badge ${s.status}">${s.status}</span></td>
              <td>${s.risk_score.toFixed(1)}</td>
              <td><button data-session="${s.id}" data-token="${s.access_token}">Мониторинг</button></td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>
    <button id="back-exams">← К списку экзаменов</button>`;

  document.getElementById("assign-btn").onclick = async () => {
    const email = document.getElementById("student-email").value.trim();
    if (!email) return alert("Введите email студента");
    const session = await api(`/exams/${state.selectedExam.id}/sessions/by-email`, {
      method: "POST",
      body: JSON.stringify({
        email,
        full_name: document.getElementById("student-name").value,
      }),
    });
    alert(`Сессия создана для ${session.student_email}.\nСтудент увидит экзамен в кабинете /student или в приложении IUP Student.`);
    examDetailView();
  };
  document.getElementById("back-exams").onclick = () => showView("exams");
  app.querySelectorAll("button[data-session]").forEach((btn) => {
    btn.onclick = () => {
      state.selectedSession = sessions.find((s) => s.id === btn.dataset.session);
      connectWs(state.selectedSession.id);
      showView("live");
    };
  });
}

async function sessionsView() {
  state.sessions = await api("/sessions");
  app.innerHTML = `
    <h2>Все сессии</h2>
    <div class="grid">
      ${state.sessions.map((s) => `
        <div class="card">
          <div><strong>${s.id.slice(0, 8)}</strong> <span class="badge ${s.status}">${s.status}</span></div>
          <div class="muted">Risk: ${s.risk_score.toFixed(1)}</div>
          <button class="primary" data-session="${s.id}">Открыть</button>
        </div>`).join("")}
    </div>`;
  app.querySelectorAll("button[data-session]").forEach((btn) => {
    btn.onclick = () => {
      state.selectedSession = state.sessions.find((s) => s.id === btn.dataset.session);
      connectWs(state.selectedSession.id);
      showView("live");
    };
  });
}

function connectWs(sessionId) {
  if (state.ws) state.ws.close();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/api/ws/sessions/${sessionId}`);
  state.ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "violation") {
      state.violations.unshift(data.data);
      if (document.getElementById("live-feed")) renderLiveFeed();
    }
  };
}

async function liveView() {
  if (!state.selectedSession) return showView("sessions");
  state.violations = await api(`/sessions/${state.selectedSession.id}/violations`);
  app.innerHTML = `
    <h2>Live: ${state.selectedSession.id.slice(0, 8)}</h2>
    <div class="card">
      <div>Статус: <span class="badge ${state.selectedSession.status}">${state.selectedSession.status}</span></div>
      <div>Risk score: <strong>${state.selectedSession.risk_score}</strong></div>
      <p class="muted">Участник проходит экзамен через <a href="/student">кабинет студента</a> или приложение IUP Student.</p>
    </div>
    <div class="card"><h3>Лента событий</h3><div id="live-feed"></div></div>`;
  renderLiveFeed();
}

function renderLiveFeed() {
  const feed = document.getElementById("live-feed");
  if (!feed) return;
  feed.innerHTML = state.violations.slice(0, 50).map((v) => `
    <div class="event ${v.is_resolved ? "resolved" : ""} ${v.is_reminder ? "reminder" : ""}">
      <strong>${v.type}</strong> — ${v.message}
      <div class="muted">${new Date(v.created_at).toLocaleString()}</div>
    </div>`).join("") || "<p class='muted'>Событий пока нет</p>";
}

async function reviewView() {
  if (!state.selectedSession) return showView("sessions");
  state.violations = await api(`/sessions/${state.selectedSession.id}/violations`);
  app.innerHTML = `
    <h2>Review</h2>
    <div class="card">
      <table>
        <thead><tr><th>Тип</th><th>Сообщение</th><th>Решение</th></tr></thead>
        <tbody>
          ${state.violations.map((v) => `
            <tr>
              <td>${v.type}</td>
              <td>${v.message}</td>
              <td>
                <button data-v="${v.id}" data-d="confirmed">Подтвердить</button>
                <button data-v="${v.id}" data-d="false_positive">Ложное</button>
                <button data-v="${v.id}" data-d="invalidate">Аннулировать</button>
              </td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>
    <button class="primary" id="export-btn">Экспорт CSV</button>`;
  app.querySelectorAll("button[data-v]").forEach((btn) => {
    btn.onclick = async () => {
      await api("/reviews", {
        method: "POST",
        body: JSON.stringify({
          session_id: state.selectedSession.id,
          violation_id: btn.dataset.v,
          decision: btn.dataset.d,
        }),
      });
      alert("Review сохранён");
    };
  });
  document.getElementById("export-btn").onclick = async () => {
    const data = await api(`/sessions/${state.selectedSession.id}/export`);
    const blob = new Blob([data.csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `session_${state.selectedSession.id}.csv`;
    a.click();
  };
}

async function adminView() {
  app.innerHTML = `
    <h2>Админ</h2>
    <div class="grid">
      <div class="card">
        <h3>Webhook</h3>
        <input id="webhook-url" placeholder="https://example.com/hook">
        <button class="primary" id="create-webhook">Добавить</button>
      </div>
    </div>`;
  document.getElementById("create-webhook").onclick = async () => {
    await api("/webhooks", {
      method: "POST",
      body: JSON.stringify({ url: document.getElementById("webhook-url").value, events: ["violation.created"] }),
    });
    alert("Webhook добавлен");
  };
}

async function showView(view) {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });
  if (!state.token && view !== "login") return loginView();
  if (view === "login") return loginView();
  if (!state.user) {
    try {
      state.user = await api("/users/me");
      userInfo.textContent = `${state.user.full_name} (${state.user.role})`;
      setupNav();
    } catch {
      state.token = "";
      localStorage.removeItem("iup_token");
      return loginView();
    }
  }
  if (view === "exams") return examsView();
  if (view === "exam-detail") return examDetailView();
  if (view === "sessions") return sessionsView();
  if (view === "live") return liveView();
  if (view === "review") return reviewView();
  if (view === "admin") return adminView();
}

async function init() {
  if (!state.token) return loginView();
  try {
    state.user = await api("/users/me");
    userInfo.textContent = `${state.user.full_name} (${state.user.role})`;
    setupNav();
    showView(defaultView());
  } catch {
    state.token = "";
    localStorage.removeItem("iup_token");
    loginView();
  }
}

init();
