const API = "/api";
const state = {
  token: localStorage.getItem("iup_student_token") || "",
  user: null,
  sessions: [],
  available: [],
};
const app = document.getElementById("app");
const userInfo = document.getElementById("user-info");

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(`${API}${path}`, { ...options, headers });
  if (!response.ok) throw new Error(await response.text());
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

function loginView() {
  app.innerHTML = `
    <div class="card" style="max-width:440px">
      <h2>Вход участника</h2>
      <label>Email<input id="email" value="student@iup.local"></label>
      <label>Пароль<input id="password" type="password" value="student123"></label>
      <button class="primary" id="login-btn">Войти</button>
      <p class="muted">Или откройте приложение <strong>IUP Student</strong> на рабочем столе — без браузера.</p>
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
    if (!response.ok) return alert("Неверный email или пароль");
    state.token = (await response.json()).access_token;
    localStorage.setItem("iup_student_token", state.token);
    state.user = await api("/users/me");
    userInfo.textContent = state.user.full_name;
    examsView();
  };
}

async function loadExams() {
  const [sessions, available] = await Promise.all([
    api("/my/sessions"),
    api("/my/available-exams"),
  ]);
  state.sessions = sessions;
  state.available = available;
}

async function joinExam(examId) {
  await api(`/exams/${examId}/join`, { method: "POST" });
  await examsView();
}

async function examsView() {
  await loadExams();
  const hasSessions = state.sessions.length > 0;
  const hasAvailable = state.available.length > 0;

  app.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
      <h2 style="margin:0">Мои экзамены</h2>
      <button id="refresh-btn">Обновить</button>
    </div>
    <div class="grid">
      ${hasSessions
        ? state.sessions.map((s) => `
        <div class="card">
          <h3>${s.exam_title || "Экзамен"}</h3>
          <div><span class="badge ${s.status}">${s.status}</span></div>
          <p class="muted">Risk: ${s.risk_score.toFixed(1)}</p>
          <button class="primary" data-id="${s.id}">Начать прокторинг</button>
          <button data-dl="${s.id}" style="margin-top:8px">${navigator.userAgent.includes("Windows") ? "Скачать запуск (.bat)" : "Скачать скрипт запуска"}</button>
        </div>`).join("")
        : `<div class="card"><p class="muted">Назначенных экзаменов пока нет.</p></div>`}
    </div>
    ${hasAvailable ? `
    <h3 style="margin-top:24px">Доступны для записи</h3>
    <div class="grid">
      ${state.available.map((e) => `
        <div class="card">
          <h3>${e.title}</h3>
          <p class="muted">${e.description || "Без описания"}</p>
          <button class="primary" data-join="${e.id}">Записаться</button>
        </div>`).join("")}
    </div>` : ""}
    ${!hasSessions && !hasAvailable ? `
    <div class="card" style="margin-top:16px">
      <p>Открытых экзаменов нет. Попросите преподавателя назначить вам экзамен или включить свободную запись.</p>
    </div>` : ""}
    <div class="card" style="margin-top:16px">
      <h3>Рекомендуемый способ</h3>
      <p>Запустите приложение <strong>IUP Student</strong> — войдите тем же email и паролем, выберите экзамен и нажмите «Начать».</p>
      <p class="muted">Windows: один файл <code>dist\IUP Student.exe</code> (сборка: <code>scripts\build_student_windows.ps1</code>)</p>
      <p class="muted">Linux: <code>./scripts/run_student.sh</code></p>
    </div>`;

  document.getElementById("refresh-btn").onclick = () => examsView();
  app.querySelectorAll("button[data-id]").forEach((btn) => {
    btn.onclick = () => startExam(btn.dataset.id);
  });
  app.querySelectorAll("button[data-dl]").forEach((btn) => {
    btn.onclick = () => downloadLauncher(btn.dataset.dl);
  });
  app.querySelectorAll("button[data-join]").forEach((btn) => {
    btn.onclick = async () => {
      try {
        await joinExam(btn.dataset.join);
      } catch (error) {
        alert("Не удалось записаться: " + error.message);
      }
    };
  });
}

async function startExam(sessionId) {
  const info = await api(`/my/sessions/${sessionId}/launch-info`);
  const msg = [
    `Экзамен: ${info.exam_title}`,
    "",
    "Для прохождения без консоли откройте приложение IUP Student",
    "(ярлык на рабочем столе или ./scripts/run_student.sh).",
    "",
    "В приложении войдите и нажмите «Начать экзамен».",
  ].join("\n");
  alert(msg);
  await downloadLauncher(sessionId);
}

async function downloadLauncher(sessionId) {
  const isWindows = navigator.userAgent.includes("Windows");
  const ext = isWindows ? "bat" : "sh";
  const response = await fetch(`${API}/my/sessions/${sessionId}/launcher.${ext}`, {
    headers: { Authorization: `Bearer ${state.token}` },
  });
  if (!response.ok) return alert("Не удалось скачать скрипт");
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `iup-exam-${sessionId.slice(0, 8)}.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
}

async function init() {
  if (!state.token) return loginView();
  try {
    state.user = await api("/users/me");
    if (state.user.role !== "student") {
      alert("Эта страница для участников. Используйте главную панель.");
      state.token = "";
      localStorage.removeItem("iup_student_token");
      return loginView();
    }
    userInfo.textContent = state.user.full_name;
    examsView();
  } catch {
    state.token = "";
    localStorage.removeItem("iup_student_token");
    loginView();
  }
}

init();
