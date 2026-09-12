const API = "/api";
const STATUS_LABELS = {
  pending: "Ожидает",
  precheck: "Проверка",
  active: "Идёт",
  completed: "Завершён",
  compromised: "Нарушения",
  cancelled: "Отменён",
};

const state = {
  token: localStorage.getItem("iup_student_token") || "",
  user: null,
  sessions: [],
  available: [],
  launchSessionId: "",
};

const app = document.getElementById("app");
const userInfo = document.getElementById("user-info");
const userBox = document.getElementById("user-box");
const toastEl = document.getElementById("toast");
const launchModal = document.getElementById("launch-modal");

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

function statusLabel(status) {
  return STATUS_LABELS[status] || status || "—";
}

function parseWhen(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function examPhase(examOrSession) {
  const settings = examOrSession?.settings || {};
  const now = Date.now();
  const opens = parseWhen(settings.opens_at || examOrSession?.exam_opens_at);
  const closes = parseWhen(settings.closes_at || examOrSession?.exam_closes_at);
  if (opens && now < opens.getTime()) return "before";
  if (closes && now > closes.getTime()) return "after";
  return "open";
}

function examWindowText(examOrSession) {
  const settings = examOrSession?.settings || {};
  const opens = parseWhen(settings.opens_at || examOrSession?.exam_opens_at);
  const closes = parseWhen(settings.closes_at || examOrSession?.exam_closes_at);
  if (!opens && !closes) return "";
  const fmt = (date) => date.toLocaleString();
  const phase = examPhase(examOrSession);
  if (phase === "before") return `Откроется ${fmt(opens)}`;
  if (phase === "after") return `Окно закрыто`;
  if (closes) return `До ${fmt(closes)}`;
  return `С ${fmt(opens)}`;
}

function toast(message, kind = "ok") {
  toastEl.hidden = false;
  toastEl.className = `toast ${kind}`;
  toastEl.textContent = message;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    toastEl.hidden = true;
  }, 3200);
}

function setAuthLayout(isAuth) {
  document.body.classList.toggle("auth-screen", isAuth);
  userBox.hidden = isAuth || !state.user;
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(`${API}${path}`, { ...options, headers });
  if (!response.ok) {
    const text = await response.text();
    try {
      const detail = JSON.parse(text).detail;
      throw new Error(typeof detail === "string" ? detail : text);
    } catch (error) {
      if (error instanceof Error && error.message && !error.message.startsWith("{") && error.message !== text) throw error;
      throw new Error(text);
    }
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

function loginView() {
  setAuthLayout(true);
  app.innerHTML = `
    <div class="auth-wrap">
      <div class="card auth-card">
        <div class="brand">
          <div class="brand-mark">IUP</div>
          <div class="brand-copy">
            <strong>Вход участника</strong>
            <span class="muted">Кабинет экзамена</span>
          </div>
        </div>
        <label>Email<input id="email" value="student@iup.local" autocomplete="username"></label>
        <label>Пароль<input id="password" type="password" value="student123" autocomplete="current-password"></label>
        <p class="error" id="login-error"></p>
        <button class="primary" id="login-btn">Войти</button>
        <p class="muted" style="margin-top:16px">
          На компьютере удобнее приложение <strong>IUP Student</strong> — тот же email и пароль.
        </p>
      </div>
    </div>`;
  const login = async () => {
    const button = document.getElementById("login-btn");
    const error = document.getElementById("login-error");
    error.textContent = "";
    button.disabled = true;
    try {
      const body = new URLSearchParams({
        username: document.getElementById("email").value,
        password: document.getElementById("password").value,
      });
      const response = await fetch(`${API}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body,
      });
      if (!response.ok) {
        error.textContent = "Неверный email или пароль.";
        return;
      }
      state.token = (await response.json()).access_token;
      localStorage.setItem("iup_student_token", state.token);
      state.user = await api("/users/me");
      userInfo.textContent = state.user.full_name;
      await examsView();
    } catch {
      error.textContent = "Не удалось подключиться к серверу.";
    } finally {
      button.disabled = false;
    }
  };
  document.getElementById("login-btn").onclick = login;
  app.querySelectorAll("input").forEach((input) => {
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") login();
    });
  });
}

function logout() {
  state.token = "";
  state.user = null;
  localStorage.removeItem("iup_student_token");
  loginView();
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
  toast("Вы записаны на экзамен");
  await examsView();
}

async function examsView() {
  setAuthLayout(false);
  userInfo.textContent = state.user?.full_name || "";
  await loadExams();
  const hasSessions = state.sessions.length > 0;
  const hasAvailable = state.available.length > 0;
  const isWindows = navigator.userAgent.includes("Windows");

  app.innerHTML = `
    <div class="page-head">
      <div>
        <h2>Мои экзамены</h2>
        <p class="muted">Выберите экзамен и запустите прокторинг на этом компьютере.</p>
      </div>
      <button class="ghost" id="refresh-btn" style="width:auto">Обновить</button>
    </div>
    <div class="grid">
      ${hasSessions
        ? state.sessions.map((session) => `
        <article class="card exam-card">
          <h3>${esc(session.exam_title || "Экзамен")}</h3>
          <div><span class="badge ${esc(session.status)}">${esc(statusLabel(session.status))}</span></div>
          <p class="meta">Риск: ${(session.risk_score ?? 0).toFixed(1)}${examWindowText(session) ? ` · ${esc(examWindowText(session))}` : ""}</p>
          <div class="actions">
            <button class="primary" data-id="${esc(session.id)}">${session.status === "completed" ? "Открыть снова" : "Начать прокторинг"}</button>
            <button class="ghost" data-dl="${esc(session.id)}">${isWindows ? "Скачать .bat" : "Скачать скрипт"}</button>
          </div>
        </article>`).join("")
        : `<div class="card empty"><p class="muted">Назначенных экзаменов пока нет.</p></div>`}
    </div>
    ${hasAvailable ? `
    <h3 style="margin-top:28px">Доступны для записи</h3>
    <div class="grid">
      ${state.available.map((exam) => `
        <article class="card exam-card">
          <h3>${esc(exam.title)}</h3>
          <p class="meta">${esc(exam.description || "Без описания")}${examWindowText(exam) ? ` · ${esc(examWindowText(exam))}` : ""}</p>
          <div class="actions">
            <button class="primary" data-join="${esc(exam.id)}" ${examPhase(exam) === "before" ? "disabled" : ""}>${examPhase(exam) === "before" ? "Ещё не начался" : "Записаться"}</button>
          </div>
        </article>`).join("")}
    </div>` : ""}
    ${!hasSessions && !hasAvailable ? `
    <div class="card empty" style="margin-top:16px">
      <p>Открытых экзаменов нет. Попросите преподавателя назначить вам экзамен или включить свободную запись.</p>
    </div>` : ""}
    <div class="card" style="margin-top:16px">
      <h3>Как проходить экзамен</h3>
      <p>Запустите <strong>IUP Student</strong>, войдите тем же email и нажмите «Начать экзамен». После этого откроется окно камеры.</p>
      <p class="muted">Windows: <code>scripts\\run_student.bat</code> · Linux: <code>./scripts/run_student.sh</code></p>
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
      btn.disabled = true;
      try {
        await joinExam(btn.dataset.join);
      } catch (error) {
        toast("Не удалось записаться: " + error.message, "err");
      } finally {
        btn.disabled = false;
      }
    };
  });
}

async function startExam(sessionId) {
  const info = await api(`/my/sessions/${sessionId}/launch-info`);
  const isWindows = navigator.userAgent.includes("Windows");
  state.launchSessionId = sessionId;
  document.getElementById("launch-title").textContent = info.exam_title || "Запуск прокторинга";
  document.getElementById("launch-text").innerHTML = isWindows
    ? "Экзамен запускается приложением <strong>IUP Student</strong> (<code>scripts\\run_student.bat</code>). Войдите тем же email и нажмите «Начать экзамен». Скрипт ниже ищет проект IUP на этом компьютере."
    : "Откройте приложение IUP Student (<code>./scripts/run_student.sh</code>), войдите и нажмите «Начать экзамен».";
  launchModal.hidden = false;
}

async function downloadLauncher(sessionId) {
  const isWindows = navigator.userAgent.includes("Windows");
  const ext = isWindows ? "bat" : "sh";
  const response = await fetch(`${API}/my/sessions/${sessionId}/launcher.${ext}`, {
    headers: { Authorization: `Bearer ${state.token}` },
  });
  if (!response.ok) return toast("Не удалось скачать скрипт", "err");
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `iup-exam-${sessionId.slice(0, 8)}.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
  toast("Скрипт запуска скачан");
}

document.getElementById("logout-btn").onclick = logout;
document.getElementById("launch-close").onclick = () => {
  launchModal.hidden = true;
};
document.getElementById("launch-download").onclick = () => {
  if (state.launchSessionId) downloadLauncher(state.launchSessionId);
};
launchModal.addEventListener("click", (event) => {
  if (event.target === launchModal) launchModal.hidden = true;
});

async function init() {
  if (!state.token) return loginView();
  try {
    state.user = await api("/users/me");
    if (state.user.role !== "student") {
      toast("Эта страница для участников. Откройте панель преподавателя.", "err");
      state.token = "";
      localStorage.removeItem("iup_student_token");
      return loginView();
    }
    userInfo.textContent = state.user.full_name;
    await examsView();
  } catch {
    state.token = "";
    localStorage.removeItem("iup_student_token");
    loginView();
  }
}

init();
