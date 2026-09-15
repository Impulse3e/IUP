const API = "/api";
const STATUS_LABELS = {
  pending: "Ожидает",
  precheck: "Проверка",
  active: "Идёт",
  completed: "Завершён",
  compromised: "Нарушения",
  cancelled: "Отменён",
};

const EVENT_LABELS = {
  no_face: "Нет лица",
  multiple_faces: "Несколько лиц",
  look_away: "Поворот головы",
  audio_quiet: "Тихий звук",
  audio_loud: "Громкий звук",
  identity_mismatch: "Другое лицо",
  forbidden_process: "Запрещённый процесс",
  window_focus_lost: "Потеря фокуса",
  agent_tamper: "Вмешательство",
  heartbeat_lost: "Агент не отвечает",
  second_monitor: "Второй монитор",
  watched_title: "Подозрительная вкладка",
};

const state = {
  token: localStorage.getItem("iup_token") || "",
  user: null,
  sessions: [],
  sessionsAt: 0,
  wsSessionId: "",
  exams: [],
  students: [],
  selectedSession: null,
  selectedExam: null,
  violations: [],
  evidence: [],
  ws: null,
  liveTimer: null,
  liveDetail: false,
};

const app = document.getElementById("app");
const userInfo = document.getElementById("user-info");
const userBox = document.getElementById("user-box");
const nav = document.getElementById("main-nav");
const toastEl = document.getElementById("toast");

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

function eventLabel(type) {
  return EVENT_LABELS[type] || type || "—";
}

function parseWhen(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function examPhase(exam) {
  const settings = exam?.settings || {};
  if (settings.closed) return "after";
  const now = Date.now();
  const opens = parseWhen(settings.opens_at);
  const closes = parseWhen(settings.closes_at);
  if (opens && now < opens.getTime()) return "before";
  if (closes && now > closes.getTime()) return "after";
  return "open";
}

function examWindowText(exam) {
  const settings = exam?.settings || {};
  if (settings.closed) return "Закрыт вручную";
  const opens = parseWhen(settings.opens_at);
  const closes = parseWhen(settings.closes_at);
  if (!opens && !closes) return "Окно не ограничено";
  const fmt = (date) => date.toLocaleString();
  const phase = examPhase(exam);
  if (phase === "before") return `Откроется ${fmt(opens)}`;
  if (phase === "after") return `Закрыт ${closes ? fmt(closes) : ""}`.trim();
  if (closes) return `До ${fmt(closes)}`;
  return `С ${fmt(opens)}`;
}

function toIsoLocal(value) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toISOString();
}

function toDatetimeLocalValue(iso) {
  const date = parseWhen(iso);
  if (!date) return "";
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function outcomeLabel(value) {
  if (value === "passed") return "зачёт";
  if (value === "invalidate") return "аннулирована";
  return value || "не вынесено";
}

function outcomeBadgeClass(value) {
  if (value === "passed") return "passed";
  if (value === "invalidate") return "invalidate";
  return "pending";
}

function parseWatchTitles(text) {
  return String(text || "")
    .split(/[\n,;]+/)
    .map((item) => item.trim())
    .filter((item) => item.length >= 2)
    .slice(0, 40);
}

function formatWatchTitles(list) {
  return (Array.isArray(list) ? list : []).join("\n");
}

function evidenceItems(violationId) {
  return (state.evidence || []).filter((item) => item.violation_id === violationId);
}

function evidenceButtonsHtml(violationId) {
  const items = evidenceItems(violationId);
  if (!items.length) return "—";
  const label = (type) => {
    if (type === "screen_still") return "Стол";
    if (type === "webcam_still") return "Камера";
    if (type === "video_clip") return "Клип";
    return "Файл";
  };
  return items.map((item) => (
    `<button class="ghost" data-clip="${esc(item.url)}" data-kind="${item.type === "video_clip" ? "video" : "image"}" style="width:auto;margin:0">${label(item.type)}</button>`
  )).join(" ");
}

function closeEvidencePreview() {
  const modal = document.getElementById("evidence-modal");
  const img = document.getElementById("evidence-image");
  if (!modal || !img) return;
  modal.hidden = true;
  if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
  img.dataset.url = "";
  img.removeAttribute("src");
}

function bindEvidenceButtons(root = document) {
  root.querySelectorAll("button[data-clip]").forEach((btn) => {
    if (btn.dataset.bound) return;
    btn.dataset.bound = "1";
    btn.onclick = async () => {
      try {
        const url = await authedBlob(btn.dataset.clip);
        const modal = document.getElementById("evidence-modal");
        const img = document.getElementById("evidence-image");
        if (!modal || !img) {
          window.open(url, "_blank");
          return;
        }
        if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
        img.dataset.url = url;
        img.src = url;
        modal.hidden = false;
      } catch {
        toast("Не удалось открыть снимок", "err");
      }
    };
  });
}

function isOnline(session) {
  return Boolean(session?.online);
}

function criticalText(session) {
  const n = Number(session?.critical_open || 0);
  if (n <= 0) return "критичных нет";
  return `критично: ${n}`;
}

function stopLiveTimer() {
  if (state.liveTimer) {
    clearInterval(state.liveTimer);
    state.liveTimer = null;
  }
}

function toast(message, kind = "ok", ms = 3200) {
  toastEl.hidden = false;
  toastEl.className = `toast ${kind}`;
  toastEl.textContent = message;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    toastEl.hidden = true;
  }, ms);
}

function setAuthLayout(isAuth) {
  document.body.classList.toggle("auth-screen", isAuth);
  if (userBox) userBox.hidden = isAuth || !state.user;
}

document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.dataset.view === "live") {
      state.liveDetail = false;
      state.selectedSession = null;
    }
    showView(btn.dataset.view);
  });
});

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

function setupNav() {
  const role = state.user?.role || "";
  const items = {
    login: !state.user,
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
  setAuthLayout(!state.user);
  if (state.user && userInfo) {
    userInfo.innerHTML = `<div class="name">${esc(state.user.full_name)}</div><div class="role">${esc(state.user.role)}</div>`;
  }
}

function logout() {
  state.token = "";
  state.user = null;
  localStorage.removeItem("iup_token");
  stopLiveTimer();
  if (state.ws) state.ws.close();
  setupNav();
  loginView();
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
  setAuthLayout(true);
  app.innerHTML = `
    <div class="auth-wrap">
      <div class="card auth-card">
        <div class="brand">
          <div class="brand-mark">IUP</div>
          <div class="brand-copy">
            <strong>Вход в панель</strong>
            <span class="muted">Преподаватель, проктор, админ</span>
          </div>
        </div>
        <label>Email<input id="email" value="admin@iup.local" autocomplete="username"></label>
        <label>Пароль<input id="password" type="password" value="admin123" autocomplete="current-password"></label>
        <p class="error" id="login-error"></p>
        <button class="primary" id="login-btn">Войти</button>
        <p class="muted" style="margin-top:16px">
          Участник экзамена? <a href="/student">Кабинет студента →</a>
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
      localStorage.setItem("iup_token", state.token);
      state.user = await api("/users/me");
      setupNav();
      showView(defaultView());
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

async function passwordView() {
  stopLiveTimer();
  app.innerHTML = `
    <div class="page-head">
      <div>
        <h2>Смена пароля</h2>
        <p class="muted">${esc(state.user?.email || "")}</p>
      </div>
    </div>
    <div class="card form-card">
      <label>Текущий пароль<input id="pw-current" type="password" autocomplete="current-password"></label>
      <label>Новый пароль<input id="pw-new" type="password" autocomplete="new-password"></label>
      <label>Ещё раз<input id="pw-again" type="password" autocomplete="new-password"></label>
      <p class="muted">Не короче 8 символов.</p>
      <button class="primary" id="save-password">Сохранить</button>
    </div>`;
  document.getElementById("save-password").onclick = async () => {
    const current = document.getElementById("pw-current").value;
    const next = document.getElementById("pw-new").value;
    const again = document.getElementById("pw-again").value;
    if (next !== again) return toast("Пароли не совпадают", "err");
    try {
      await api("/users/me/password", {
        method: "POST",
        body: JSON.stringify({ current_password: current, new_password: next }),
      });
      toast("Пароль изменён");
      showView(defaultView());
    } catch (error) {
      toast(error.message || "Не удалось сменить пароль", "err");
    }
  };
}

async function examsView() {
  state.exams = await api("/exams");
  app.innerHTML = `
    <div class="page-head">
      <div>
        <h2>Экзамены</h2>
        <p class="muted">Создайте экзамен и назначьте участников.</p>
      </div>
    </div>
    <div class="card form-card">
      <h3>Новый экзамен</h3>
      <label>Название<input id="exam-title" placeholder="Например, Алгебра, сессия 1"></label>
      <label>Описание<input id="exam-desc" placeholder="Необязательно"></label>
      <label>Email студента<input id="exam-student-email" placeholder="сразу назначить, необязательно"></label>
      <label>Имя студента<input id="exam-student-name" placeholder="для нового аккаунта"></label>
      <label>Начало окна<input id="exam-opens" type="datetime-local"></label>
      <label>Конец окна<input id="exam-closes" type="datetime-local"></label>
      <label>Слова в заголовках вкладок<textarea id="exam-watch" rows="3" placeholder="chatgpt&#10;переводчик&#10;wikipedia"></textarea></label>
      <button class="primary" id="create-exam">Создать</button>
      <p class="muted">Пустые даты — экзамен открыт всегда. Слова — предупреждение, если они есть в названии активного окна. К событию приложатся кадр стола и камеры. Студента не блокируем.</p>
    </div>
    <div class="grid" style="margin-top:16px">
      ${state.exams.map((exam) => `
        <article class="card exam-card">
          <h3>${esc(exam.title)}</h3>
          <p class="meta">${esc(exam.description || "Без описания")}</p>
          <p class="meta">${exam.settings?.closed ? `<span class="badge closed">закрыт</span> ` : ""}<span class="badge ${examPhase(exam) === "open" ? "active" : "pending"}">${esc(examWindowText(exam))}</span></p>
          <div class="actions">
            <button class="primary" data-exam="${esc(exam.id)}">Управление</button>
          </div>
        </article>`).join("") || "<div class='card empty'><p class='muted'>Экзаменов пока нет</p></div>"}
    </div>`;
  document.getElementById("create-exam").onclick = async () => {
    const title = document.getElementById("exam-title").value.trim();
    if (!title) return toast("Введите название", "err");
    const created = await api("/exams", {
      method: "POST",
      body: JSON.stringify({
        title,
        description: document.getElementById("exam-desc").value,
        student_email: document.getElementById("exam-student-email").value.trim(),
        student_name: document.getElementById("exam-student-name").value.trim(),
        settings: {
          open_enrollment: true,
          opens_at: toIsoLocal(document.getElementById("exam-opens").value),
          closes_at: toIsoLocal(document.getElementById("exam-closes").value),
          watch_titles: parseWatchTitles(document.getElementById("exam-watch").value),
        },
      }),
    });
    if (created.initial_password) {
      toast(`Студент ${created.created_student_email}: пароль ${created.initial_password}`, "ok", 12000);
    } else {
      toast("Экзамен создан");
    }
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
  try {
    state.selectedExam = await api(`/exams/${state.selectedExam.id}`);
  } catch {
    /* keep the list copy if refresh fails */
  }
  const sessions = await api(`/exams/${state.selectedExam.id}/sessions`);
  const settings = state.selectedExam.settings || {};
  const closed = Boolean(settings.closed);
  const passed = sessions.filter((s) => s.outcome === "passed").length;
  const invalidated = sessions.filter((s) => s.outcome === "invalidate").length;
  const undecided = sessions.length - passed - invalidated;
  const inProgress = sessions.filter((s) => ["pending", "precheck", "active"].includes(s.status)).length;
  const ranked = [...sessions].sort((left, right) => {
    const rank = (item) => {
      if (["active", "precheck"].includes(item.status)) return 0;
      if (!item.outcome && item.status === "pending") return 1;
      if (item.outcome === "invalidate") return 2;
      if (!item.outcome) return 3;
      return 4;
    };
    const byRank = rank(left) - rank(right);
    if (byRank) return byRank;
    return String(left.student_name || left.student_email || "").localeCompare(
      String(right.student_name || right.student_email || ""),
      "ru",
    );
  });
  app.innerHTML = `
    <div class="page-head">
      <div>
        <h2>${esc(state.selectedExam.title)}</h2>
        <p class="muted">${esc(state.selectedExam.description || "Назначьте участников и откройте мониторинг.")} · ${esc(examWindowText(state.selectedExam))}</p>
      </div>
      <button class="ghost" id="back-exams" style="width:auto">← К списку</button>
    </div>
    <div class="card form-card">
      <h3>Окно экзамена</h3>
      <label>Начало окна<input id="exam-opens" type="datetime-local" value="${esc(toDatetimeLocalValue(settings.opens_at))}"></label>
      <label>Конец окна<input id="exam-closes" type="datetime-local" value="${esc(toDatetimeLocalValue(settings.closes_at))}"></label>
      <div class="btn-row" style="margin-top:12px">
        <button class="primary" id="save-window" style="width:auto;margin:0">Сохранить окно</button>
        ${closed
          ? `<button class="ghost" id="reopen-exam" style="width:auto;margin:0">Открыть снова</button>`
          : `<button class="ghost" id="close-exam" style="width:auto;margin:0">Закрыть экзамен</button>`}
      </div>
      <p class="muted">${closed
        ? "Экзамен закрыт вручную: студенты не входят, события не пишутся. Даты окна сохраняются и снова действуют после «Открыть снова»."
        : "Пустые даты — окно не ограничено. «Закрыть экзамен» сразу останавливает вход и запись, не дожидаясь конца окна."}</p>
    </div>
    <div class="card form-card">
      <h3>Контроль вкладок</h3>
      <label>Слова в заголовке активного окна<textarea id="exam-watch" rows="4" placeholder="chatgpt&#10;claude&#10;переводчик&#10;wikipedia">${esc(formatWatchTitles(settings.watch_titles))}</textarea></label>
      <button class="primary" id="save-watch" style="width:auto;margin-top:12px">Сохранить список</button>
      <p class="muted">Одно слово или фраза на строку. Если оно есть в названии окна (вкладки Chrome/Edge), преподавателю уйдёт предупреждение со снимком стола и камеры. Не пишите «chrome» — сработает на любую вкладку.</p>
    </div>
    <div class="card form-card">
      <h3>Назначить участнику</h3>
      <label>Email<input id="student-email" placeholder="student@mail.ru"></label>
      <label>Имя<input id="student-name" placeholder="для нового студента"></label>
      <button class="primary" id="assign-btn">Создать сессию</button>
      <p class="muted">Если студента ещё нет, будет создан аккаунт с одноразовым паролем.</p>
    </div>
    <div class="card" style="margin-top:16px">
      <div class="page-head" style="margin-bottom:8px">
        <div>
          <h3>Исходы группы</h3>
          <p class="muted">${sessions.length} участников · зачёт ${passed} · аннулировано ${invalidated} · без решения ${undecided}${inProgress ? ` · ещё идут ${inProgress}` : ""}</p>
        </div>
        <div class="btn-row">
          <button class="ghost" id="export-group-csv" style="width:auto;margin:0">CSV</button>
          <button class="ghost" id="export-group-html" style="width:auto;margin:0">HTML</button>
        </div>
      </div>
      <div class="table-scroll">
        <table>
          <thead><tr><th>Студент</th><th>Статус</th><th>Риск</th><th>Исход</th><th>Комментарий</th><th></th></tr></thead>
          <tbody>
            ${ranked.map((s) => `
              <tr>
                <td>${esc(s.student_name || s.student_email)}<div class="muted">${esc(s.student_email || "")}</div></td>
                <td><span class="badge ${esc(s.status)}">${esc(statusLabel(s.status))}</span></td>
                <td>${Number(s.risk_score || 0).toFixed(1)}</td>
                <td><span class="badge ${outcomeBadgeClass(s.outcome)}">${esc(outcomeLabel(s.outcome))}</span></td>
                <td title="${esc(s.outcome_comment || "")}">${esc(s.outcome_comment || "—")}</td>
                <td class="btn-row">
                  <button class="ghost" data-session="${esc(s.id)}" style="width:auto;margin:0">Live</button>
                  <button class="ghost" data-review="${esc(s.id)}" style="width:auto;margin:0">Проверка</button>
                </td>
              </tr>`).join("") || "<tr><td colspan='6' class='muted'>Сессий пока нет</td></tr>"}
          </tbody>
        </table>
      </div>
    </div>`;

  document.getElementById("assign-btn").onclick = async () => {
    const email = document.getElementById("student-email").value.trim();
    if (!email) return toast("Введите email студента", "err");
    const session = await api(`/exams/${state.selectedExam.id}/sessions/by-email`, {
      method: "POST",
      body: JSON.stringify({
        email,
        full_name: document.getElementById("student-name").value,
      }),
    });
    if (session.initial_password) {
      toast(`Сессия для ${session.student_email}. Пароль: ${session.initial_password}`, "ok", 12000);
    } else {
      toast(`Сессия создана для ${session.student_email}`);
    }
    examDetailView();
  };
  document.getElementById("save-window").onclick = async () => {
    try {
      state.selectedExam = await api(`/exams/${state.selectedExam.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          opens_at: toIsoLocal(document.getElementById("exam-opens").value),
          closes_at: toIsoLocal(document.getElementById("exam-closes").value),
        }),
      });
      toast("Окно экзамена сохранено");
      examDetailView();
    } catch (error) {
      toast(error.message || "Не удалось сохранить окно", "err");
    }
  };
  document.getElementById("save-watch").onclick = async () => {
    try {
      state.selectedExam = await api(`/exams/${state.selectedExam.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          watch_titles: parseWatchTitles(document.getElementById("exam-watch").value),
        }),
      });
      toast("Список вкладок сохранён");
      examDetailView();
    } catch (error) {
      toast(error.message || "Не удалось сохранить список", "err");
    }
  };
  document.getElementById("close-exam")?.addEventListener("click", async () => {
    if (!window.confirm("Закрыть экзамен сейчас? Студенты не смогут войти, события перестанут писаться.")) return;
    try {
      state.selectedExam = await api(`/exams/${state.selectedExam.id}`, {
        method: "PATCH",
        body: JSON.stringify({ closed: true }),
      });
      toast("Экзамен закрыт");
      examDetailView();
    } catch (error) {
      toast(error.message || "Не удалось закрыть экзамен", "err");
    }
  });
  document.getElementById("reopen-exam")?.addEventListener("click", async () => {
    try {
      state.selectedExam = await api(`/exams/${state.selectedExam.id}`, {
        method: "PATCH",
        body: JSON.stringify({ closed: false }),
      });
      toast("Экзамен снова открыт");
      examDetailView();
    } catch (error) {
      toast(error.message || "Не удалось открыть экзамен", "err");
    }
  });
  async function downloadGroup(format, filename, mime) {
    const data = await api(`/exams/${state.selectedExam.id}/outcomes?format=${format}`);
    const body = format === "csv" ? data.csv : data;
    const blob = new Blob([body], { type: mime });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
  }
  const slug = (state.selectedExam.title || "exam").replace(/[^\w\-]+/g, "_").slice(0, 40);
  document.getElementById("export-group-csv").onclick = () => downloadGroup("csv", `${slug}_outcomes.csv`, "text/csv");
  document.getElementById("export-group-html").onclick = () => downloadGroup("html", `${slug}_outcomes.html`, "text/html");
  document.getElementById("back-exams").onclick = () => showView("exams");
  app.querySelectorAll("button[data-session]").forEach((btn) => {
    btn.onclick = () => openSession(sessions.find((s) => s.id === btn.dataset.session));
  });
  app.querySelectorAll("button[data-review]").forEach((btn) => {
    btn.onclick = () => {
      state.selectedSession = sessions.find((s) => s.id === btn.dataset.review);
      showView("review");
    };
  });
}

function sessionRank(session) {
  const heartbeat = session.last_heartbeat ? Date.now() - new Date(session.last_heartbeat).getTime() : 1e12;
  const fresh = heartbeat < 45000;
  if (session.status === "active" && fresh) return 0;
  if (session.status === "precheck" && fresh) return 1;
  if (session.status === "pending") return 2;
  if (session.status === "active") return 3;
  if (session.status === "precheck") return 4;
  if (session.status === "compromised") return 5;
  if (session.status === "completed") return 6;
  return 9;
}

function sessionStamp(session) {
  return new Date(session.last_heartbeat || session.started_at || session.consent_at || 0).getTime();
}

async function pickLiveSession() {
  const now = Date.now();
  if (!state.sessions.length || now - (state.sessionsAt || 0) > 5000) {
    state.sessions = await api("/sessions");
    state.sessionsAt = now;
  }
  if (state.selectedSession) {
    const fresh = state.sessions.find((item) => item.id === state.selectedSession.id);
    if (fresh) {
      state.selectedSession = fresh;
      return fresh;
    }
  }
  const ranked = [...state.sessions].sort((left, right) => {
    const byStatus = sessionRank(left) - sessionRank(right);
    if (byStatus) return byStatus;
    return sessionStamp(right) - sessionStamp(left);
  });
  state.selectedSession = ranked[0] || null;
  return state.selectedSession;
}

function openSession(session) {
  if (!session) return toast("Сессия не найдена", "err");
  state.selectedSession = session;
  state.liveDetail = true;
  showView("live");
}

async function sessionsView() {
  state.sessions = await api("/sessions");
  app.innerHTML = `
    <div class="page-head">
      <div>
        <h2>Все сессии</h2>
        <p class="muted">Нажмите Live, чтобы открыть камеру участника.</p>
      </div>
    </div>
    <div class="grid">
      ${state.sessions.map((s) => `
        <article class="card exam-card">
          <h3>${esc(s.student_name || s.student_email || s.id.slice(0, 8))}</h3>
          <div class="status-row">
            <span class="dot ${isOnline(s) ? "on" : "off"}"></span>
            <span class="badge ${esc(s.status)}">${esc(statusLabel(s.status))}</span>
            ${Number(s.critical_open) ? `<span class="badge compromised">критично ${esc(s.critical_open)}</span>` : ""}
          </div>
          <p class="meta">${esc(s.exam_title || "Экзамен")} · риск ${Number(s.risk_score || 0).toFixed(1)} · ${esc(criticalText(s))}</p>
          <div class="actions">
            <button class="primary" data-session="${esc(s.id)}">Live</button>
            <button class="ghost" data-review="${esc(s.id)}">Проверка</button>
          </div>
        </article>`).join("") || "<div class='card empty'><p class='muted'>Сессий пока нет</p></div>"}
    </div>`;
  app.querySelectorAll("button[data-session]").forEach((btn) => {
    btn.onclick = () => openSession(state.sessions.find((s) => s.id === btn.dataset.session));
  });
  app.querySelectorAll("button[data-review]").forEach((btn) => {
    btn.onclick = () => {
      state.selectedSession = state.sessions.find((s) => s.id === btn.dataset.review);
      showView("review");
    };
  });
}

function connectWs(sessionId) {
  if (state.ws && state.wsSessionId === sessionId && state.ws.readyState < 2) {
    return;
  }
  if (state.ws) state.ws.close();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.wsSessionId = sessionId;
  state.ws = new WebSocket(`${proto}://${location.host}/api/ws/sessions/${sessionId}`);
  state.ws.onopen = () => {
    if (state.token) {
      state.ws.send(JSON.stringify({ type: "auth", token: state.token }));
    }
  };
  state.ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "violation") {
      state.violations.unshift(data.data);
      if (document.getElementById("live-feed")) renderLiveFeed();
    }
    if (data.type === "evidence") {
      loadEvidence();
    }
    if (data.type === "heartbeat") {
      const el = document.getElementById("live-heartbeat");
      if (el) el.textContent = `Агент на связи · ${new Date().toLocaleTimeString()}`;
    }
    if (data.type === "identity_verified") {
      const el = document.getElementById("live-identity");
      if (!el) return;
      if (data.match === false) el.textContent = "Лицо не совпало с эталоном";
      else if (data.match === true) el.textContent = "Лицо совпало с эталоном";
      else el.textContent = "Эталон лица сохранён";
    }
  };
}

async function loadEvidence() {
  if (!state.selectedSession) return;
  try {
    state.evidence = await api(`/sessions/${state.selectedSession.id}/evidence`);
  } catch {
    state.evidence = [];
  }
  renderLiveClips();
}

function evidenceFor(violationId) {
  return evidenceItems(violationId)[0];
}

async function authedBlob(url) {
  const response = await fetch(url, { headers: { Authorization: `Bearer ${state.token}` } });
  if (!response.ok) throw new Error("file");
  return URL.createObjectURL(await response.blob());
}

async function refreshLiveFrame() {
  const img = document.getElementById("live-frame");
  const hint = document.getElementById("live-frame-hint");
  if (!img || !state.selectedSession) return;
  try {
    const url = await authedBlob(`/api/sessions/${state.selectedSession.id}/live-frame?t=${Date.now()}`);
    if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
    img.dataset.url = url;
    img.src = url;
    img.hidden = false;
    if (hint) hint.hidden = true;
  } catch {
    if (hint) hint.hidden = false;
  }
}

async function refreshWallFrame(sessionId) {
  const img = document.getElementById(`wall-${sessionId}`);
  if (!img) return;
  try {
    const url = await authedBlob(`/api/sessions/${sessionId}/live-frame?t=${Date.now()}`);
    if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
    img.dataset.url = url;
    img.src = url;
  } catch {
    img.removeAttribute("src");
  }
}

async function wallView() {
  stopLiveTimer();
  if (state.ws) {
    state.ws.close();
    state.ws = null;
    state.wsSessionId = "";
  }
  try {
    state.sessions = await api("/sessions");
    state.sessionsAt = Date.now();
  } catch {
    app.innerHTML = `<div class="card empty"><p class="muted">Не удалось загрузить сессии.</p></div>`;
    toast("Не удалось открыть Live", "err");
    return;
  }
  const tiles = state.sessions.filter((item) => item.online || ["active", "precheck"].includes(item.status));
  app.innerHTML = `
    <div class="page-head">
      <div>
        <h2>Стена камер</h2>
        <p class="muted">Красная рамка — есть критичные события. Нажмите карточку, чтобы открыть Live.</p>
      </div>
    </div>
    <div class="wall-grid">
      ${tiles.map((item) => `
        <article class="wall-tile ${Number(item.critical_open) ? "critical" : ""}" data-session="${esc(item.id)}">
          <img class="wall-frame" id="wall-${esc(item.id)}" alt="">
          <div class="wall-meta">
            <strong>${esc(item.student_name || item.student_email || item.id.slice(0, 8))}</strong>
            <div class="status-row">
              <span class="dot ${isOnline(item) ? "on" : "off"}"></span>
              <span class="badge ${esc(item.status)}">${esc(statusLabel(item.status))}</span>
              ${Number(item.critical_open) ? `<span class="badge compromised">критично ${esc(item.critical_open)}</span>` : ""}
            </div>
            <p class="muted">${esc(item.exam_title || "Экзамен")} · риск ${Number(item.risk_score || 0).toFixed(1)}</p>
          </div>
        </article>`).join("") || ""}
    </div>
    ${tiles.length ? "" : `
      <div class="card empty">
        <p class="muted">Сейчас никто не в эфире. Когда участник зайдёт на экзамен, его камера появится здесь.</p>
        <button class="primary" id="to-sessions" style="width:auto;margin:16px auto 0">К сессиям</button>
      </div>`}`;
  document.getElementById("to-sessions")?.addEventListener("click", () => showView("sessions"));
  app.querySelectorAll(".wall-tile").forEach((tile) => {
    tile.onclick = () => openSession(state.sessions.find((item) => item.id === tile.dataset.session));
  });
  tiles.forEach((item) => refreshWallFrame(item.id));
  state.liveTimer = setInterval(() => {
    tiles.forEach((item) => refreshWallFrame(item.id));
  }, 4000);
}

async function liveView() {
  if (!state.liveDetail) {
    return wallView();
  }
  stopLiveTimer();
  try {
    await pickLiveSession();
  } catch (error) {
    app.innerHTML = `<div class="card empty"><p class="muted">Не удалось загрузить сессии.</p></div>`;
    toast("Не удалось открыть Live", "err");
    return;
  }
  if (!state.selectedSession) {
    return wallView();
  }
  connectWs(state.selectedSession.id);
  try {
    state.violations = await api(`/sessions/${state.selectedSession.id}/violations`);
  } catch {
    state.violations = [];
  }
  await loadEvidence();
  const current = state.selectedSession;
  const openCritical = state.violations.filter((v) => !v.is_resolved && !v.is_reminder && EVENT_LABELS[v.type] && ["identity_mismatch", "agent_tamper", "forbidden_process", "multiple_faces", "heartbeat_lost"].includes(v.type));
  const openAll = state.violations.filter((v) => !v.is_resolved && !v.is_reminder);
  app.innerHTML = `
    <div class="page-head">
      <div>
        <h2>Live-мониторинг</h2>
        <p class="muted">${esc(current.student_name || current.student_email || current.id.slice(0, 8))} · ${esc(current.exam_title || "экзамен")}</p>
      </div>
      <div class="btn-row">
        <button class="ghost" id="to-wall" style="width:auto">К стене</button>
        <button class="ghost" id="open-review" style="width:auto">К проверке</button>
      </div>
    </div>
    <div class="card">
      <label>Участник
        <select id="live-session">
          ${state.sessions.map((item) => `
            <option value="${esc(item.id)}" ${item.id === current.id ? "selected" : ""}>
              ${esc(item.student_name || item.student_email || item.id.slice(0, 8))} — ${esc(statusLabel(item.status))}${item.online ? " · онлайн" : ""}
            </option>`).join("")}
        </select>
      </label>
    </div>
    <div class="grid">
      <div class="card">
        <h3>Камера</h3>
        <img id="live-frame" class="live-frame" alt="Кадр с камеры" hidden>
        <p class="muted" id="live-frame-hint">Кадр появится, когда агент начнёт отправку.</p>
      </div>
      <div class="card">
        <h3>Сводка</h3>
        <div class="status-row">
          <span class="dot ${isOnline(current) ? "on" : "off"}"></span>
          <strong>${isOnline(current) ? "Онлайн" : "Не на связи"}</strong>
          <span class="badge ${esc(current.status)}">${esc(statusLabel(current.status))}</span>
        </div>
        <div style="margin-top:10px">Риск: <strong>${Number(current.risk_score || 0).toFixed(1)}</strong></div>
        <div style="margin-top:6px">Открытых событий: <strong>${openAll.length}</strong></div>
        <div style="margin-top:6px">Критических: <strong>${openCritical.length || current.critical_open || 0}</strong></div>
        <p class="muted" id="live-heartbeat">Ожидание heartbeat…</p>
        <p class="muted" id="live-identity">Идентификация появится после кадра лица.</p>
        ${openCritical.length ? `<p class="crit-list">${openCritical.slice(0, 4).map((v) => esc(eventLabel(v.type))).join(" · ")}</p>` : "<p class='muted'>Критичных нарушений нет.</p>"}
      </div>
    </div>
    <div class="card"><h3>Лента событий</h3><div id="live-feed"></div></div>
    <div class="card"><h3>Клипы</h3><div id="live-clips"></div></div>`;
  document.getElementById("to-wall").onclick = () => {
    state.liveDetail = false;
    state.selectedSession = null;
    showView("live");
  };
  document.getElementById("open-review").onclick = () => showView("review");
  document.getElementById("live-session").onchange = (event) => {
    openSession(state.sessions.find((item) => item.id === event.target.value));
  };
  renderLiveFeed();
  renderLiveClips();
  refreshLiveFrame();
  state.liveTimer = setInterval(refreshLiveFrame, 3000);
}

function renderLiveFeed() {
  const feed = document.getElementById("live-feed");
  if (!feed) return;
  feed.innerHTML = state.violations.slice(0, 50).map((v) => `
    <div class="event ${v.is_resolved ? "resolved" : ""} ${v.is_reminder ? "reminder" : ""}">
      <strong>${esc(eventLabel(v.type))}</strong> — ${esc(v.message)}
      <div class="muted">${new Date(v.created_at).toLocaleString()}</div>
      <div class="btn-row" style="margin-top:8px">${evidenceButtonsHtml(v.id)}</div>
    </div>`).join("") || "<p class='muted'>Событий пока нет</p>";
  bindEvidenceButtons(feed);
}

function renderLiveClips() {
  const root = document.getElementById("live-clips");
  if (!root) return;
  const clips = state.evidence.filter((item) => (
    item.type === "video_clip" || item.type === "screen_still" || item.type === "webcam_still"
  )).slice(0, 12);
  if (!clips.length) {
    root.innerHTML = "<p class='muted'>Снимков пока нет</p>";
    return;
  }
  const label = (type) => {
    if (type === "screen_still") return "Стол";
    if (type === "webcam_still") return "Камера";
    return "Клип";
  };
  root.innerHTML = clips.map((item) => `
    <div class="clip-row">
      <button class="ghost" data-clip="${esc(item.url)}" data-kind="image" style="width:auto;margin:0">${label(item.type)}</button>
      <span class="muted">${new Date(item.created_at).toLocaleString()}</span>
    </div>`).join("");
  bindEvidenceButtons(root);
}

async function reviewView() {
  stopLiveTimer();
  try {
    await pickLiveSession();
  } catch {
    toast("Не удалось открыть проверку", "err");
    return showView("sessions");
  }
  if (!state.selectedSession) return showView("sessions");
  state.violations = await api(`/sessions/${state.selectedSession.id}/violations`);
  await loadEvidence();
  const outcome = state.selectedSession.summary?.outcome;
  const outcomeComment = state.selectedSession.summary?.outcome_comment || "";
  app.innerHTML = `
    <div class="page-head">
      <div>
        <h2>Проверка</h2>
        <p class="muted">Разберите события сессии, вынесите решение и выгрузите отчёт.</p>
      </div>
    </div>
    <div class="card">
      <h3>Решение по сессии</h3>
      <p>Сейчас: <strong>${esc(outcomeLabel(outcome))}</strong></p>
      <label>Комментарий<textarea id="outcome-comment" rows="3" placeholder="Кратко, почему зачёт или аннулирование">${esc(outcomeComment)}</textarea></label>
      <div class="btn-row" style="margin-top:12px">
        <button class="primary" id="pass-btn" style="width:auto;margin:0">Зачёт</button>
        <button class="ghost" id="invalidate-btn" style="width:auto;margin:0">Аннулировать</button>
      </div>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Тип</th><th>Сообщение</th><th>Снимки</th><th>Решение</th></tr></thead>
        <tbody>
          ${state.violations.map((v) => `
            <tr>
              <td>${esc(eventLabel(v.type))}</td>
              <td>${esc(v.message)}</td>
              <td class="btn-row">${evidenceButtonsHtml(v.id)}</td>
              <td class="btn-row">
                <button class="ghost" data-v="${esc(v.id)}" data-d="confirmed" style="width:auto;margin:0">Подтвердить</button>
                <button class="ghost" data-v="${esc(v.id)}" data-d="false_positive" style="width:auto;margin:0">Ложное</button>
                <button class="ghost" data-v="${esc(v.id)}" data-d="invalidate" style="width:auto;margin:0">Аннулировать</button>
              </td>
            </tr>`).join("") || "<tr><td colspan='4' class='muted'>Событий нет</td></tr>"}
        </tbody>
      </table>
    </div>
    <div class="btn-row">
      <button class="primary" id="export-csv" style="width:auto;margin:0">CSV</button>
      <button class="ghost" id="export-json" style="width:auto;margin:0">JSON</button>
      <button class="ghost" id="export-html" style="width:auto;margin:0">HTML</button>
    </div>`;
  async function saveOutcome(decision) {
    await api("/reviews", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.selectedSession.id,
        decision,
        comment: document.getElementById("outcome-comment").value.trim(),
      }),
    });
    toast(decision === "passed" ? "Зачёт сохранён" : "Сессия аннулирована");
    reviewView();
  }
  document.getElementById("pass-btn").onclick = () => saveOutcome("passed");
  document.getElementById("invalidate-btn").onclick = () => saveOutcome("invalidate");
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
      toast("Решение сохранено");
    };
  });
  async function downloadExport(format, filename, mime) {
    const data = await api(`/sessions/${state.selectedSession.id}/export?format=${format}`);
    const body = format === "csv" ? data.csv : format === "json" ? JSON.stringify(data, null, 2) : data;
    const blob = new Blob([body], { type: mime });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
  }
  document.getElementById("export-csv").onclick = () => downloadExport("csv", `session_${state.selectedSession.id}.csv`, "text/csv");
  document.getElementById("export-json").onclick = () => downloadExport("json", `session_${state.selectedSession.id}.json`, "application/json");
  document.getElementById("export-html").onclick = () => downloadExport("html", `session_${state.selectedSession.id}.html`, "text/html");
  bindEvidenceButtons(app);
}

async function adminView() {
  app.innerHTML = `
    <div class="page-head">
      <div>
        <h2>Админ</h2>
        <p class="muted">Интеграции и служебные настройки.</p>
      </div>
    </div>
    <div class="grid">
      <div class="card">
        <h3>Webhook</h3>
        <label>URL<input id="webhook-url" placeholder="https://example.com/hook"></label>
        <button class="primary" id="create-webhook">Добавить</button>
      </div>
      <div class="card">
        <h3>Копия базы</h3>
        <p class="muted">SQLite копируется в <code>data/backups</code>. Сервер также делает снимок примерно раз в 6 часов.</p>
        <button class="primary" id="backup-btn">Сделать копию сейчас</button>
      </div>
    </div>`;
  document.getElementById("create-webhook").onclick = async () => {
    await api("/webhooks", {
      method: "POST",
      body: JSON.stringify({ url: document.getElementById("webhook-url").value, events: ["violation.created"] }),
    });
    toast("Webhook добавлен");
  };
  document.getElementById("backup-btn").onclick = async () => {
    try {
      const result = await api("/admin/backup", { method: "POST", body: "{}" });
      toast(`Сохранено: ${result.path}`);
    } catch {
      toast("Не удалось сделать копию базы", "err");
    }
  };
}

async function showView(view) {
  if (view !== "live") stopLiveTimer();
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });
  if (!state.token && view !== "login") return loginView();
  if (view === "login") return loginView();
  if (!state.user) {
    try {
      state.user = await api("/users/me");
      userInfo.innerHTML = `<div class="name">${esc(state.user.full_name)}</div><div class="role">${esc(state.user.role)}</div>`;
      setupNav();
    } catch {
      state.token = "";
      localStorage.removeItem("iup_token");
      return loginView();
    }
  }
  try {
    if (view === "exams") return await examsView();
    if (view === "exam-detail") return await examDetailView();
    if (view === "sessions") return await sessionsView();
    if (view === "live") return await liveView();
    if (view === "review") return await reviewView();
    if (view === "admin") return await adminView();
  } catch (error) {
    console.error(error);
    toast("Не удалось открыть экран. Обновите страницу.", "err");
    console.error(error);
  }
}

async function init() {
  if (!state.token) return loginView();
  try {
    state.user = await api("/users/me");
    setupNav();
    showView(defaultView());
  } catch {
    state.token = "";
    localStorage.removeItem("iup_token");
    loginView();
  }
}

document.getElementById("logout-btn")?.addEventListener("click", logout);
document.getElementById("password-btn")?.addEventListener("click", () => {
  if (!state.user) return loginView();
  passwordView();
});
document.getElementById("evidence-close")?.addEventListener("click", closeEvidencePreview);
document.getElementById("evidence-modal")?.addEventListener("click", (event) => {
  if (event.target === event.currentTarget) closeEvidencePreview();
});

init();
