'use strict';

/* The login form. The server owns the attempt counting and the lockout, so this
   only submits the password and renders whatever the server says came back. */

const form = document.getElementById('gate-form');
const input = document.getElementById('password');
const submit = document.getElementById('gate-submit');
const error = document.getElementById('gate-error');

function showError(message) {
  error.textContent = message;
  error.hidden = !message;
}

/* A lock is worth counting down: without it the page just says "try later"
   and the reader has no idea when. */
function startCountdown(seconds) {
  let left = Math.max(0, Math.round(seconds));
  input.disabled = true;
  submit.disabled = true;

  const tick = () => {
    if (left <= 0) {
      clearInterval(timer);
      input.disabled = false;
      submit.disabled = false;
      showError('');
      input.focus();
      return;
    }
    const minutes = Math.floor(left / 60);
    const rest = String(left % 60).padStart(2, '0');
    showError(`تم قفل الدخول مؤقتاً. المتبقي ${minutes}:${rest}`);
    left -= 1;
  };

  tick();
  const timer = setInterval(tick, 1000);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!input.value) return;

  submit.disabled = true;
  showError('');

  let response;
  try {
    response = await fetch('/api/login', {
      method: 'POST',
      body: new FormData(form),
    });
  } catch (err) {
    submit.disabled = false;
    showError('تعذّر الاتصال بالخادم.');
    return;
  }

  const data = await response.json().catch(() => ({}));

  if (response.ok && data.ok) {
    window.location.replace('/');
    return;
  }

  input.value = '';
  if (response.status === 429) {
    startCountdown(data.retry_after || 0);
    return;
  }

  submit.disabled = false;
  showError(data.detail || 'كلمة المرور غير صحيحة.');
  input.focus();
});
