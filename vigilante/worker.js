/**
 * Vigilante de LaLiga Predictor (Cloudflare Worker).
 *
 * - Cada hora comprueba cuándo se actualizaron por última vez las predicciones.
 *   Si llevan demasiado tiempo sin actualizarse, te avisa por Telegram con un
 *   botón para lanzar la actualización.
 * - Escucha tus órdenes en Telegram:
 *     /actualizar  lanza la actualización en GitHub
 *     /estado      te dice cuándo fue la última actualización y cuántos partidos tienen cuotas
 *
 * Secretos que hay que configurar en el Worker (Settings → Variables and Secrets):
 *   TELEGRAM_TOKEN    token del bot (el mismo que en GitHub)
 *   TELEGRAM_CHAT_ID  tu número de chat (el mismo que en GitHub)
 *   GITHUB_TOKEN      token de GitHub con permiso «Actions: Read and write» solo en laliga-predictor
 *   WEBHOOK_SECRET    una contraseña inventada (letras y números) para proteger el Worker
 */

const REPO = "juanvlc23/laliga-predictor";
const WORKFLOW = "update.yml";
const DATA_URL = `https://raw.githubusercontent.com/${REPO}/main/docs/data/predicciones.json`;
const WEB = "https://juanvlc23.github.io/laliga-predictor/";
const ACTIONS = `https://github.com/${REPO}/actions/workflows/${WORKFLOW}`;

// ---------- Hora de Madrid ----------
function madrid(date) {
  const parts = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Europe/Madrid", weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false,
  }).formatToParts(date);
  const get = (t) => parts.find((p) => p.type === t).value;
  const days = { Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6, Sun: 0 };
  return { dow: days[get("weekday")], hour: +get("hour") % 24, minute: +get("minute") };
}
function fmtMadrid(date) {
  return new Intl.DateTimeFormat("es-ES", {
    timeZone: "Europe/Madrid", weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
  }).format(date);
}

// Ventana de comprobación cada hora: de jueves 00:00 a sábado 14:00 (hora de Madrid).
function enVentanaHoraria(m) {
  return m.dow === 4 || m.dow === 5 || (m.dow === 6 && m.hour < 14);
}
// Minutos sin actualizar a partir de los cuales se considera que algo va mal.
function umbralMinutos(m) {
  return enVentanaHoraria(m) ? 90 : 14 * 60;
}
function horasSilencio(m) {
  return m.hour < 8; // de 00:00 a 08:00 no se envían avisos
}

// Decide si hay que avisar. Sin guardar estado: avisa al pasar el umbral y
// recuerda cada 3 horas mientras siga sin actualizarse.
export function debeAvisar(edadMin, ahora) {
  const m = madrid(ahora);
  if (horasSilencio(m)) return false;
  const exceso = edadMin - umbralMinutos(m);
  return exceso >= 0 && exceso % 180 < 60;
}

// ---------- Telegram y GitHub ----------
async function telegram(env, method, payload) {
  const r = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_TOKEN}/${method}`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload),
  });
  return r.json();
}
function enviar(env, texto, conBoton = false) {
  const payload = { chat_id: env.TELEGRAM_CHAT_ID, text: texto, parse_mode: "HTML", disable_web_page_preview: true };
  if (conBoton) payload.reply_markup = { inline_keyboard: [[{ text: "🔄 Actualizar ahora", callback_data: "actualizar" }]] };
  return telegram(env, "sendMessage", payload);
}
async function lanzarActualizacion(env) {
  const r = await fetch(`https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/dispatches`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`, Accept: "application/vnd.github+json",
      "User-Agent": "laliga-predictor-vigilante", "X-GitHub-Api-Version": "2022-11-28",
    },
    body: JSON.stringify({ ref: "main" }),
  });
  return r.status === 204 ? null : `GitHub respondió ${r.status}: ${(await r.text()).slice(0, 200)}`;
}
async function leerEstado() {
  const r = await fetch(`${DATA_URL}?t=${Date.now()}`, { cf: { cacheTtl: 0 } });
  if (!r.ok) throw new Error(`No se pudo leer el estado (HTTP ${r.status})`);
  const d = await r.json();
  const actualizado = new Date(d.meta.actualizado);
  const partidos = d.partidos || [];
  return {
    actualizado,
    edadMin: Math.round((Date.now() - actualizado.getTime()) / 60000),
    total: partidos.length,
    conCuotas: partidos.filter((p) => p.casas).length,
  };
}
function textoEdad(min) {
  if (min < 60) return `${min} min`;
  const h = Math.floor(min / 60), m = min % 60;
  return m ? `${h} h ${m} min` : `${h} h`;
}

// ---------- Comprobación programada ----------
export async function comprobar(env, ahora = new Date()) {
  const e = await leerEstado();
  if (!debeAvisar(e.edadMin, ahora)) return "ok";
  await enviar(env,
    `⚠️ <b>La app no se actualiza</b>\nÚltima actualización: ${fmtMadrid(e.actualizado)} (hace ${textoEdad(e.edadMin)}).\n` +
    `Pulsa el botón para lanzarla ahora o escribe /actualizar.`, true);
  return "avisado";
}

// ---------- Órdenes desde Telegram ----------
async function orden(env, texto, chatId) {
  if (String(chatId) !== String(env.TELEGRAM_CHAT_ID)) return; // solo obedece a tu chat
  const cmd = (texto || "").trim().split(/[\s@]/)[0].toLowerCase();
  if (cmd === "/actualizar" || cmd === "actualizar") {
    const err = await lanzarActualizacion(env);
    await enviar(env, err
      ? `❌ No he podido lanzar la actualización.\n${err}\nPuedes lanzarla a mano: ${ACTIONS}`
      : "🔄 Actualización lanzada. Tarda 1–2 minutos; escribe /estado para comprobarla.");
  } else if (cmd === "/estado" || cmd === "estado") {
    try {
      const e = await leerEstado();
      await enviar(env,
        `📊 <b>Estado de la app</b>\nÚltima actualización: ${fmtMadrid(e.actualizado)} (hace ${textoEdad(e.edadMin)}).\n` +
        `Próximos partidos: ${e.total} (${e.conCuotas} con cuotas de las casas).\n${WEB}`);
    } catch (err) {
      await enviar(env, `❌ ${err.message}`);
    }
  } else if (cmd === "/start" || cmd === "/ayuda") {
    await enviar(env, "Órdenes disponibles:\n/estado → última actualización y partidos con cuotas\n/actualizar → lanza la actualización ahora");
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    // Registro del webhook (se abre una vez en el navegador): /setup?key=WEBHOOK_SECRET
    if (url.pathname === "/setup") {
      if (url.searchParams.get("key") !== env.WEBHOOK_SECRET) return new Response("Clave incorrecta", { status: 403 });
      const hook = await telegram(env, "setWebhook", {
        url: `${url.origin}/telegram`, secret_token: env.WEBHOOK_SECRET, allowed_updates: ["message", "callback_query"],
      });
      await telegram(env, "setMyCommands", { commands: [
        { command: "estado", description: "Última actualización y partidos con cuotas" },
        { command: "actualizar", description: "Lanzar la actualización ahora" },
      ] });
      return new Response(hook.ok ? "✅ Listo. Escribe /estado a tu bot en Telegram." : `Error: ${JSON.stringify(hook)}`,
        { headers: { "content-type": "text/plain; charset=utf-8" } });
    }
    if (url.pathname === "/telegram" && request.method === "POST") {
      if (request.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.WEBHOOK_SECRET) return new Response("", { status: 403 });
      const u = await request.json();
      if (u.message) await orden(env, u.message.text, u.message.chat.id);
      if (u.callback_query) {
        await telegram(env, "answerCallbackQuery", { callback_query_id: u.callback_query.id });
        await orden(env, u.callback_query.data, u.callback_query.message.chat.id);
      }
      return new Response("ok");
    }
    return new Response("Vigilante de LaLiga Predictor", { status: 200 });
  },
  async scheduled(event, env, ctx) {
    ctx.waitUntil(comprobar(env).catch((e) => enviar(env, `❌ El vigilante no pudo comprobar la app: ${e.message}`)));
  },
};
