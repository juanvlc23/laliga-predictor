"""
Avisos por Telegram: cuando aparecen las cuotas de las casas para los próximos partidos
y cuando, después, cambia el signo más probable de alguno (por movimiento de cuotas).

Necesita dos secretos en GitHub (Settings → Secrets and variables → Actions):
  TELEGRAM_TOKEN    el token del bot (lo da @BotFather)
  TELEGRAM_CHAT_ID  tu número de chat (lo da @userinfobot)

Si no están configurados, no hace nada. Recuerda en data/avisos.json qué partidos
ya se avisaron y qué signo tenía cada uno, para no repetir el mismo aviso.
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ESTADO = ROOT / "data" / "avisos.json"
WEB = "https://juanvlc23.github.io/laliga-predictor/"
PICK = {"1": "1", "X": "X", "2": "2"}
MADRID = ZoneInfo("Europe/Madrid")
LONDRES = ZoneInfo("Europe/London")   # las horas de los partidos vienen en hora británica
# Un signo solo «cambia» si el nuevo supera al anterior por este margen. Evita avisos
# de ida y vuelta en partidos donde dos signos están prácticamente empatados.
MARGEN_CAMBIO = 0.01
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


ACTIONS = "https://github.com/juanvlc23/laliga-predictor/actions/workflows/update.yml"
BOTONES = {"inline_keyboard": [[{"text": "📱 Abrir la app", "url": WEB},
                                {"text": "🔄 Forzar actualización", "url": ACTIONS}]]}


def enviar(texto: str, silencioso: bool = False, botones: bool = True) -> bool:
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    import requests
    datos = {"chat_id": chat, "text": texto, "parse_mode": "HTML", "disable_web_page_preview": True,
             "disable_notification": silencioso}
    if botones:
        datos["reply_markup"] = BOTONES
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=30, json=datos)
        if r.status_code != 200:
            print(f"Aviso Telegram: error HTTP {r.status_code}: {r.text[:200]}")
            return False
        return True
    except requests.RequestException as e:
        print(f"Aviso Telegram: error de red: {e}")
        return False


def clave(p: dict) -> str:
    return f"{p['fecha']}|{p['local']}|{p['visitante']}"


def leer_estado() -> dict:
    try:
        e = json.loads(ESTADO.read_text())
        return e if isinstance(e, dict) else {}
    except (OSError, ValueError):
        return {}


def guardar_estado(estado: dict) -> None:
    ESTADO.write_text(json.dumps(estado, ensure_ascii=False, indent=1))


def inicio(p: dict) -> datetime | None:
    """Hora de comienzo del partido (None si no se conoce la fecha)."""
    try:
        h, m = (int(x) for x in (p.get("hora") or "23:59").split(":")[:2])
        return datetime.fromisoformat(p["fecha"]).replace(hour=h, minute=m, tzinfo=LONDRES)
    except (TypeError, ValueError, KeyError):
        return None


def avisar_cuotas(preds: list[dict]) -> bool:
    configurado = bool(os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))

    if os.environ.get("PRUEBA_AVISO", "").lower() == "true":
        ok = enviar("✅ <b>LaLiga Predictor</b>\nLos avisos funcionan. Te escribiré cuando haya cuotas nuevas.")
        print("Aviso de prueba:", "enviado" if ok else "NO enviado (revisa los secretos TELEGRAM_TOKEN y TELEGRAM_CHAT_ID)")

    if not configurado:
        return False
    estado = leer_estado()
    avisados = set(estado.get("avisados", []))
    nuevos = [p for p in preds if p.get("casas") and clave(p) not in avisados]
    if not nuevos:
        return False

    por_div = {}
    for p in nuevos:
        por_div[p["division"]] = por_div.get(p["division"], 0) + 1
    resumen = " · ".join(f"{d}: {n}" for d, n in sorted(por_div.items()))
    claros = sorted(nuevos, key=lambda p: -max(p["prob"].values()))[:5]
    lineas = [f"• {html.escape(p['local'])} – {html.escape(p['visitante'])}: <b>{PICK[p['pick']]}</b> "
              f"({round(100 * p['prob'][p['pick']])}%)" for p in claros]
    texto = ("⚽ <b>Ya hay cuotas de las casas</b>\n"
             f"Partidos nuevos con cuotas → {resumen}\n\n"
             "Los más claros:\n" + "\n".join(lineas) +
             "\n\nSi ya cargaste el boleto, abre la app y se recalcula sola.")
    if enviar(texto):
        avisados |= {clave(p) for p in nuevos}
        # guardar solo partidos de los próximos listados (para que el archivo no crezca sin fin)
        vigentes = {clave(p) for p in preds}
        estado["avisados"] = sorted(avisados & vigentes)
        guardar_estado(estado)
        print(f"Aviso enviado: {len(nuevos)} partidos con cuotas nuevas.")
        return True
    return False


def avisar_cambios(preds: list[dict], ahora: datetime | None = None) -> bool:
    """
    Avisa cuando cambia el signo más probable de un partido que aún no ha empezado.
    La primera vez que ve un partido solo anota su signo, sin avisar.
    """
    if not (os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")):
        return False
    ahora = ahora or datetime.now(timezone.utc)
    estado = leer_estado()
    antes = estado.get("signos") or {}
    signos, cambios = {}, []
    for p in preds:
        k, prob = clave(p), p["prob"]
        ant = antes.get(k)
        actual = {"pick": p["pick"], "prob": prob}
        comienzo = inicio(p)
        if ant is None or ant.get("pick") not in prob:
            signos[k] = actual                  # partido nuevo: se anota
        elif comienzo is not None and comienzo <= ahora:
            signos[k] = ant                     # ya empezado: no se toca
        elif p["pick"] == ant["pick"]:
            signos[k] = actual
        elif prob[p["pick"]] - prob[ant["pick"]] >= MARGEN_CAMBIO:
            signos[k] = actual
            cambios.append((p, ant))
        else:
            signos[k] = ant                     # diferencia mínima: se espera a que se confirme

    enviado = False
    if cambios:
        lineas = []
        for p, ant in cambios:
            c = inicio(p)
            cuando = f"{DIAS[c.astimezone(MADRID).weekday()]} {c.astimezone(MADRID):%d/%m}" if c else ""
            pr = p["prob"]
            lineas.append(
                f"• <b>{html.escape(p['local'])} – {html.escape(p['visitante'])}</b>"
                f"{' (' + cuando + ')' if cuando else ''}\n"
                f"   antes <b>{PICK[ant['pick']]}</b> ({round(100 * ant['prob'][ant['pick']])}%) → "
                f"ahora <b>{PICK[p['pick']]}</b> ({round(100 * pr[p['pick']])}%)\n"
                f"   1: {round(100 * pr['1'])}% · X: {round(100 * pr['X'])}% · 2: {round(100 * pr['2'])}%")
        n = len(cambios)
        texto = (f"🔁 <b>{'Ha cambiado el signo de 1 partido' if n == 1 else f'Ha cambiado el signo de {n} partidos'}</b>\n\n"
                 + "\n".join(lineas) +
                 "\n\nSi ya rellenaste la columna, revisa " + ("este partido" if n == 1 else "estos partidos") + " en la app.")
        enviado = enviar(texto)
        if enviado:
            print(f"Aviso enviado: {n} cambios de signo.")
        else:                                   # no se pudo enviar: se reintenta en la próxima actualización
            for p, ant in cambios:
                signos[clave(p)] = ant
    if signos != antes:
        estado["signos"] = signos
        estado.setdefault("avisados", [])
        guardar_estado(estado)
    return enviado


def avisar_ejecucion(preds: list[dict], meta: dict, cuotas: dict | None = None) -> None:
    """Mensaje silencioso en cada actualización, para saber que la app sigue funcionando."""
    if os.environ.get("AVISO_CADA_EJECUCION", "true").lower() == "false":
        return
    hora = datetime.now(timezone.utc).astimezone(MADRID).strftime("%d/%m %H:%M")
    con = sum(1 for p in preds if p.get("casas"))
    origen = {"schedule": "programada", "workflow_dispatch": "manual", "push": "por un cambio en la app"}.get(
        os.environ.get("GITHUB_EVENT_NAME", ""), "")
    texto = (f"✅ App actualizada ({hora}{', ' + origen if origen else ''})\n"
             f"Próximos partidos: {len(preds)}, {con} con cuotas de las casas.")
    if cuotas and cuotas.get("activo"):
        if cuotas.get("leidas"):
            leidas = datetime.fromisoformat(cuotas["leidas"].replace("Z", "+00:00")).astimezone(MADRID)
            texto += f"\nCuotas en vivo: {cuotas['partidos']} partidos, leídas el {leidas:%d/%m} a las {leidas:%H:%M}."
        else:
            texto += "\nCuotas en vivo: ahora mismo no hay; se usan las de football-data."
        if cuotas.get("restantes") is not None:
            texto += f" Quedan {cuotas['restantes']} consultas este mes."
        if cuotas.get("error"):
            texto += f"\n⚠️ Cuotas en vivo: {html.escape(cuotas['error'])}."
        if cuotas.get("sin_emparejar"):
            texto += "\n⚠️ Partidos de las cuotas en vivo que no he sabido emparejar: " + html.escape(
                "; ".join(cuotas["sin_emparejar"][:6]))
    if enviar(texto, silencioso=True):
        print("Aviso de ejecución enviado.")
