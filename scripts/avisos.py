"""
Avisos por Telegram cuando aparecen las cuotas de las casas para los próximos partidos.

Necesita dos secretos en GitHub (Settings → Secrets and variables → Actions):
  TELEGRAM_TOKEN    el token del bot (lo da @BotFather)
  TELEGRAM_CHAT_ID  tu número de chat (lo da @userinfobot)

Si no están configurados, no hace nada. Recuerda en data/avisos.json qué partidos
ya se avisaron, para no repetir el mismo aviso en cada actualización.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ESTADO = ROOT / "data" / "avisos.json"
WEB = "https://juanvlc23.github.io/laliga-predictor/"
PICK = {"1": "1", "X": "X", "2": "2"}


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


def avisar_cuotas(preds: list[dict]) -> bool:
    configurado = bool(os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))

    if os.environ.get("PRUEBA_AVISO", "").lower() == "true":
        ok = enviar("✅ <b>LaLiga Predictor</b>\nLos avisos funcionan. Te escribiré cuando haya cuotas nuevas.")
        print("Aviso de prueba:", "enviado" if ok else "NO enviado (revisa los secretos TELEGRAM_TOKEN y TELEGRAM_CHAT_ID)")

    if not configurado:
        return False
    estado = json.loads(ESTADO.read_text()) if ESTADO.exists() else {"avisados": []}
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
        ESTADO.write_text(json.dumps({"avisados": sorted(avisados & vigentes)}, ensure_ascii=False, indent=1))
        print(f"Aviso enviado: {len(nuevos)} partidos con cuotas nuevas.")
        return True
    return False


def avisar_ejecucion(preds: list[dict], meta: dict) -> None:
    """Mensaje silencioso en cada actualización, para saber que la app sigue funcionando."""
    if os.environ.get("AVISO_CADA_EJECUCION", "true").lower() == "false":
        return
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    hora = datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/Madrid")).strftime("%d/%m %H:%M")
    con = sum(1 for p in preds if p.get("casas"))
    origen = {"schedule": "programada", "workflow_dispatch": "manual", "push": "por un cambio en la app"}.get(
        os.environ.get("GITHUB_EVENT_NAME", ""), "")
    texto = (f"✅ App actualizada ({hora}{', ' + origen if origen else ''})\n"
             f"Próximos partidos: {len(preds)}, {con} con cuotas de las casas.")
    if enviar(texto, silencioso=True):
        print("Aviso de ejecución enviado.")
