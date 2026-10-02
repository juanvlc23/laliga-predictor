"""
Cuotas en vivo de The Odds API (https://the-odds-api.com), como segunda fuente.

football-data.co.uk recoge las cuotas una sola vez (viernes por la tarde para el
fin de semana). Este módulo las vuelve a leer hasta el cierre de la Quiniela,
para que la app vea los movimientos de viernes a sábado.

Necesita un secreto en GitHub (Settings → Secrets and variables → Actions):
  ODDS_API_KEY   la clave gratuita de the-odds-api.com

Si no está configurado no hace nada y la app sigue usando football-data.co.uk.

El plan gratis da 500 consultas al mes y cada lectura gasta 2 (Primera y
Segunda). Para no agotarlo, solo se vuelve a leer cuando ha pasado un tiempo
mínimo desde la última lectura (más corto el viernes y el sábado hasta el
cierre). Entre lecturas se usan las cuotas guardadas en data/cuotas_vivo.json.
"""

from __future__ import annotations

import difflib
import json
import os
import statistics
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ESTADO = ROOT / "data" / "cuotas_vivo.json"
API = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SPORTS = {"SP1": "soccer_spain_la_liga", "SP2": "soccer_spain_segunda_division"}
MADRID = ZoneInfo("Europe/Madrid")
LONDRES = ZoneInfo("Europe/London")   # football-data.co.uk da las horas en hora británica

DIAS_ADELANTE = 8        # partidos de la API que aún no están en football-data: solo los cercanos
HORAS_VALIDEZ = 12       # una lectura guardada deja de usarse si es más vieja que esto
RESERVA = 60             # por debajo de estas consultas restantes se entra en modo ahorro
COSTE = len(SPORTS)      # consultas que gasta cada lectura

# Nombres que no se resuelven solos al quitar acentos y siglas (clave: nombre normalizado).
ALIAS = {
    "athletic bilbao": "Ath Bilbao", "athletic": "Ath Bilbao", "athletic club": "Ath Bilbao",
    "atletico madrid": "Ath Madrid", "atl madrid": "Ath Madrid", "atletico": "Ath Madrid",
    "celta vigo": "Celta", "espanyol": "Espanol",
    "rayo vallecano": "Vallecano", "rayo": "Vallecano",
    "racing santander": "Santander", "racing": "Santander",
    "deportivo la coruna": "La Coruna", "deportivo": "La Coruna", "deportivo coruna": "La Coruna",
    "deportivo alaves": "Alaves",
    "sporting gijon": "Sp Gijon", "sporting": "Sp Gijon",
    "celta vigo b": "Celta B", "celta fortuna": "Celta B",
    "sociedad sanse": "Sociedad B", "sanse": "Sociedad B",
    "madrid castilla": "Real Madrid B", "castilla": "Real Madrid B",
    "barcelona atletic": "Barcelona B", "barca atletic": "Barcelona B",
    "bilbao athletic": "Ath Bilbao B", "athletic bilbao b": "Ath Bilbao B",
    "sevilla atletico": "Sevilla B", "villarreal b": "Villarreal B",
    "racing ferrol": "Ferrol", "cultural leonesa": "Leonesa",
}
# Palabras que delatan a un filial: nunca se empareja un filial con el primer equipo.
FILIAL = {"b", "c", "atletico", "atletic", "castilla", "promesas", "fortuna", "sanse", "mirandilla", "juvenil"}
SIGLAS = {"cf", "fc", "cd", "sd", "ud", "ca", "rc", "rcd", "ad", "ce", "sad", "club", "de", "real",
          "futbol", "balompie"}


# --------------------------------------------------------------------------
# Emparejar los nombres de la API con los de football-data.co.uk
# --------------------------------------------------------------------------

def normaliza(nombre: str) -> str:
    s = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode().lower()
    s = "".join(c if c.isalnum() else " " for c in s)
    fichas = ["b" if t == "ii" else t for t in s.split() if t not in SIGLAS]
    return " ".join(fichas)


def resolver(nombre: str, candidatos: list[str]) -> str | None:
    """Devuelve el nombre de football-data que corresponde a `nombre`, o None si hay duda."""
    n = normaliza(nombre)
    if not n:
        return None
    if n in ALIAS:                       # nombre conocido: o es ese equipo o no es ninguno
        return ALIAS[n] if ALIAS[n] in candidatos else None
    indice = {normaliza(c): c for c in candidatos}
    if n in indice:
        return indice[n]
    fichas = set(n.split())
    filial = bool(fichas & FILIAL)

    def compatible(k: str) -> bool:
        f = set(k.split())
        return bool(f & FILIAL) == filial and (f <= fichas or fichas <= f)

    cerca = [c for k, c in indice.items() if compatible(k)]
    if len(cerca) == 1:
        return cerca[0]
    parecidos = [k for k in difflib.get_close_matches(n, list(indice), n=2, cutoff=0.85)
                 if bool(set(k.split()) & FILIAL) == filial]
    return indice[parecidos[0]] if len(parecidos) == 1 else None


# --------------------------------------------------------------------------
# Cupo: cuándo toca volver a leer
# --------------------------------------------------------------------------

def minutos_minimos(ahora: datetime, restantes: int | None) -> int | None:
    """Minutos que deben pasar entre dos lecturas. None = no leer."""
    m = ahora.astimezone(MADRID)
    # Ventana que importa: viernes y sábado hasta el cierre de la Quiniela (14:00), de día.
    critico = (m.weekday() == 4 or (m.weekday() == 5 and m.hour < 14)) and m.hour >= 7
    if restantes is not None:
        if restantes < COSTE:
            return None
        if restantes < RESERVA:           # modo ahorro: solo en la ventana que importa
            return 230 if critico else None
    return 50 if critico else 320


def leer_estado() -> dict:
    try:
        e = json.loads(ESTADO.read_text())
        return e if isinstance(e, dict) else {}
    except (OSError, ValueError):
        return {}


def _fecha(txt: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(str(txt).replace("Z", "+00:00"))
    except ValueError:
        return None


def _iso(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Lectura de la API
# --------------------------------------------------------------------------

def mediana_cuotas(ev: dict) -> tuple[list[float], int] | None:
    """Cuota mediana 1-X-2 entre las casas (ignora mercados incompletos)."""
    h, d, a = [], [], []
    for casa in ev.get("bookmakers") or []:
        for mercado in casa.get("markets") or []:
            if mercado.get("key") != "h2h":
                continue
            precios = {o.get("name"): o.get("price") for o in mercado.get("outcomes") or []}
            trio = (precios.get(ev.get("home_team")), precios.get("Draw"), precios.get(ev.get("away_team")))
            if all(isinstance(x, (int, float)) and x > 1 for x in trio):
                h.append(trio[0]); d.append(trio[1]); a.append(trio[2])
    if not h:
        return None
    return [round(statistics.median(x), 3) for x in (h, d, a)], len(h)


def consultar(clave: str) -> tuple[dict[str, list], int | None, str | None]:
    """Devuelve (eventos por división, consultas restantes, error o None)."""
    import requests
    eventos, restantes, error = {}, None, None
    for div, sport in SPORTS.items():
        try:
            r = requests.get(API.format(sport=sport), timeout=30, params={
                "apiKey": clave, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal", "dateFormat": "iso"})
        except requests.RequestException as e:
            error = f"error de red ({type(e).__name__})"   # sin la URL: lleva la clave
            print(f"Cuotas en vivo: {error}.")
            continue
        try:
            restantes = int(float(r.headers.get("x-requests-remaining")))
        except (TypeError, ValueError):
            pass
        if r.status_code != 200:
            error = {401: "la clave ODDS_API_KEY no es válida", 429: "cupo mensual agotado"}.get(
                r.status_code, f"la API respondió HTTP {r.status_code}")
            print(f"Cuotas en vivo: {sport} -> {error}")
            continue
        try:
            datos = r.json()
        except ValueError:
            datos = None
        if not isinstance(datos, list):
            error = "respuesta de la API ilegible"
            continue
        eventos[div] = datos
    return eventos, restantes, error


def actualizar(estado: dict, eventos: dict[str, list], equipos_div: dict[str, list[str]], ahora: datetime) -> list[str]:
    """Guarda en `estado` las cuotas previas al partido. Devuelve los partidos sin emparejar."""
    partidos = estado.setdefault("partidos", {})
    sin_emparejar = []
    for div, lista in eventos.items():
        candidatos = equipos_div.get(div) or []
        for ev in lista:
            inicio = _fecha(ev.get("commence_time"))
            if inicio is None or inicio <= ahora:
                continue        # ya empezado: sus cuotas son «en directo» y no sirven
            cuotas = mediana_cuotas(ev)
            if cuotas is None:
                continue
            local, visitante = resolver(ev.get("home_team", ""), candidatos), resolver(ev.get("away_team", ""), candidatos)
            if not local or not visitante or local == visitante:
                if inicio <= ahora + timedelta(days=DIAS_ADELANTE):
                    sin_emparejar.append(f"{ev.get('home_team')} – {ev.get('away_team')}")
                continue
            partidos[f"{div}|{local}|{visitante}"] = {
                "inicio": _iso(inicio), "cuotas": cuotas[0], "casas": cuotas[1], "leido": _iso(ahora)}
    # olvidar partidos de hace más de 3 días
    for k in [k for k, v in partidos.items() if (_fecha(v.get("inicio")) or ahora) < ahora - timedelta(days=3)]:
        del partidos[k]
    return sin_emparejar


# --------------------------------------------------------------------------
# Aplicar a los próximos partidos
# --------------------------------------------------------------------------

def aplicar(fixtures: pd.DataFrame, equipos_div: dict[str, list[str]], ahora: datetime,
            offline: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Sustituye las cuotas de football-data por las leídas en vivo cuando las hay, y añade
    los partidos cercanos que la API ya tiene y football-data aún no ha publicado.
    Devuelve los partidos y un resumen para la web y los avisos.
    """
    fx = fixtures.copy() if not fixtures.empty else pd.DataFrame(
        columns=["div", "date", "time", "home", "away", "oh", "od", "oa"])
    fx["fuente"] = ["football-data" if pd.notna(x) else None for x in fx.get("oh", [])]
    fx["leido"] = None
    info = {"activo": False, "partidos": 0, "leidas": None, "restantes": None, "sin_emparejar": [], "error": None}

    clave = os.environ.get("ODDS_API_KEY", "").strip()
    estado = leer_estado()
    if not clave and not estado.get("partidos"):
        return fx, info
    info["activo"] = bool(clave)

    if clave and not offline:
        ultima = _fecha(estado.get("consultado"))
        espera = minutos_minimos(ahora, estado.get("restantes"))
        if espera is None:
            print(f"Cuotas en vivo: quedan {estado.get('restantes')} consultas este mes; no se lee ahora.")
            # el cupo se renueva cada mes: volver a probar pasado un día para enterarse
            if ultima is None or ahora - ultima >= timedelta(days=1):
                espera = 0
        if espera is not None and (ultima is None or ahora - ultima >= timedelta(minutes=espera)):
            eventos, restantes, error = consultar(clave)
            if restantes is not None:
                estado["restantes"] = restantes
            estado["error"] = error
            if eventos:
                estado["sin_emparejar"] = actualizar(estado, eventos, equipos_div, ahora)
                if error is None:
                    estado["consultado"] = _iso(ahora)
                n = sum(len(v) for v in eventos.values())
                print(f"Cuotas en vivo: {n} partidos leídos; quedan {estado.get('restantes')} consultas este mes.")
            # si la lectura falla no se anota la hora: se reintenta en la siguiente actualización
            ESTADO.parent.mkdir(parents=True, exist_ok=True)
            ESTADO.write_text(json.dumps(estado, ensure_ascii=False, indent=1))
        elif ultima is not None:
            print(f"Cuotas en vivo: se usan las de {ultima.astimezone(MADRID):%d/%m %H:%M} (aún no toca volver a leer).")

    nuevas, leidas = [], []
    for k, v in (estado.get("partidos") or {}).items():
        div, local, visitante = k.split("|")
        inicio, leido = _fecha(v.get("inicio")), _fecha(v.get("leido"))
        cuotas = v.get("cuotas") or []
        if inicio is None or leido is None or len(cuotas) != 3:
            continue
        # válida si se leyó hace poco (o poco antes de empezar, si el partido ya empezó)
        if min(ahora, inicio) - leido > timedelta(hours=HORAS_VALIDEZ):
            continue
        fila = fx.index[(fx["div"] == div) & (fx["home"] == local) & (fx["away"] == visitante)]
        if len(fila):
            fx.loc[fila[0], ["oh", "od", "oa"]] = cuotas
            fx.loc[fila[0], "fuente"] = "en vivo"
            fx.loc[fila[0], "leido"] = v["leido"]
        elif ahora < inicio <= ahora + timedelta(days=DIAS_ADELANTE):
            uk = inicio.astimezone(LONDRES)
            nuevas.append({"div": div, "date": pd.Timestamp(uk.date()), "time": uk.strftime("%H:%M"),
                           "home": local, "away": visitante, "oh": cuotas[0], "od": cuotas[1], "oa": cuotas[2],
                           "fuente": "en vivo", "leido": v["leido"]})
        else:
            continue
        leidas.append(leido)
    if nuevas:
        fx = pd.concat([fx, pd.DataFrame(nuevas)], ignore_index=True)
        fx["date"] = pd.to_datetime(fx["date"])
        fx = fx.sort_values(["div", "date", "time"]).reset_index(drop=True)

    info.update(partidos=len(leidas), leidas=_iso(max(leidas)) if leidas else None,
                restantes=estado.get("restantes"), sin_emparejar=estado.get("sin_emparejar") or [],
                error=estado.get("error") if clave else None)
    return fx, info
