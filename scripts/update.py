"""
LaLiga Predictor - actualización de datos, modelo y predicciones 1X2.

Qué hace:
  1. Descarga el histórico de Primera División desde football-data.co.uk.
     Las temporadas pasadas solo se descargan una vez (quedan en data/raw/).
     La temporada actual se vuelve a descargar en cada ejecución.
  2. Descarga los próximos partidos (fixtures.csv de football-data.co.uk).
  3. Ajusta un modelo Dixon-Coles (Poisson con fuerza de ataque/defensa por
     equipo, ventaja de campo y corrección de empates), dando más peso a los
     partidos recientes.
  4. Calcula probabilidades 1/X/2 para los próximos partidos y aplica los
     ajustes manuales de data/ajustes.csv (bajas, rotaciones...).
  5. Mide cómo habría acertado el modelo en la temporada pasada y la actual
     (sin mirar el futuro) y lo compara con las casas de apuestas.
  6. Escribe los resultados en docs/data/*.json para la web.

Uso:
  python scripts/update.py              # normal (descarga de internet)
  python scripts/update.py --offline    # usa solo lo que ya hay en data/raw
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "docs" / "data"
AJUSTES = ROOT / "data" / "ajustes.csv"

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/SP1.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FIRST_SEASON = 2016          # 2016/17 -> 10 temporadas de histórico
DIVISION = "SP1"             # Primera División en football-data.co.uk

# Parámetros del modelo
XI = 0.0025                  # decaimiento temporal por día (vida media ~9 meses)
WINDOW_DAYS = 3 * 365        # partidos usados para ajustar el modelo
RIDGE = 0.5                  # regularización (ayuda con equipos recién ascendidos)
MAX_GOALS = 10


# --------------------------------------------------------------------------
# Descarga y caché
# --------------------------------------------------------------------------

def season_start(today: datetime) -> int:
    return today.year if today.month >= 7 else today.year - 1


def season_code(start: int) -> str:
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def http_get(url: str) -> bytes | None:
    import requests
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "laliga-predictor"})
    except requests.RequestException as e:
        print(f"  ! error de red en {url}: {e}")
        return None
    if r.status_code != 200 or not r.content.strip():
        print(f"  ! {url} -> HTTP {r.status_code}")
        return None
    return r.content


def download_history(offline: bool, today: datetime) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    current = season_start(today)
    for start in range(FIRST_SEASON, current + 1):
        code = season_code(start)
        path = RAW / f"SP1_{code}.csv"
        is_current = start == current
        if offline:
            continue
        if path.exists() and not is_current:
            continue  # temporada cerrada: ya la tenemos, no se vuelve a bajar
        print(f"Descargando temporada {start}/{start + 1}...")
        content = http_get(BASE_URL.format(code=code))
        if content:
            path.write_bytes(content)
    if not offline:
        print("Descargando próximos partidos...")
        content = http_get(FIXTURES_URL)
        if content:
            (RAW / "fixtures.csv").write_bytes(content)


def read_csv_bytes(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip")


def load_matches() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW.glob("SP1_*.csv")):
        df = read_csv_bytes(path)
        if "HomeTeam" not in df.columns:
            continue
        code = path.stem.split("_")[1]
        out = pd.DataFrame({
            "season": f"20{code[:2]}/{code[2:]}",
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "home": df["HomeTeam"].astype(str).str.strip(),
            "away": df["AwayTeam"].astype(str).str.strip(),
            "hg": pd.to_numeric(df.get("FTHG"), errors="coerce"),
            "ag": pd.to_numeric(df.get("FTAG"), errors="coerce"),
        })
        # Cuotas medias del mercado (el nombre de la columna cambió con los años)
        for trio in (("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A")):
            if all(c in df.columns for c in trio):
                for k, c in zip(("oh", "od", "oa"), trio):
                    out[k] = pd.to_numeric(df[c], errors="coerce")
                break
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    m = pd.concat(frames, ignore_index=True)
    m = m.dropna(subset=["date", "hg", "ag"])
    m = m[(m["home"] != "nan") & (m["away"] != "nan")]
    m["hg"] = m["hg"].astype(int)
    m["ag"] = m["ag"].astype(int)
    m["res"] = np.where(m.hg > m.ag, "1", np.where(m.hg == m.ag, "X", "2"))
    return m.sort_values("date").drop_duplicates(["date", "home", "away"]).reset_index(drop=True)


def load_fixtures(matches: pd.DataFrame) -> pd.DataFrame:
    path = RAW / "fixtures.csv"
    if not path.exists():
        return pd.DataFrame()
    df = read_csv_bytes(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    if "Div" not in df.columns:
        return pd.DataFrame()
    df = df[df["Div"].astype(str).str.strip() == DIVISION]
    if df.empty:
        return pd.DataFrame()
    fx = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "time": df["Time"].astype(str) if "Time" in df.columns else "",
        "home": df["HomeTeam"].astype(str).str.strip(),
        "away": df["AwayTeam"].astype(str).str.strip(),
    })
    for trio in (("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")):
        if all(c in df.columns for c in trio):
            for k, c in zip(("oh", "od", "oa"), trio):
                fx[k] = pd.to_numeric(df[c], errors="coerce").values
            break
    # quitar partidos que ya están jugados en el histórico
    if not matches.empty:
        played = set(zip(matches.home, matches.away, matches.date.dt.date))
        fx = fx[[(h, a, d.date() if pd.notna(d) else None) not in played
                 for h, a, d in zip(fx.home, fx.away, fx.date)]]
    return fx.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Modelo Dixon-Coles
# --------------------------------------------------------------------------

class Model:
    def __init__(self, teams, mu, home, att, dfn, rho):
        self.teams = list(teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.mu, self.home, self.att, self.dfn, self.rho = mu, home, att, dfn, rho

    def knows(self, team: str) -> bool:
        return team in self.idx

    def lambdas(self, h: str, a: str):
        i, j = self.idx[h], self.idx[a]
        lh = math.exp(self.mu + self.home + self.att[i] + self.dfn[j])
        la = math.exp(self.mu + self.att[j] + self.dfn[i])
        return lh, la

    def score_matrix(self, lh: float, la: float) -> np.ndarray:
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        r = self.rho
        m[0, 0] *= 1 - lh * la * r
        m[0, 1] *= 1 + lh * r
        m[1, 0] *= 1 + la * r
        m[1, 1] *= 1 - r
        m = np.clip(m, 0, None)
        return m / m.sum()


def fit_model(matches: pd.DataFrame, ref_date: pd.Timestamp) -> Model | None:
    data = matches[(matches.date < ref_date) & (matches.date >= ref_date - timedelta(days=WINDOW_DAYS))]
    if len(data) < 100:
        return None
    teams = sorted(set(data.home) | set(data.away))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = data.home.map(idx).to_numpy()
    ai = data.away.map(idx).to_numpy()
    hg = data.hg.to_numpy(float)
    ag = data.ag.to_numpy(float)
    days = (ref_date - data.date).dt.days.to_numpy(float)
    w = np.exp(-XI * days)

    def nll(p):
        mu, home = p[0], p[1]
        att, dfn = p[2:2 + n], p[2 + n:]
        llh = mu + home + att[hi] + dfn[ai]
        lla = mu + att[ai] + dfn[hi]
        lh, la = np.exp(llh), np.exp(lla)
        f = -np.sum(w * (hg * llh - lh + ag * lla - la)) + RIDGE * (att @ att + dfn @ dfn)
        rh = w * (hg - lh)
        ra = w * (ag - la)
        g = np.zeros_like(p)
        g[0] = -(rh.sum() + ra.sum())
        g[1] = -rh.sum()
        g_att = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * RIDGE * att
        g_def = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * RIDGE * dfn
        g[2:2 + n] = g_att
        g[2 + n:] = g_def
        return f, g

    p0 = np.zeros(2 + 2 * n)
    p0[0] = math.log(max(ag.mean(), 0.5))
    p0[1] = 0.2
    res = minimize(nll, p0, jac=True, method="L-BFGS-B")
    p = res.x
    mu, home, att, dfn = p[0], p[1], p[2:2 + n], p[2 + n:]

    # Corrección de Dixon-Coles para marcadores bajos (sobre todo empates 0-0 y 1-1)
    lh = np.exp(mu + home + att[hi] + dfn[ai])
    la = np.exp(mu + att[ai] + dfn[hi])
    low = (hg <= 1) & (ag <= 1)

    def neg_tau(r):
        t = np.ones(low.sum())
        h_, a_, lh_, la_ = hg[low], ag[low], lh[low], la[low]
        t = np.where((h_ == 0) & (a_ == 0), 1 - lh_ * la_ * r, t)
        t = np.where((h_ == 0) & (a_ == 1), 1 + lh_ * r, t)
        t = np.where((h_ == 1) & (a_ == 0), 1 + la_ * r, t)
        t = np.where((h_ == 1) & (a_ == 1), 1 - r, t)
        if np.any(t <= 0):
            return 1e9
        return -np.sum(w[low] * np.log(t))

    rho = minimize_scalar(neg_tau, bounds=(-0.25, 0.25), method="bounded").x
    return Model(teams, mu, home, att, dfn, rho)


def probs_1x2(model: Model, lh: float, la: float):
    m = model.score_matrix(lh, la)
    p1 = np.tril(m, -1).sum()
    px = np.trace(m)
    p2 = np.triu(m, 1).sum()
    i, j = np.unravel_index(np.argmax(m), m.shape)
    over = 1 - sum(m[a, b] for a in range(3) for b in range(3) if a + b <= 2)
    return float(p1), float(px), float(p2), f"{i}-{j}", float(over)


def implied(oh, od, oa):
    if any(pd.isna(x) or x <= 1 for x in (oh, od, oa)):
        return None
    inv = np.array([1 / oh, 1 / od, 1 / oa])
    return inv / inv.sum()


# --------------------------------------------------------------------------
# Elo (para la tabla de fuerza)
# --------------------------------------------------------------------------

def compute_elo(matches: pd.DataFrame, k=20.0, hfa=65.0) -> dict:
    elo: dict[str, float] = {}
    for h, a, hg, ag in zip(matches.home, matches.away, matches.hg, matches.ag):
        rh, ra = elo.get(h, 1500.0), elo.get(a, 1500.0)
        exp_h = 1 / (1 + 10 ** ((ra - rh - hfa) / 400))
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        mult = 1.0 if abs(hg - ag) <= 1 else (1.5 if abs(hg - ag) == 2 else (11 + abs(hg - ag)) / 8)
        delta = k * mult * (score - exp_h)
        elo[h], elo[a] = rh + delta, ra - delta
    return elo


# --------------------------------------------------------------------------
# Backtest: cómo habría acertado el modelo sin mirar el futuro
# --------------------------------------------------------------------------

def backtest(matches: pd.DataFrame, seasons: list[str]) -> dict:
    test = matches[matches.season.isin(seasons)].copy()
    if test.empty:
        return {}
    test["week"] = test.date.dt.to_period("W-MON").dt.start_time
    rows = []
    for week, block in test.groupby("week"):
        model = fit_model(matches, pd.Timestamp(week))
        if model is None:
            continue
        for r in block.itertuples():
            if not (model.knows(r.home) and model.knows(r.away)):
                continue
            lh, la = model.lambdas(r.home, r.away)
            p1, px, p2, _, _ = probs_1x2(model, lh, la)
            book = implied(getattr(r, "oh", np.nan), getattr(r, "od", np.nan), getattr(r, "oa", np.nan))
            rows.append({"season": r.season, "res": r.res, "p": (p1, px, p2),
                         "book": None if book is None else tuple(book)})

    def summarize(rs):
        if not rs:
            return None
        labels = ["1", "X", "2"]
        n = len(rs)
        hits = sum(labels[int(np.argmax(x["p"]))] == x["res"] for x in rs)
        ll = -np.mean([math.log(max(x["p"][labels.index(x["res"])], 1e-9)) for x in rs])
        with_book = [x for x in rs if x["book"] is not None]
        book_hits = sum(labels[int(np.argmax(x["book"]))] == x["res"] for x in with_book)
        buckets = []
        for thr in (0.5, 0.6, 0.7, 0.8):
            sel = [x for x in rs if max(x["p"]) >= thr]
            buckets.append({
                "umbral": thr,
                "partidos": len(sel),
                "aciertos": sum(labels[int(np.argmax(x["p"]))] == x["res"] for x in sel),
            })
        dist = {k: sum(x["res"] == k for x in rs) / n for k in labels}
        return {
            "partidos": n,
            "aciertos": int(hits),
            "acierto": hits / n,
            "logloss": float(ll),
            "acierto_casas": (book_hits / len(with_book)) if with_book else None,
            "partidos_con_cuotas": len(with_book),
            "por_confianza": buckets,
            "reparto_real": dist,
        }

    out = {"total": summarize(rows), "temporadas": {}}
    for s in seasons:
        out["temporadas"][s] = summarize([x for x in rows if x["season"] == s])
    return out


# --------------------------------------------------------------------------
# Ajustes manuales (bajas, rotaciones)
# --------------------------------------------------------------------------

def load_ajustes() -> dict:
    if not AJUSTES.exists():
        return {}
    df = pd.read_csv(AJUSTES, comment="#", skipinitialspace=True)
    out = {}
    for r in df.itertuples():
        try:
            out[str(r.equipo).strip()] = (float(r.factor_ataque), float(r.factor_defensa), str(getattr(r, "nota", "") or ""))
        except (ValueError, AttributeError):
            continue
    return out


# --------------------------------------------------------------------------
# Principal
# --------------------------------------------------------------------------

def confidence_label(pmax: float) -> str:
    if pmax >= 0.70:
        return "alta"
    if pmax >= 0.55:
        return "media"
    return "baja"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--today", help="fecha de referencia AAAA-MM-DD (para pruebas)")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    today = datetime.fromisoformat(args.today).replace(tzinfo=timezone.utc) if args.today else now
    download_history(args.offline, today)

    matches = load_matches()
    if matches.empty:
        print("No hay datos de partidos. Nada que hacer.")
        sys.exit(1)
    print(f"Partidos en el histórico: {len(matches)} "
          f"({matches.season.min()} a {matches.season.max()}, último {matches.date.max().date()})")

    ref = pd.Timestamp(today.date()) + timedelta(days=1)
    model = fit_model(matches, ref)
    if model is None:
        print("No hay suficientes partidos para ajustar el modelo.")
        sys.exit(1)
    print(f"Modelo ajustado: ventaja local x{math.exp(model.home):.2f}, rho={model.rho:.3f}")

    current_season = matches.season.max()
    season_teams = sorted(set(matches[matches.season == current_season].home))
    ajustes = load_ajustes()

    # Próximos partidos
    fixtures = load_fixtures(matches)
    preds = []
    skipped = []
    for r in fixtures.itertuples():
        if not (model.knows(r.home) and model.knows(r.away)):
            skipped.append(f"{r.home} - {r.away}")
            continue
        lh, la = model.lambdas(r.home, r.away)
        notas = []
        if r.home in ajustes:
            fa, fd, nota = ajustes[r.home]
            lh *= fa; la *= fd
            if nota: notas.append(f"{r.home}: {nota}")
        if r.away in ajustes:
            fa, fd, nota = ajustes[r.away]
            la *= fa; lh *= fd
            if nota: notas.append(f"{r.away}: {nota}")
        p1, px, p2, score, over = probs_1x2(model, lh, la)
        probs = {"1": p1, "X": px, "2": p2}
        pick = max(probs, key=probs.get)
        double = "1X" if p1 + px >= p2 + px else "X2"
        book = implied(getattr(r, "oh", np.nan), getattr(r, "od", np.nan), getattr(r, "oa", np.nan))
        preds.append({
            "fecha": r.date.strftime("%Y-%m-%d") if pd.notna(r.date) else None,
            "hora": str(r.time) if str(r.time) not in ("nan", "") else None,
            "local": r.home,
            "visitante": r.away,
            "goles_esperados": [round(lh, 2), round(la, 2)],
            "prob": {k: round(v, 4) for k, v in probs.items()},
            "pick": pick,
            "confianza": confidence_label(probs[pick]),
            "doble_oportunidad": {"pick": double, "prob": round(max(p1 + px, p2 + px), 4)},
            "marcador_probable": score,
            "over_2_5": round(over, 4),
            "casas": None if book is None else {k: round(float(v), 4) for k, v in zip("1X2", book)},
            "notas": notas,
        })
    if skipped:
        print("Partidos sin datos suficientes:", ", ".join(skipped))

    # Tabla de fuerza
    elo = compute_elo(matches)
    avg_goals = math.exp(model.mu)
    table = []
    for t in season_teams:
        if not model.knows(t):
            continue
        i = model.idx[t]
        table.append({
            "equipo": t,
            "elo": round(elo.get(t, 1500)),
            "ataque": round(math.exp(model.att[i]), 3),     # >1 marca más que la media
            "defensa": round(math.exp(model.dfn[i]), 3),    # <1 encaja menos que la media
        })
    table.sort(key=lambda x: -x["elo"])

    # Backtest de la temporada pasada y la actual
    seasons_sorted = sorted(matches.season.unique())
    bt_seasons = seasons_sorted[-2:]
    print("Calculando backtest de", ", ".join(bt_seasons), "...")
    bt = backtest(matches, bt_seasons)

    # Últimos resultados vs predicción (para ver aciertos recientes)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "actualizado": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "temporada": current_season,
        "partidos_historico": int(len(matches)),
        "ultimo_partido": matches.date.max().strftime("%Y-%m-%d"),
        "ventaja_local": round(math.exp(model.home), 3),
        "goles_media_visitante": round(avg_goals, 3),
    }
    (OUT / "predicciones.json").write_text(json.dumps({"meta": meta, "partidos": preds}, ensure_ascii=False, indent=1))
    (OUT / "equipos.json").write_text(json.dumps({"meta": meta, "equipos": table}, ensure_ascii=False, indent=1))
    (OUT / "backtest.json").write_text(json.dumps({"meta": meta, **bt}, ensure_ascii=False, indent=1))
    print(f"Listo: {len(preds)} predicciones, {len(table)} equipos.")
    if bt.get("total"):
        t = bt["total"]
        casas = f" (casas de apuestas: {t['acierto_casas']:.1%})" if t["acierto_casas"] else ""
        print(f"Backtest: {t['aciertos']}/{t['partidos']} = {t['acierto']:.1%}{casas}")


if __name__ == "__main__":
    main()
