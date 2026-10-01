"""
LaLiga Predictor - actualización de datos, modelo y predicciones 1X2.

Qué hace:
  1. Descarga el histórico de Primera (SP1) y Segunda (SP2) desde
     football-data.co.uk. Las temporadas pasadas solo se descargan una vez
     (quedan en data/raw/). La temporada actual se vuelve a descargar siempre.
  2. Descarga los próximos partidos (fixtures.csv de football-data.co.uk).
  3. Ajusta un modelo Dixon-Coles conjunto para las dos divisiones (Poisson
     con fuerza de ataque/defensa por equipo, ventaja de campo y nivel de
     goles propios de cada división, y corrección de empates), dando más peso
     a los partidos recientes. Al ser conjunto, los equipos que suben o bajan
     conservan su historial.
  4. Mezcla la probabilidad del modelo con la de las casas de apuestas, con el
     peso que mejor funcionó en el backtest.
  5. Mide cómo habría acertado (modelo, casas y mezcla) en la temporada pasada
     y la actual, sin mirar el futuro.
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

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FIRST_SEASON = 2016          # 2016/17 -> 10 temporadas de histórico
DIVISIONS = {"SP1": "Primera", "SP2": "Segunda"}

# Parámetros del modelo
XI = 0.0025                  # decaimiento temporal por día (vida media ~9 meses)
WINDOW_DAYS = 3 * 365        # partidos usados para ajustar el modelo
RIDGE = 0.5                  # regularización (ayuda con equipos con pocos datos)
MAX_GOALS = 10
BLEND_GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
DEFAULT_BLEND = 0.5          # peso del modelo si no hay datos para elegirlo
LABELS = ["1", "X", "2"]


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
    if offline:
        return
    current = season_start(today)
    for div in DIVISIONS:
        for start in range(FIRST_SEASON, current + 1):
            code = season_code(start)
            path = RAW / f"{div}_{code}.csv"
            if path.exists() and start != current:
                continue  # temporada cerrada: ya la tenemos, no se vuelve a bajar
            print(f"Descargando {DIVISIONS[div]} {start}/{start + 1}...")
            content = http_get(BASE_URL.format(code=code, div=div))
            if content:
                path.write_bytes(content)
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
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df


def parse_dates(col: pd.Series) -> pd.Series:
    col = col.astype(str).str.strip()
    d = pd.to_datetime(col, format="%d/%m/%Y", errors="coerce")
    return d.fillna(pd.to_datetime(col, format="%d/%m/%y", errors="coerce"))


def add_odds(out: pd.DataFrame, df: pd.DataFrame) -> None:
    """Cuotas medias del mercado (el nombre de la columna cambió con los años)."""
    for trio in (("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A")):
        if all(c in df.columns for c in trio):
            for k, c in zip(("oh", "od", "oa"), trio):
                out[k] = pd.to_numeric(df[c], errors="coerce").values
            return
    for k in ("oh", "od", "oa"):
        out[k] = np.nan


def load_matches() -> pd.DataFrame:
    frames = []
    for div in DIVISIONS:
        for path in sorted(RAW.glob(f"{div}_*.csv")):
            df = read_csv_bytes(path)
            if "HomeTeam" not in df.columns:
                continue
            code = path.stem.split("_")[1]
            out = pd.DataFrame({
                "div": div,
                "season": f"20{code[:2]}/{code[2:]}",
                "date": parse_dates(df["Date"]),
                "home": df["HomeTeam"].astype(str).str.strip(),
                "away": df["AwayTeam"].astype(str).str.strip(),
                "hg": pd.to_numeric(df.get("FTHG"), errors="coerce"),
                "ag": pd.to_numeric(df.get("FTAG"), errors="coerce"),
            })
            add_odds(out, df)
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
    if "Div" not in df.columns:
        return pd.DataFrame()
    df = df[df["Div"].astype(str).str.strip().isin(DIVISIONS)].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    fx = pd.DataFrame({
        "div": df["Div"].astype(str).str.strip(),
        "date": parse_dates(df["Date"]),
        "time": df["Time"].astype(str) if "Time" in df.columns else "",
        "home": df["HomeTeam"].astype(str).str.strip(),
        "away": df["AwayTeam"].astype(str).str.strip(),
    })
    add_odds(fx, df)
    # quitar partidos que ya están jugados en el histórico
    if not matches.empty:
        played = set(zip(matches.home, matches.away, matches.date.dt.date))
        fx = fx[[(h, a, d.date() if pd.notna(d) else None) not in played
                 for h, a, d in zip(fx.home, fx.away, fx.date)]]
    return fx.sort_values(["div", "date", "time"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Modelo Dixon-Coles conjunto (Primera + Segunda)
# --------------------------------------------------------------------------

class Model:
    def __init__(self, teams, mu, home, mu2, home2, att, dfn, rho):
        self.teams = list(teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.mu, self.home, self.mu2, self.home2 = mu, home, mu2, home2
        self.att, self.dfn, self.rho = att, dfn, rho

    def knows(self, team: str) -> bool:
        return team in self.idx

    def lambdas(self, h: str, a: str, div: str = "SP1"):
        i, j = self.idx[h], self.idx[a]
        d = 1.0 if div == "SP2" else 0.0
        base = self.mu + self.mu2 * d
        lh = math.exp(base + self.home + self.home2 * d + self.att[i] + self.dfn[j])
        la = math.exp(base + self.att[j] + self.dfn[i])
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
    d2 = (data["div"] == "SP2").to_numpy(float)
    days = (ref_date - data.date).dt.days.to_numpy(float)
    w = np.exp(-XI * days)
    K = 4  # mu, home, mu2, home2

    def nll(p):
        mu, home, mu2, home2 = p[:K]
        att, dfn = p[K:K + n], p[K + n:]
        base = mu + mu2 * d2
        llh = base + home + home2 * d2 + att[hi] + dfn[ai]
        lla = base + att[ai] + dfn[hi]
        lh, la = np.exp(llh), np.exp(lla)
        f = -np.sum(w * (hg * llh - lh + ag * lla - la)) + RIDGE * (att @ att + dfn @ dfn)
        rh = w * (hg - lh)
        ra = w * (ag - la)
        g = np.zeros_like(p)
        g[0] = -(rh.sum() + ra.sum())
        g[1] = -rh.sum()
        g[2] = -((rh + ra) * d2).sum()
        g[3] = -(rh * d2).sum()
        g[K:K + n] = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * RIDGE * att
        g[K + n:] = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * RIDGE * dfn
        return f, g

    p0 = np.zeros(K + 2 * n)
    p0[0] = math.log(max(ag.mean(), 0.5))
    p0[1] = 0.2
    p = minimize(nll, p0, jac=True, method="L-BFGS-B").x
    mu, home, mu2, home2 = p[:K]
    att, dfn = p[K:K + n], p[K + n:]

    # Corrección de Dixon-Coles para marcadores bajos (sobre todo empates 0-0 y 1-1)
    base = mu + mu2 * d2
    lh = np.exp(base + home + home2 * d2 + att[hi] + dfn[ai])
    la = np.exp(base + att[ai] + dfn[hi])
    low = (hg <= 1) & (ag <= 1)

    def neg_tau(r):
        h_, a_, lh_, la_ = hg[low], ag[low], lh[low], la[low]
        t = np.ones(low.sum())
        t = np.where((h_ == 0) & (a_ == 0), 1 - lh_ * la_ * r, t)
        t = np.where((h_ == 0) & (a_ == 1), 1 + lh_ * r, t)
        t = np.where((h_ == 1) & (a_ == 0), 1 + la_ * r, t)
        t = np.where((h_ == 1) & (a_ == 1), 1 - r, t)
        if np.any(t <= 0):
            return 1e9
        return -np.sum(w[low] * np.log(t))

    rho = minimize_scalar(neg_tau, bounds=(-0.25, 0.25), method="bounded").x
    return Model(teams, mu, home, mu2, home2, att, dfn, rho)


def match_outputs(model: Model, lh: float, la: float) -> dict:
    m = model.score_matrix(lh, la)
    p = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
    i, j = np.unravel_index(np.argmax(m), m.shape)
    over = 1 - sum(m[a, b] for a in range(3) for b in range(3) if a + b <= 2)
    # Pleno al 15: goles de cada equipo en categorías 0, 1, 2, M (3 o más)
    cat = lambda g: min(g, 3)
    pleno = np.zeros((4, 4))
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            pleno[cat(a), cat(b)] += m[a, b]
    return {"p": p, "score": f"{i}-{j}", "over": float(over), "pleno": pleno}


def implied(oh, od, oa):
    if any(pd.isna(x) or x <= 1 for x in (oh, od, oa)):
        return None
    inv = np.array([1 / oh, 1 / od, 1 / oa])
    return inv / inv.sum()


def blend(pm: np.ndarray, pb: np.ndarray | None, w: float) -> np.ndarray:
    """Mezcla lineal: w * modelo + (1 - w) * casas."""
    if pb is None:
        return pm
    p = w * pm + (1 - w) * pb
    return p / p.sum()


# --------------------------------------------------------------------------
# Elo (para la tabla de fuerza)
# --------------------------------------------------------------------------

def compute_elo(matches: pd.DataFrame, k=20.0, hfa=65.0) -> dict:
    elo: dict[str, float] = {}
    for h, a, hg, ag in zip(matches.home, matches.away, matches.hg, matches.ag):
        rh, ra = elo.get(h, 1500.0), elo.get(a, 1500.0)
        exp_h = 1 / (1 + 10 ** ((ra - rh - hfa) / 400))
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        diff = abs(hg - ag)
        mult = 1.0 if diff <= 1 else (1.5 if diff == 2 else (11 + diff) / 8)
        delta = k * mult * (score - exp_h)
        elo[h], elo[a] = rh + delta, ra - delta
    return elo


# --------------------------------------------------------------------------
# Backtest: cómo habría acertado sin mirar el futuro
# --------------------------------------------------------------------------

def backtest_rows(matches: pd.DataFrame, seasons: list[str]) -> list[dict]:
    test = matches[matches.season.isin(seasons)].copy()
    if test.empty:
        return []
    test["week"] = test.date.dt.to_period("W-MON").dt.start_time
    rows = []
    for week, block in test.groupby("week"):
        model = fit_model(matches, pd.Timestamp(week))
        if model is None:
            continue
        for r in block.itertuples():
            if not (model.knows(r.home) and model.knows(r.away)):
                continue
            lh, la = model.lambdas(r.home, r.away, r.div)
            rows.append({
                "div": r.div, "season": r.season, "res": r.res,
                "model": match_outputs(model, lh, la)["p"],
                "book": implied(r.oh, r.od, r.oa),
            })
    return rows


def choose_blend(rows: list[dict]) -> float:
    rs = [x for x in rows if x["book"] is not None]
    if len(rs) < 200:
        return DEFAULT_BLEND
    best, best_ll = DEFAULT_BLEND, float("inf")
    for w in BLEND_GRID:
        ll = -np.mean([math.log(max(blend(x["model"], x["book"], w)[LABELS.index(x["res"])], 1e-9)) for x in rs])
        if ll < best_ll:
            best, best_ll = w, ll
    return best


def summarize(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    n = len(rows)

    def score(key_fn, rs):
        if not rs:
            return None
        probs = [key_fn(x) for x in rs]
        hits = sum(LABELS[int(np.argmax(p))] == x["res"] for p, x in zip(probs, rs))
        ll = -np.mean([math.log(max(p[LABELS.index(x["res"])], 1e-9)) for p, x in zip(probs, rs)])
        buckets = []
        for thr in (0.5, 0.6, 0.7, 0.8):
            sel = [(p, x) for p, x in zip(probs, rs) if max(p) >= thr]
            buckets.append({"umbral": thr, "partidos": len(sel),
                            "aciertos": int(sum(LABELS[int(np.argmax(p))] == x["res"] for p, x in sel))})
        # doble oportunidad: los dos resultados más probables
        dbl = sum(x["res"] in [LABELS[i] for i in np.argsort(p)[-2:]] for p, x in zip(probs, rs))
        return {"partidos": len(rs), "aciertos": int(hits), "acierto": hits / len(rs),
                "logloss": float(ll), "acierto_doble": dbl / len(rs), "por_confianza": buckets}

    with_book = [x for x in rows if x["book"] is not None]
    return {
        "partidos": n,
        "modelo": score(lambda x: x["model"], rows),
        # comparación justa en los partidos que tienen cuotas
        "modelo_con_cuotas": score(lambda x: x["model"], with_book),
        "casas": score(lambda x: x["book"], with_book),
        "mezcla": score(lambda x: blend(x["model"], x["book"], x["w"]), rows),
        "reparto_real": {k: sum(x["res"] == k for x in rows) / n for k in LABELS},
    }


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
            nota = getattr(r, "nota", "")
            out[str(r.equipo).strip()] = (float(r.factor_ataque), float(r.factor_defensa),
                                          "" if pd.isna(nota) else str(nota))
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


def r4(x) -> float:
    return round(float(x), 4)


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
    for div, name in DIVISIONS.items():
        md = matches[matches["div"] == div]
        if not md.empty:
            print(f"{name}: {len(md)} partidos ({md.season.min()} a {md.season.max()}, último {md.date.max().date()})")

    ref = pd.Timestamp(today.date()) + timedelta(days=1)
    model = fit_model(matches, ref)
    if model is None:
        print("No hay suficientes partidos para ajustar el modelo.")
        sys.exit(1)
    print(f"Modelo: ventaja local Primera x{math.exp(model.home):.2f}, "
          f"Segunda x{math.exp(model.home + model.home2):.2f}, rho={model.rho:.3f}")

    # Backtest (temporada pasada + actual) y peso de la mezcla
    seasons_sorted = sorted(matches.season.unique())
    bt_seasons = seasons_sorted[-3:]
    print("Calculando backtest de", ", ".join(bt_seasons), "...")
    rows = backtest_rows(matches, bt_seasons)
    weights = {}
    for div, name in DIVISIONS.items():
        rd = [x for x in rows if x["div"] == div]
        weights[div] = choose_blend(rd)
        for x in rd:
            x["w"] = weights[div]
        print(f"{name}: peso del modelo en la mezcla {weights[div]:.0%} (casas {1 - weights[div]:.0%})")

    backtest = {"peso_modelo": {DIVISIONS[d]: v for d, v in weights.items()},
                "temporadas_evaluadas": bt_seasons, "divisiones": {}}
    for div, name in DIVISIONS.items():
        rd = [x for x in rows if x["div"] == div]
        backtest["divisiones"][name] = {
            "total": summarize(rd),
            "temporadas": {s: summarize([x for x in rd if x["season"] == s]) for s in bt_seasons},
        }

    # Próximos partidos
    ajustes = load_ajustes()
    fixtures = load_fixtures(matches)
    preds, skipped = [], []
    for r in fixtures.itertuples():
        if not (model.knows(r.home) and model.knows(r.away)):
            skipped.append(f"{r.home} - {r.away}")
            continue
        lh, la = model.lambdas(r.home, r.away, r.div)
        notas = []
        if r.home in ajustes:
            fa, fd, nota = ajustes[r.home]
            lh *= fa; la *= fd
            if nota: notas.append(f"{r.home}: {nota}")
        if r.away in ajustes:
            fa, fd, nota = ajustes[r.away]
            la *= fa; lh *= fd
            if nota: notas.append(f"{r.away}: {nota}")
        mo = match_outputs(model, lh, la)
        book = implied(r.oh, r.od, r.oa)
        final = blend(mo["p"], book, weights[r.div])
        k = int(np.argmax(final))
        top2 = sorted(np.argsort(final)[-2:])
        preds.append({
            "division": DIVISIONS[r.div],
            "fecha": r.date.strftime("%Y-%m-%d") if pd.notna(r.date) else None,
            "hora": str(r.time) if str(r.time) not in ("nan", "") else None,
            "local": r.home,
            "visitante": r.away,
            "goles_esperados": [round(lh, 2), round(la, 2)],
            "prob": dict(zip(LABELS, map(r4, final))),
            "prob_modelo": dict(zip(LABELS, map(r4, mo["p"]))),
            "casas": None if book is None else dict(zip(LABELS, map(r4, book))),
            "pick": LABELS[k],
            "confianza": confidence_label(final[k]),
            "doble_oportunidad": {"pick": "".join(LABELS[i] for i in top2), "prob": r4(final[top2].sum())},
            "marcador_probable": mo["score"],
            "over_2_5": r4(mo["over"]),
            "pleno": [[r4(x) for x in row] for row in mo["pleno"]],  # filas local 0,1,2,M; columnas visitante
            "notas": notas,
        })
    if skipped:
        print("Partidos sin datos suficientes:", ", ".join(skipped))

    # Tablas de fuerza por división
    elo = compute_elo(matches)
    current_season = matches.season.max()
    tables = {}
    for div, name in DIVISIONS.items():
        cur = matches[(matches["div"] == div) & (matches.season == current_season)]
        rowsT = []
        for t in sorted(set(cur.home) | set(cur.away)):
            if not model.knows(t):
                continue
            i = model.idx[t]
            rowsT.append({"equipo": t, "elo": round(elo.get(t, 1500)),
                          "ataque": round(math.exp(model.att[i]), 3),
                          "defensa": round(math.exp(model.dfn[i]), 3)})
        rowsT.sort(key=lambda x: -x["elo"])
        tables[name] = rowsT

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "actualizado": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "temporada": current_season,
        "partidos_historico": int(len(matches)),
        "ultimo_partido": {DIVISIONS[d]: matches[matches["div"] == d].date.max().strftime("%Y-%m-%d")
                           for d in DIVISIONS if (matches["div"] == d).any()},
        "peso_modelo": {DIVISIONS[d]: v for d, v in weights.items()},
    }
    (OUT / "predicciones.json").write_text(json.dumps({"meta": meta, "partidos": preds}, ensure_ascii=False, indent=1))
    (OUT / "equipos.json").write_text(json.dumps({"meta": meta, "divisiones": tables}, ensure_ascii=False, indent=1))
    (OUT / "backtest.json").write_text(json.dumps({"meta": meta, **backtest}, ensure_ascii=False, indent=1))

    # Parámetros del modelo, para que la web pueda predecir cualquier partido
    # de Primera o Segunda (por ejemplo los del boleto de la Quiniela) aunque
    # aún no esté en el listado de próximos partidos.
    team_div = {}
    for div in DIVISIONS:
        cur = matches[(matches["div"] == div) & (matches.season == current_season)]
        for t in set(cur.home) | set(cur.away):
            team_div[t] = div
    modelo = {
        "mu": model.mu, "home": model.home, "mu2": model.mu2, "home2": model.home2,
        "rho": model.rho, "max_goles": MAX_GOALS,
        "equipos": {t: {"att": float(model.att[model.idx[t]]), "def": float(model.dfn[model.idx[t]]),
                        "div": team_div.get(t)} for t in model.teams if t in team_div},
        "ajustes": {t: [fa, fd] for t, (fa, fd, _) in ajustes.items()},
        "reparto_medio": backtest["divisiones"]["Primera"]["total"]["reparto_real"]
        if backtest["divisiones"]["Primera"]["total"] else {"1": 0.45, "X": 0.27, "2": 0.28},
    }
    (OUT / "modelo.json").write_text(json.dumps({"meta": meta, **modelo}, ensure_ascii=False, indent=1))

    # Aviso por Telegram si han aparecido cuotas nuevas
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import avisos
        avisos.avisar_cuotas(preds)
    except Exception as e:  # nunca debe romper la actualización
        print(f"Avisos: error ({e}); se omite.")

    # Selecciones nacionales (para el Pleno al 15 y los parones de la Quiniela)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import selecciones
        selecciones.run(args.offline, ref, meta)
    except Exception as e:  # nunca debe romper la actualización de los clubes
        print(f"Selecciones: error ({e}); se omite.")
    print(f"Listo: {len(preds)} predicciones.")
    for name, b in backtest["divisiones"].items():
        t = b["total"]
        if not t:
            continue
        cm, cc, mz = t["modelo_con_cuotas"], t["casas"], t["mezcla"]
        print(f"  {name}: {t['partidos']} partidos | modelo {t['modelo']['acierto']:.1%} | "
              f"mezcla {mz['acierto']:.1%}" + (f" | casas {cc['acierto']:.1%} (modelo en esos {cm['acierto']:.1%})" if cc else ""))


if __name__ == "__main__":
    main()
