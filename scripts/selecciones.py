"""
Modelo para partidos de selecciones nacionales (masculinas).

Datos: base de datos pública de resultados internacionales desde 1872
(https://github.com/martj42/international_results). Se descarga en cada
ejecución (es un solo archivo) y no se guarda en el repositorio.

Modelo: el mismo Dixon-Coles que para los clubes (ataque y defensa por
selección, ventaja de jugar en casa solo si el partido no es en campo neutral,
corrección de empates), con más peso a los partidos recientes y menos a los
amistosos. Exporta los parámetros a docs/data/selecciones.json para que la web
calcule cualquier partido (por ejemplo el Pleno al 15 en los parones).
"""

from __future__ import annotations

import io
import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "internacional.csv"
OUT = ROOT / "docs" / "data" / "selecciones.json"
URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

XI = 0.0005           # vida media ~4 años (lo que mejor predijo el último año)
WINDOW_DAYS = 6 * 365
RIDGE = 1.0
FRIENDLY_WEIGHT = 0.5
LABELS = ["1", "X", "2"]

# Nombres en español (como los escribe Loterías) -> nombre en la base de datos
ES = {
    "espana": "Spain", "rep checa": "Czech Republic", "republica checa": "Czech Republic", "chequia": "Czech Republic",
    "alemania": "Germany", "francia": "France", "italia": "Italy", "portugal": "Portugal", "inglaterra": "England",
    "escocia": "Scotland", "gales": "Wales", "irlanda del norte": "Northern Ireland", "irlanda": "Republic of Ireland",
    "rep irlanda": "Republic of Ireland", "republica de irlanda": "Republic of Ireland", "paises bajos": "Netherlands",
    "holanda": "Netherlands", "belgica": "Belgium", "suiza": "Switzerland", "austria": "Austria", "croacia": "Croatia",
    "serbia": "Serbia", "eslovenia": "Slovenia", "eslovaquia": "Slovakia", "hungria": "Hungary", "polonia": "Poland",
    "dinamarca": "Denmark", "suecia": "Sweden", "noruega": "Norway", "finlandia": "Finland", "islandia": "Iceland",
    "turquia": "Turkey", "grecia": "Greece", "rumania": "Romania", "bulgaria": "Bulgaria", "ucrania": "Ukraine",
    "rusia": "Russia", "bielorrusia": "Belarus", "georgia": "Georgia", "armenia": "Armenia", "azerbaiyan": "Azerbaijan",
    "kazajistan": "Kazakhstan", "albania": "Albania", "macedonia del norte": "North Macedonia", "macedonia": "North Macedonia",
    "montenegro": "Montenegro", "bosnia": "Bosnia and Herzegovina", "bosnia herzegovina": "Bosnia and Herzegovina",
    "bosnia y herzegovina": "Bosnia and Herzegovina", "kosovo": "Kosovo", "moldavia": "Moldova", "lituania": "Lithuania",
    "letonia": "Latvia", "estonia": "Estonia", "chipre": "Cyprus", "malta": "Malta", "luxemburgo": "Luxembourg",
    "liechtenstein": "Liechtenstein", "andorra": "Andorra", "san marino": "San Marino", "gibraltar": "Gibraltar",
    "islas feroe": "Faroe Islands", "feroe": "Faroe Islands", "israel": "Israel",
    "argentina": "Argentina", "brasil": "Brazil", "uruguay": "Uruguay", "colombia": "Colombia", "chile": "Chile",
    "peru": "Peru", "ecuador": "Ecuador", "paraguay": "Paraguay", "bolivia": "Bolivia", "venezuela": "Venezuela",
    "mexico": "Mexico", "estados unidos": "United States", "eeuu": "United States", "ee uu": "United States",
    "canada": "Canada", "costa rica": "Costa Rica", "panama": "Panama", "honduras": "Honduras", "jamaica": "Jamaica",
    "marruecos": "Morocco", "argelia": "Algeria", "tunez": "Tunisia", "egipto": "Egypt", "senegal": "Senegal",
    "nigeria": "Nigeria", "ghana": "Ghana", "camerun": "Cameroon", "costa de marfil": "Ivory Coast", "mali": "Mali",
    "sudafrica": "South Africa", "cabo verde": "Cape Verde", "guinea ecuatorial": "Equatorial Guinea",
    "japon": "Japan", "corea del sur": "South Korea", "corea": "South Korea", "australia": "Australia", "iran": "Iran",
    "arabia saudi": "Saudi Arabia", "arabia saudita": "Saudi Arabia", "catar": "Qatar", "qatar": "Qatar",
    "china": "China", "nueva zelanda": "New Zealand", "irak": "Iraq", "jordania": "Jordan", "uzbekistan": "Uzbekistan",
    "emiratos arabes": "United Arab Emirates", "emiratos arabes unidos": "United Arab Emirates",
}


def download(offline: bool) -> bool:
    if offline:
        return RAW.exists()
    import requests
    try:
        r = requests.get(URL, timeout=60)
        if r.status_code == 200 and len(r.content) > 100000:
            RAW.parent.mkdir(parents=True, exist_ok=True)
            RAW.write_bytes(r.content)
    except requests.RequestException as e:
        print(f"  ! selecciones: error de red: {e}")
    return RAW.exists()


def load() -> pd.DataFrame:
    d = pd.read_csv(RAW)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "home_score", "away_score"])
    d["neutral"] = d["neutral"].astype(str).str.upper().eq("TRUE")
    d["friendly"] = d["tournament"].eq("Friendly")
    d["hg"] = d["home_score"].astype(int)
    d["ag"] = d["away_score"].astype(int)
    d["res"] = np.where(d.hg > d.ag, "1", np.where(d.hg == d.ag, "X", "2"))
    return d.sort_values("date").reset_index(drop=True)


def fit(d: pd.DataFrame, ref: pd.Timestamp):
    data = d[(d.date < ref) & (d.date >= ref - timedelta(days=WINDOW_DAYS))]
    teams = sorted(set(data.home_team) | set(data.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = data.home_team.map(idx).to_numpy()
    ai = data.away_team.map(idx).to_numpy()
    hg = data.hg.to_numpy(float)
    ag = data.ag.to_numpy(float)
    hf = (~data.neutral).to_numpy(float)
    w = np.exp(-XI * (ref - data.date).dt.days.to_numpy(float)) * np.where(data.friendly, FRIENDLY_WEIGHT, 1.0)

    def nll(p):
        mu, home = p[0], p[1]
        att, dfn = p[2:2 + n], p[2 + n:]
        llh = mu + home * hf + att[hi] + dfn[ai]
        lla = mu + att[ai] + dfn[hi]
        lh, la = np.exp(llh), np.exp(lla)
        f = -np.sum(w * (hg * llh - lh + ag * lla - la)) + RIDGE * (att @ att + dfn @ dfn)
        rh, ra = w * (hg - lh), w * (ag - la)
        g = np.zeros_like(p)
        g[0] = -(rh.sum() + ra.sum())
        g[1] = -(rh * hf).sum()
        g[2:2 + n] = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * RIDGE * att
        g[2 + n:] = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * RIDGE * dfn
        return f, g

    p0 = np.zeros(2 + 2 * n)
    p0[0] = math.log(max(ag.mean(), 0.5))
    p = minimize(nll, p0, jac=True, method="L-BFGS-B").x
    mu, home, att, dfn = p[0], p[1], p[2:2 + n], p[2 + n:]
    lh = np.exp(mu + home * hf + att[hi] + dfn[ai])
    la = np.exp(mu + att[ai] + dfn[hi])
    low = (hg <= 1) & (ag <= 1)

    def neg_tau(r):
        h_, a_, lh_, la_ = hg[low], ag[low], lh[low], la[low]
        t = np.ones(low.sum())
        t = np.where((h_ == 0) & (a_ == 0), 1 - lh_ * la_ * r, t)
        t = np.where((h_ == 0) & (a_ == 1), 1 + lh_ * r, t)
        t = np.where((h_ == 1) & (a_ == 0), 1 + la_ * r, t)
        t = np.where((h_ == 1) & (a_ == 1), 1 - r, t)
        return 1e9 if np.any(t <= 0) else -np.sum(w[low] * np.log(t))

    rho = minimize_scalar(neg_tau, bounds=(-0.25, 0.25), method="bounded").x
    # partidos jugados en la ventana (para no fiarse de selecciones casi sin datos)
    games = np.bincount(hi, minlength=n) + np.bincount(ai, minlength=n)
    return {"mu": mu, "home": home, "rho": rho, "teams": teams, "att": att, "def": dfn, "games": games}


def probs(m, h, a, neutral=False):
    i, j = m["teams"].index(h), m["teams"].index(a)
    lh = math.exp(m["mu"] + (0 if neutral else m["home"]) + m["att"][i] + m["def"][j])
    la = math.exp(m["mu"] + m["att"][j] + m["def"][i])
    g = np.arange(11)
    from scipy.stats import poisson
    mat = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    r = m["rho"]
    mat[0, 0] *= 1 - lh * la * r; mat[0, 1] *= 1 + lh * r; mat[1, 0] *= 1 + la * r; mat[1, 1] *= 1 - r
    mat = np.clip(mat, 0, None); mat /= mat.sum()
    return np.array([np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()])


def backtest(d: pd.DataFrame, ref: pd.Timestamp) -> dict:
    """Ajusta con los datos de hace un año y predice el último año."""
    start = ref - timedelta(days=365)
    m = fit(d, start)
    test = d[(d.date >= start) & (d.date < ref)]
    hits, ll, n = 0, 0.0, 0
    for r in test.itertuples():
        if r.home_team not in m["teams"] or r.away_team not in m["teams"]:
            continue
        p = probs(m, r.home_team, r.away_team, r.neutral)
        hits += LABELS[int(np.argmax(p))] == r.res
        ll -= math.log(max(p[LABELS.index(r.res)], 1e-9))
        n += 1
    return {"partidos": n, "acierto": hits / n if n else None, "logloss": ll / n if n else None,
            "desde": start.strftime("%Y-%m-%d"), "hasta": ref.strftime("%Y-%m-%d")}


def run(offline: bool, ref: pd.Timestamp, meta: dict) -> None:
    if not download(offline):
        print("Selecciones: sin datos, se omite.")
        return
    d = load()
    m = fit(d, ref)
    bt = backtest(d, ref)
    teams = {t: {"att": float(m["att"][k]), "def": float(m["def"][k]), "partidos": int(m["games"][k])}
             for k, t in enumerate(m["teams"]) if m["games"][k] >= 8}
    out = {"meta": {**meta, "ultimo_partido_selecciones": d.date.max().strftime("%Y-%m-%d")},
           "mu": m["mu"], "home": m["home"], "rho": m["rho"], "max_goles": 10,
           "equipos": teams, "nombres_es": ES, "backtest": bt}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    ok = sum(1 for v in set(ES.values()) if v in teams)
    print(f"Selecciones: {len(teams)} equipos (último partido {out['meta']['ultimo_partido_selecciones']}), "
          f"backtest {bt['partidos']} partidos, acierto {bt['acierto']:.1%}; nombres en español válidos {ok}/{len(set(ES.values()))}")
