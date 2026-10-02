# LaLiga Predictor

Predicciones 1X2 de Primera y Segunda División y generador de columnas de Quiniela. Combina un modelo Dixon-Coles (Poisson con fuerza de ataque y defensa por equipo, ventaja de campo y corrección de empates) con las cuotas de las casas de apuestas, con el peso que mejor funcionó en temporadas anteriores.

## Cómo funciona

- **Datos:** resultados y cuotas de Primera (SP1) y Segunda (SP2) de [football-data.co.uk](https://www.football-data.co.uk), desde la temporada 2016/17. Las temporadas cerradas se descargan una sola vez y quedan guardadas en `data/raw/`. La temporada actual y los próximos partidos se vuelven a descargar en cada actualización.
- **Actualización automática:** GitHub Actions ejecuta `scripts/update.py` cada día y guarda los resultados en `docs/data/`.
- **Web:** GitHub Pages publica `docs/index.html`, que muestra las predicciones, el acierto real del modelo en temporadas pasadas y la fuerza de cada equipo. https://juanvlc23.github.io/laliga-predictor/

## Cuotas en vivo (opcional)

football-data.co.uk recoge las cuotas una sola vez, el viernes por la tarde. Para ver cómo se mueven hasta el cierre de la Quiniela, la app puede leerlas también de [The Odds API](https://the-odds-api.com) (plan gratis: 500 consultas al mes, la app gasta unas 260).

1. Crea una cuenta gratis en the-odds-api.com y copia la clave que te envían por correo.
2. En GitHub: Settings → Secrets and variables → Actions → New repository secret, con el nombre `ODDS_API_KEY` y la clave como valor.

Con la clave puesta, `scripts/cuotas_vivo.py` vuelve a leer las cuotas cada hora o dos el viernes y el sábado hasta las 14:00, y unas cuatro veces al día el resto de la semana. Si cambia el signo más probable de un partido, llega un aviso por Telegram. Sin la clave, la app sigue funcionando solo con football-data.co.uk.

## Historial de jugadas

En la pestaña Quiniela, el botón «Guardar esta jugada en el historial» apunta la columna que has jugado. La pestaña Historial cuenta los aciertos de cada jornada cuando se publican los resultados (`docs/data/resultados.json`), y lleva la cuenta de lo gastado y lo cobrado. El historial se guarda solo en el navegador; desde esa pestaña se puede descargar y recuperar una copia.

## Ajustes manuales (bajas, rotaciones)

Edita `data/ajustes.csv` desde la web de GitHub (icono del lápiz). Por ejemplo:

```
equipo,factor_ataque,factor_defensa,nota
Real Madrid,0.85,1.0,Baja del delantero titular
Sevilla,1.0,1.15,Portero lesionado
```

Al guardar, las predicciones se recalculan solas en un par de minutos.

## Aviso

Ningún modelo acierta la mayoría de los partidos con seguridad: el acierto realista en 1X2 está en torno al 50%. Consulta la pestaña "¿Cuánto acierta?" antes de fiarte de un pronóstico.
