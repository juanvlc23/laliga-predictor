# LaLiga Predictor

Predicciones 1X2 de Primera División con un modelo Dixon-Coles (Poisson con fuerza de ataque y defensa por equipo, ventaja de campo y corrección de empates).

## Cómo funciona

- **Datos:** resultados y cuotas de [football-data.co.uk](https://www.football-data.co.uk), desde la temporada 2016/17. Las temporadas cerradas se descargan una sola vez y quedan guardadas en `data/raw/`. La temporada actual y los próximos partidos se vuelven a descargar en cada actualización.
- **Actualización automática:** GitHub Actions ejecuta `scripts/update.py` cada día y guarda los resultados en `docs/data/`.
- **Web:** GitHub Pages publica `docs/index.html`, que muestra las predicciones, el acierto real del modelo en temporadas pasadas y la fuerza de cada equipo.

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
