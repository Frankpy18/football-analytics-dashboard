# Dashboard interactivo en Python (Dash + Plotly) para análisis táctico y de rendimiento de partidos

Convierte un CSV con datos de partidos en un panel visual con métricas clave: PPG, Win %, Clean Sheets, BTTS, Over 2.5, análisis Home vs Away, análisis por mitades y un resumen H2H (Head-to-Head).

🎯 **Resumen rápido**

- **Script principal:** dashboard_final.py
- **Stack:** Python 3.8+, Pandas, NumPy, Dash, Plotly
- **Interfaz:** modo oscuro, layout en grid, tablas interactivas y gráficos dinámicos
- **Uso:** arrastra tu CSV en la interfaz y obtén análisis automáticos e interactivos

🚀 **Instalación rápida**

```
# Clona el repositorio
git clone https://github.com/TU_USUARIO/football-analytics-dashboard.git
cd football-analytics-dashboard

# Crea y activa un entorno virtual
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instala dependencias
pip install -r requirements.txt

# Si no tienes requirements.txt, instálalas manualmente:
pip install pandas numpy dash plotly
```

▶️ **Ejecutar el dashboard**

```
python dashboard_final.py
```

Luego abre tu navegador en: 👉 http://127.0.0.1:8050/  
Arrastra o selecciona tu archivo CSV. El dashboard procesará los datos y mostrará el análisis completo.

📂 **Formato del CSV**  
El script espera un CSV con columnas similares (mínimo las marcadas):

```
date_iso, aid, bid, aid_name, bid_name, liveA, liveB, type, side, scope, bc
```

| Columna      | Descripción                          |
|--------------|--------------------------------------|
| aid, bid     | IDs de los equipos                   |
| aid_name, bid_name | Nombres de los equipos          |
| liveA, liveB | Goles full-time                      |
| type         | team_history o h2h                   |
| side         | Indica el rol (A o B)                |
| bc           | (Opcional) marcador parcial o texto de resultado |

El script detecta automáticamente columnas HT (htA, htB, bc) y fechas (date_iso, date, kickoff, etc.). Si hay datos incompletos, el dashboard los omite para mantener la precisión.

📊 **Qué incluye el análisis**  

⚙️ **Detección automática**

- Identificación de equipos A (local) y B (visitante)
- Filtrado de datos inválidos y duplicados
- Procesamiento dinámico del CSV cargado

📈 **Métricas calculadas**

- **Generales:** PPG, Win %, BTTS, CS, FTS, AVG, xG, xGA
- **H2H (últimos 8):** victorias, empates, Over 1.5/2.5/3.5, BTTS, clean sheets
- **Segmentación Home/Away:** últimos 7 partidos locales y visitantes
- **Por Mitades:** promedio y % de goles 1T y 2T

📊 **Visualizaciones interactivas**

- Gráfico donut de dominio H2H
- Barras comparativas Home vs Away
- Tablas interactivas (Dash DataTable)
- Diseño profesional en modo oscuro

🧩 **Estructura recomendada**

```
football-analytics-dashboard/
├─ dashboard_final.py
├─ data/
│  └─ data.csv
├─ assets/
│  └─ demo.gif
├─ requirements.txt
└─ README.md
```
