🏈 Football Analytics Dashboard
Un dashboard web interactivo construido con Python, Dash y Plotly que transforma archivos CSV de datos de partidos en un análisis de rendimiento visual y accionable.

Este proyecto fue creado como una pieza central de portafolio para demostrar habilidades en procesamiento de datos, cálculo de métricas complejas y visualización de datos interactiva.

🎥 Demo en Vivo
¡Un dashboard interactivo se ve mejor en acción!

🌟 Características Principales
Este dashboard va más allá de mostrar datos simples; calcula y presenta métricas clave para un análisis táctico:

Carga Dinámica de Archivos: Sube cualquier archivo CSV (con el formato esperado) y el dashboard generará el análisis completo al instante.

Análisis Cara a Cara (H2H):

Gráfico de "dona" (donut) que muestra el dominio histórico (Victorias/Empates/Derrotas).

Métricas clave como el % de partidos con +2.5 Goles y el % de Ambos Equipos Anotan (BTTS).

Métricas de Rendimiento del Equipo:

PPG (Puntos Por Partido): La métrica definitiva de consistencia.

CS (Clean Sheets / Valla Invicta): El % de partidos que el equipo mantiene su portería a cero.

FTS (Failed to Score / Sin Anotar): El % de partidos en los que el ataque del equipo falla.

Análisis Local vs. Visitante:

Segmenta automáticamente todas las métricas de rendimiento para mostrar la diferencia entre jugar en casa o fuera.

Patrones por Tiempos:

Analiza el promedio de goles anotados y recibidos en la Primera Mitad vs. la Segunda Mitad.

Diseño Profesional:

Construido con un layout de cuadrícula (grid) claro, profesional y en modo oscuro (dark mode) para una fácil lectura.

📊 Stack Tecnológico
Motor de Datos y Lógica: Python, Pandas, NumPy

Interfaz y Visualización: Dash, Plotly

Entorno: venv (estándar de Python)

🚀 Cómo Ejecutar el Proyecto Localmente
Sigue estos pasos para levantar el proyecto en tu máquina local.

1. Prerrequisitos
Python 3.8 o superior

Git

2. Instalación
Clona el repositorio:

Bash

git clone https://github.com/TU_USUARIO_DE_GITHUB/football-analytics-dashboard.git
cd football-analytics-dashboard
Crea y activa un entorno virtual (Recomendado):

Bash

# Para Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Para Windows
python -m venv venv
.\venv\Scripts\activate
Instala las dependencias: (Asegúrate de tener un archivo requirements.txt en tu repo)

Bash

pip install -r requirements.txt
(Nota: Si no tienes un requirements.txt, créalo con pip freeze > requirements.txt después de instalar pandas, dash y plotly).

📖 Uso
Una vez instaladas las dependencias, ejecuta el script principal:

Bash

python dashboard_final_v3_portfolio.py
Abre tu navegador web y ve a la siguiente dirección:

http://127.0.0.1:8050/
¡Arrastra tu archivo CSV de datos y mira cómo sucede la magia!

Datos de Muestra
Para probar el dashboard, puedes usar el archivo data.csv incluido en este repositorio.
