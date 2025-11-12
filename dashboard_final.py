
import re
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import base64
import io



def detect_teams(df):
    """
    Detecta el equipo focus para side='A' (local) y side='B' (visitante):
    - Para cada side, cuenta frecuencia de team IDs en aid y bid.
    - El ID más frecuente es el focus (asumiendo CSV enfocado en matchup).
    - Extrae nombre del primer row coincidente.
    Adaptada para H2H si se llama en ese contexto.
    """
    # Side A (local focus)
    a_rows = df[df['side'] == 'A'].copy()
    a_team_ids = pd.concat([a_rows['aid'].dropna(), a_rows['bid'].dropna()]).astype(str)
    a_focus_id_str = Counter(a_team_ids).most_common(1)[0][0]
    a_focus_id = int(a_focus_id_str)
    a_name = a_rows[a_rows['aid'].astype(str) == a_focus_id_str]['aid_name'].iloc[0] if not a_rows[a_rows['aid'].astype(str) == a_focus_id_str].empty else \
             a_rows[a_rows['bid'].astype(str) == a_focus_id_str]['bid_name'].iloc[0]

    # Side B (visitante focus)
    b_rows = df[df['side'] == 'B'].copy()
    b_team_ids = pd.concat([b_rows['aid'].dropna(), b_rows['bid'].dropna()]).astype(str)
    b_focus_id_str = Counter(b_team_ids).most_common(1)[0][0]
    b_focus_id = int(b_focus_id_str)
    b_name = b_rows[b_rows['aid'].astype(str) == b_focus_id_str]['aid_name'].iloc[0] if not b_rows[b_rows['aid'].astype(str) == b_focus_id_str].empty else \
             b_rows[b_rows['bid'].astype(str) == b_focus_id_str]['bid_name'].iloc[0]

    return a_focus_id, a_name, b_focus_id, b_name


def prepare_team_data(df, team_id, team_name=None):
    """
    Prepara un DataFrame limpio para un equipo: incluye parseo de goles FT y extracción de HT.
    is_home = (aid == team_id), goals_for/against basado en posición.
    Usa todos los partidos válidos (scope != 'match').
    Retorna DF general para calcular Home/Away usando is_home, con columnas para mitades.
    """
    team_df = df[
        (df['aid'] == team_id) | (df['bid'] == team_id)
    ].copy()

    
    team_df = team_df[team_df['type'] == 'team_history']

    # is_home basado en posición en el partido (aid=home siempre)
    team_df['is_home'] = team_df['aid'] == team_id

    # Goles FT: Asegurar numéricos
    team_df['liveA_num'] = pd.to_numeric(team_df['liveA'], errors='coerce')
    team_df['liveB_num'] = pd.to_numeric(team_df['liveB'], errors='coerce')

    team_df['goals_for'] = np.where(team_df['is_home'], team_df['liveA_num'], team_df['liveB_num'])
    team_df['goals_against'] = np.where(team_df['is_home'], team_df['liveB_num'], team_df['liveA_num'])
    team_df['result'] = np.where(team_df['goals_for'] > team_df['goals_against'], 'W',
                                np.where(team_df['goals_for'] < team_df['goals_against'], 'L', 'D'))
    team_df['total_goals'] = team_df['goals_for'] + team_df['goals_against']
    team_df['points'] = np.where(team_df['result'] == 'W', 3, np.where(team_df['result'] == 'D', 1, 0))
    team_df['opp_id'] = np.where(team_df['is_home'], team_df['bid'], team_df['aid'])
    team_df['opp_name'] = np.where(team_df['is_home'], team_df['bid_name'], team_df['aid_name'])

    # Filtrar scopes relevantes (excluye 'match')
    team_df = team_df[team_df['scope'] != 'match'].sort_values('date_iso', ascending=False)

    # Limpiar NaNs en goles FT
    team_df = team_df.dropna(subset=['goals_for', 'goals_against'])

    # Eliminar duplicados por ID de partido (por si hay overlaps)
    team_df = team_df.drop_duplicates(subset=['id'])

    # Extracción de half-times (de Script 3)
    team_df = extract_half_times(team_df)

    # Ordenar por fecha (de Script 3, si es necesario)
    possible_date_cols = ['date_iso', 'date', 'kickoff', 'match_date', 'event_date', 'datetime']
    date_col = None
    for col in possible_date_cols:
        if col in team_df.columns:
            try:
                temp_df = team_df[col].copy()
                parsed = pd.to_datetime(temp_df, errors='coerce', dayfirst=True)
                if not parsed.isna().all():
                    team_df[col] = parsed
                    date_col = col
                    break
                parsed = pd.to_datetime(temp_df, errors='coerce')
                if not parsed.isna().all():
                    team_df[col] = parsed
                    date_col = col
                    break
            except Exception:
                continue
    if date_col:
        team_df = team_df.sort_values(date_col, ascending=False)

    return team_df[['date_iso', 'is_home', 'goals_for', 'goals_against', 'total_goals', 'result', 'points',
                    'opp_id', 'opp_name', 'match_name', 'scope', 'rq', 'rql', 'bc', 'id',
                    'ht_for', 'ht_against', 'sh_for', 'sh_against']]  # Incluir columnas de mitades

def extract_half_times(df):
    """
    Intenta obtener HT para A/B desde columnas comunes o parseando 'bc'.
    Agrega columnas: htA, htB, ht_for, ht_against, sh_for (second half for), sh_against.
    Si no encuentra HT, deja NaN en htA/htB y en second half.
    """
    df = df.copy()


    candidate_pairs = [
        ('htA', 'htB'), ('ht_a', 'ht_b'), ('htA_num', 'htB_num'), ('half_timeA', 'half_timeB'),
        ('htA_score', 'htB_score'), ('liveHTA', 'liveHTB'), ('htA1', 'htB1')
    ]

    found = False
    for a_col, b_col in candidate_pairs:
        if a_col in df.columns and b_col in df.columns:
            try:
                df['htA'] = pd.to_numeric(df[a_col], errors='coerce')
                df['htB'] = pd.to_numeric(df[b_col], errors='coerce')
                found = True
                break
            except Exception:
                continue

    if not found and 'bc' in df.columns:
        def parse_bc_to_ht(row):
            bc = row.get('bc')
            if pd.isna(bc):
                return (np.nan, np.nan)
            matches = re.findall(r"(\d+)[\s]*[-:\u2013][\s]*(\d+)", str(bc))
            if not matches:
                return (np.nan, np.nan)
            first = matches[0]
            try:
                return (int(first[0]), int(first[1]))
            except Exception:
                return (np.nan, np.nan)

        parsed = df.apply(parse_bc_to_ht, axis=1, result_type='expand')
        parsed.columns = ['htA_parsed', 'htB_parsed']
        df['htA'] = pd.to_numeric(parsed['htA_parsed'], errors='coerce')
        df['htB'] = pd.to_numeric(parsed['htB_parsed'], errors='coerce')


    def assign_team_ht_for(row):
        if pd.isna(row.get('htA')) or pd.isna(row.get('htB')):
            return (np.nan, np.nan)
        if row['is_home']:
            return (row['htA'], row['htB'])
        else:
            return (row['htB'], row['htA'])

    ht_pairs = df.apply(assign_team_ht_for, axis=1, result_type='expand')
    ht_pairs.columns = ['ht_for', 'ht_against']
    df['ht_for'] = pd.to_numeric(ht_pairs['ht_for'], errors='coerce')
    df['ht_against'] = pd.to_numeric(ht_pairs['ht_against'], errors='coerce')

    # second half = FT - HT (si HT disponible)
    df['sh_for'] = df['goals_for'] - df['ht_for']
    df['sh_against'] = df['goals_against'] - df['ht_against']

    # Llenar valores invalidos (si negative etc)
    df.loc[df['sh_for'] < 0, 'sh_for'] = np.nan
    df.loc[df['sh_against'] < 0, 'sh_against'] = np.nan

    return df

# Función para calcular Form (últ. 5 resultados como string)
def get_form_string(team_df, n=5, is_home_filter=None):
    """
    Retorna string de últimos n resultados (W/D/L), filtrado por home si especificado.
    """
    if is_home_filter is not None:
        filtered = team_df[team_df['is_home'] == is_home_filter].copy()
    else:
        filtered = team_df.copy()

    recent_results = filtered['result'].head(n).tolist()
    form_str = ''.join(recent_results[-5:])
    return form_str if len(form_str) == 5 else form_str.ljust(5, '-')

# Función para calcular PPG
def get_ppg(team_df):
    """
    PPG = avg points.
    """
    if len(team_df) == 0:
        return np.nan
    return team_df['points'].mean()

# Función para calcular FTS % (Failed to Score)
def get_fts_pct(team_df):
    """
    % matches where goals_for == 0.
    """
    if len(team_df) == 0:
        return np.nan
    return (team_df['goals_for'] == 0).mean() * 100

# Función para stats generales extendidas (Overall, Home, Away) - MODIFICADA PARA N/A SI <8
def calculate_general_stats(team_df, team_name):
    """
    Calcula stats generales (todos partidos): Form, PPG, Win%, AVG, Scored, Conceded, BTTS, CS, FTS, xG/xGA (proxy).
    Retorna 2 DataFrames: Form table y Stats table.
    Home/Away: solo los 7 más recientes (head(7)), usando is_home, eliminando N/A. Si 0, N/A.
    """
    if len(team_df) == 0:
        # Retornar DFs con N/A si no hay data
        form_df = pd.DataFrame({
            'Form': ['Overall', 'Home', 'Away'],
            'Results': ['N/A', 'N/A', 'N/A'],
            'PPG': ['N/A', 'N/A', 'N/A']
        })
        stats_df = pd.DataFrame({
            'Stats': ['Win %', 'AVG', 'Scored', 'Conceded', 'BTTS', 'CS', 'FTS', 'xG', 'xGA'],
            'Overall': ['N/A']*9,
            'Home': ['N/A']*9,
            'Away': ['N/A']*9
        })
        return form_df, stats_df


    overall_df = team_df.drop_duplicates(subset=['id']).copy()

    # Condicional para Overall: mínimo 8 partidos -> N/A si no
    if len(overall_df) < 8:
        print(f"Advertencia: {team_name} Overall tiene menos de 8 partidos ({len(overall_df)}), se usan N/A.")
        overall_form = 'N/A'
        overall_ppg = np.nan
        overall_dict = {'Win %': 'N/A', 'AVG': 'N/A', 'Scored': 'N/A', 'Conceded': 'N/A', 'BTTS': 'N/A', 'CS': 'N/A', 'FTS': 'N/A', 'xG': 'N/A', 'xGA': 'N/A'}
    else:
        overall_form = get_form_string(overall_df)
        overall_ppg = get_ppg(overall_df)
        overall_dict = basic_for_scope(overall_df)

    home_df = overall_df[overall_df['is_home'] == True].head(7).copy()  # 7 más recientes home
    away_df = overall_df[overall_df['is_home'] == False].head(7).copy()  # 7 más recientes away

    # Home Form y PPG (N/A si 0)
    if len(home_df) == 0:
        home_form = 'N/A'
        home_ppg = np.nan
        home_dict = {'Win %': 'N/A', 'AVG': 'N/A', 'Scored': 'N/A', 'Conceded': 'N/A', 'BTTS': 'N/A', 'CS': 'N/A', 'FTS': 'N/A', 'xG': 'N/A', 'xGA': 'N/A'}
    else:
        home_form = get_form_string(home_df)
        home_ppg = get_ppg(home_df)
        home_dict = basic_for_scope(home_df)

    # Away Form y PPG (N/A si 0)
    if len(away_df) == 0:
        away_form = 'N/A'
        away_ppg = np.nan
        away_dict = {'Win %': 'N/A', 'AVG': 'N/A', 'Scored': 'N/A', 'Conceded': 'N/A', 'BTTS': 'N/A', 'CS': 'N/A', 'FTS': 'N/A', 'xG': 'N/A', 'xGA': 'N/A'}
    else:
        away_form = get_form_string(away_df)
        away_ppg = get_ppg(away_df)
        away_dict = basic_for_scope(away_df)

    # Form table
    form_df = pd.DataFrame({
        'Form': ['Overall', 'Home', 'Away'],
        'Results': [overall_form, home_form, away_form],
        'PPG': [f"{overall_ppg:.2f}" if not np.isnan(overall_ppg) else 'N/A',
                f"{home_ppg:.2f}" if not np.isnan(home_ppg) else 'N/A',
                f"{away_ppg:.2f}" if not np.isnan(away_ppg) else 'N/A']
    })

    # Stats table
    stats_df = pd.DataFrame({
        'Stats': list(overall_dict.keys()),
        'Overall': list(overall_dict.values()),
        'Home': [home_dict.get(k, 'N/A') for k in overall_dict.keys()],
        'Away': [away_dict.get(k, 'N/A') for k in overall_dict.keys()]
    })

    return form_df, stats_df

def basic_for_scope(df_scope):
    if len(df_scope) == 0:
        return {'Win %': 'N/A', 'AVG': 'N/A', 'Scored': 'N/A', 'Conceded': 'N/A', 'BTTS': 'N/A', 'CS': 'N/A', 'FTS': 'N/A', 'xG': 'N/A', 'xGA': 'N/A'}
    wins_pct = (df_scope['result'] == 'W').mean() * 100
    avg_total = df_scope['total_goals'].mean()
    scored = df_scope['goals_for'].mean()
    conceded = df_scope['goals_against'].mean()
    btts = ((df_scope['goals_for'] > 0) & (df_scope['goals_against'] > 0)).mean() * 100
    cs = (df_scope['goals_against'] == 0).mean() * 100
    fts = get_fts_pct(df_scope)
    xg = scored  # proxy
    xga = conceded
    return {
        'Win %': f"{round(wins_pct)}%",
        'AVG': f"{avg_total:.2f}",
        'Scored': f"{scored:.2f}",
        'Conceded': f"{conceded:.2f}",
        'BTTS': f"{round(btts)}%",
        'CS': f"{round(cs)}%",
        'FTS': f"{round(fts)}%",
        'xG': f"{xg:.2f}",
        'xGA': f"{xga:.2f}"
    }


def scope_metrics(df_scope):
    """
    Calcula métricas solicitadas para un subconjunto (por ejemplo últimos 7 home matches).
    Retorna diccionario con métricas. Si df_scope vacío o insuficiente, retorna N/A.
    """
    out = {}
    n = len(df_scope)
    out['samples'] = n
    if n == 0:
        out.update({
            'goals_per_match': np.nan,
            'over_0_5_pct': np.nan,
            'over_1_5_pct': np.nan,
            'over_2_5_pct': np.nan,
            'over_3_5_pct': np.nan,
            'fts_pct': np.nan,
            'scored_1h_pct': np.nan,
            'scored_2h_pct': np.nan,
            'scored_both_halves_pct': np.nan,
            'avg_1h': np.nan,
            'avg_2h': np.nan,
            'conceded_per_match': np.nan,
            'con_over_0_5_pct': np.nan,
            'con_over_1_5_pct': np.nan,
            'con_over_2_5_pct': np.nan,
            'con_over_3_5_pct': np.nan,
            'clean_sheets_pct': np.nan,
            'clean_1h_pct': np.nan,
            'clean_2h_pct': np.nan,
            'avg_con_1h': np.nan,
            'avg_con_2h': np.nan
        })
        return out

    # Goals per match (scored by the team)
    out['goals_per_match'] = df_scope['goals_for'].mean()

    # Overs (FT, basados solo en goles anotados por el equipo - individual)
    out['over_0_5_pct'] = (df_scope['goals_for'] > 0.5).mean() * 100
    out['over_1_5_pct'] = (df_scope['goals_for'] > 1.5).mean() * 100
    out['over_2_5_pct'] = (df_scope['goals_for'] > 2.5).mean() * 100
    out['over_3_5_pct'] = (df_scope['goals_for'] > 3.5).mean() * 100

    # Failed to score
    out['fts_pct'] = (df_scope['goals_for'] == 0).mean() * 100

    # Halftime/Second half metrics (si existen columnas ht_for/sh_for y no todas NaN)
    if 'ht_for' in df_scope.columns and df_scope['ht_for'].notna().sum() >= 1:
        out['scored_1h_pct'] = (df_scope['ht_for'] > 0).mean() * 100
        out['avg_1h'] = df_scope['ht_for'].mean()
    else:
        out['scored_1h_pct'] = np.nan
        out['avg_1h'] = np.nan

    if 'sh_for' in df_scope.columns and df_scope['sh_for'].notna().sum() >= 1:
        out['scored_2h_pct'] = (df_scope['sh_for'] > 0).mean() * 100
        out['avg_2h'] = df_scope['sh_for'].mean()
    else:
        out['scored_2h_pct'] = np.nan
        out['avg_2h'] = np.nan

    # Scored in both halves
    if ('ht_for' in df_scope.columns and 'sh_for' in df_scope.columns) and (df_scope[['ht_for', 'sh_for']].notna().sum(axis=1) > 0).any():
        sub = df_scope.dropna(subset=['ht_for', 'sh_for'])
        if len(sub) > 0:
            out['scored_both_halves_pct'] = ((sub['ht_for'] > 0) & (sub['sh_for'] > 0)).mean() * 100
        else:
            out['scored_both_halves_pct'] = np.nan
    else:
        out['scored_both_halves_pct'] = np.nan

    # Conceded metrics
    out['conceded_per_match'] = df_scope['goals_against'].mean()

    # Con Overs (FT, basados solo en goles concedidos por el equipo - individual)
    out['con_over_0_5_pct'] = (df_scope['goals_against'] > 0.5).mean() * 100
    out['con_over_1_5_pct'] = (df_scope['goals_against'] > 1.5).mean() * 100
    out['con_over_2_5_pct'] = (df_scope['goals_against'] > 2.5).mean() * 100
    out['con_over_3_5_pct'] = (df_scope['goals_against'] > 3.5).mean() * 100

    # Clean sheets
    out['clean_sheets_pct'] = (df_scope['goals_against'] == 0).mean() * 100


    if 'ht_against' in df_scope.columns and df_scope['ht_against'].notna().sum() >= 1:
        out['clean_1h_pct'] = (df_scope['ht_against'] == 0).mean() * 100
        out['avg_con_1h'] = df_scope['ht_against'].mean()
    else:
        out['clean_1h_pct'] = np.nan
        out['avg_con_1h'] = np.nan

    if 'sh_against' in df_scope.columns and df_scope['sh_against'].notna().sum() >= 1:
        out['clean_2h_pct'] = (df_scope['sh_against'] == 0).mean() * 100
        out['avg_con_2h'] = df_scope['sh_against'].mean()
    else:
        out['clean_2h_pct'] = np.nan
        out['avg_con_2h'] = np.nan

    return out

# Función para mostrar métricas de goles (adaptada de Script 3, retorna dict para unificación)
def display_scoring_metrics(team_a_df, team_b_df, team_a_name, team_b_name):
    """
    Muestra comparación de métricas de goles para A Home (últimos 7) vs B Away (últimos 7).
    Retorna dict con métricas para exportación.
    """
    overall_a = team_a_df.drop_duplicates(subset=['id'])
    overall_b = team_b_df.drop_duplicates(subset=['id'])

    a_home = overall_a[overall_a['is_home'] == True].head(7).copy()
    b_away = overall_b[overall_b['is_home'] == False].head(7).copy()

    metrics_a_home = scope_metrics(a_home)
    metrics_b_away = scope_metrics(b_away)

    # Retornar dict unificado de métricas para exportación (con prefijos para A/B)
    scoring_dict = {}
    for key, val in metrics_a_home.items():
        if key != 'samples':
            scoring_dict[f'A_Home_{key}'] = val
    for key, val in metrics_b_away.items():
        if key != 'samples':
            scoring_dict[f'B_Away_{key}'] = val
    scoring_dict['A_Home_samples'] = len(a_home)
    scoring_dict['B_Away_samples'] = len(b_away)

    return scoring_dict, metrics_a_home, metrics_b_away  # Retornar también métricas individuales para mostrar en tablas


def calculate_h2h_metrics_df(df, team_a_id, team_a_name, team_b_id, team_b_name):
    """
    Filtra H2H (type='h2h'), toma últimos 8 (o menos si no hay), calcula wins, goles, overs, BTTS, clean sheets.
    Retorna DataFrame ordenado con métricas. Si 0, retorna DF vacío con N/A.
    """
    h2h_rows = df[df['type'] == 'h2h'].copy()
    if len(h2h_rows) == 0:
        h2h_metrics_df = pd.DataFrame({
            'Metric': ['Total Matches', f'{team_a_name} Wins', 'Draws', f'{team_b_name} Wins',
                       f'{team_a_name} Win %', 'Draw %', f'{team_b_name} Win %',
                       f'{team_a_name} Total Goals', f'{team_b_name} Total Goals',
                       'Over 1.5 %', 'Over 1.5 Count', 'Over 2.5 %', 'Over 2.5 Count',
                       'Over 3.5 %', 'Over 3.5 Count', 'BTTS %', 'BTTS Count',
                       f'{team_a_name} Clean Sheets %', f'{team_b_name} Clean Sheets %'],
            'Value': ['N/A'] * 19
        })
        return h2h_metrics_df

    # Limpiar goles numéricos y parsear fecha
    h2h_rows['liveA_num'] = pd.to_numeric(h2h_rows['liveA'], errors='coerce')
    h2h_rows['liveB_num'] = pd.to_numeric(h2h_rows['liveB'], errors='coerce')
    h2h_rows['date_iso'] = pd.to_datetime(h2h_rows['date_iso'])

    # Ordenar por fecha descendente y tomar últimos 8 (o menos)
    h2h_rows = h2h_rows.sort_values('date_iso', ascending=False).head(8).reset_index(drop=True)
    n_matches = len(h2h_rows)

    if n_matches == 0:
        h2h_metrics_df = pd.DataFrame({
            'Metric': ['Total Matches', f'{team_a_name} Wins', 'Draws', f'{team_b_name} Wins',
                       f'{team_a_name} Win %', 'Draw %', f'{team_b_name} Win %',
                       f'{team_a_name} Total Goals', f'{team_b_name} Total Goals',
                       'Over 1.5 %', 'Over 1.5 Count', 'Over 2.5 %', 'Over 2.5 Count',
                       'Over 3.5 %', 'Over 3.5 Count', 'BTTS %', 'BTTS Count',
                       f'{team_a_name} Clean Sheets %', f'{team_b_name} Clean Sheets %'],
            'Value': ['N/A'] * 19
        })
        return h2h_metrics_df

    # Inicializar contadores
    a_wins = 0
    draws = 0
    b_wins = 0
    a_goals_total = 0
    b_goals_total = 0
    over_15_count = 0
    over_25_count = 0
    over_35_count = 0
    btts_count = 0
    a_clean_sheets = 0
    b_clean_sheets = 0

    for _, row in h2h_rows.iterrows():
        home_goals = row['liveA_num']
        away_goals = row['liveB_num']
        total_goals = home_goals + away_goals

        # Determinar goles para equipo A y B
        if row['aid'] == team_a_id:
            a_goals = home_goals
            b_goals = away_goals
        else:
            a_goals = away_goals
            b_goals = home_goals

        a_goals_total += a_goals
        b_goals_total += b_goals

        # Resultado
        if a_goals > b_goals:
            a_wins += 1
        elif a_goals < b_goals:
            b_wins += 1
        else:
            draws += 1

        # Overs
        if total_goals > 1.5:
            over_15_count += 1
        if total_goals > 2.5:
            over_25_count += 1
        if total_goals > 3.5:
            over_35_count += 1

        # BTTS
        if a_goals > 0 and b_goals > 0:
            btts_count += 1

        # Clean sheets
        if b_goals == 0:  # Opponente no anota vs A
            a_clean_sheets += 1
        if a_goals == 0:  # Opponente no anota vs B
            b_clean_sheets += 1

    # Porcentajes
    a_win_pct = (a_wins / n_matches) * 100
    draw_pct = (draws / n_matches) * 100
    b_win_pct = (b_wins / n_matches) * 100

    over_15_pct = (over_15_count / n_matches) * 100
    over_25_pct = (over_25_count / n_matches) * 100
    over_35_pct = (over_35_count / n_matches) * 100
    btts_pct = (btts_count / n_matches) * 100
    a_cs_pct = (a_clean_sheets / n_matches) * 100
    b_cs_pct = (b_clean_sheets / n_matches) * 100

    # Crear DataFrame ordenado
    metrics_df = pd.DataFrame({
        'Metric': [
            'Total Matches',
            f'{team_a_name} Wins',
            'Draws',
            f'{team_b_name} Wins',
            f'{team_a_name} Win %',
            'Draw %',
            f'{team_b_name} Win %',
            f'{team_a_name} Total Goals',
            f'{team_b_name} Total Goals',
            'Over 1.5 %',
            f'Over 1.5 Count',
            'Over 2.5 %',
            f'Over 2.5 Count',
            'Over 3.5 %',
            f'Over 3.5 Count',
            'BTTS %',
            f'BTTS Count',
            f'{team_a_name} Clean Sheets %',
            f'{team_b_name} Clean Sheets %'
        ],
        'Value': [
            n_matches,
            a_wins,
            draws,
            b_wins,
            f"{round(a_win_pct)}%",
            f"{round(draw_pct)}%",
            f"{round(b_win_pct)}%",
            a_goals_total,
            b_goals_total,
            f"{round(over_15_pct)}%",
            f"{over_15_count} / {n_matches}",
            f"{round(over_25_pct)}%",
            f"{over_25_count} / {n_matches}",
            f"{round(over_35_pct)}%",
            f"{over_35_count} / {n_matches}",
            f"{round(btts_pct)}%",
            f"{btts_count} / {n_matches}",
            f"{round(a_cs_pct)}%",
            f"{round(b_cs_pct)}%"
        ]
    })

    return metrics_df


def create_unified_metrics_df(form_a, stats_a, form_b, stats_b, scoring_dict, h2h_df, team_a_name, team_b_name):
    """
    Crea un DataFrame unificado con todas las métricas en columnas.
    Rows: Métricas específicas. Columns: Categorías (e.g., A_Overall_Win%, A_Home_Scored, Scoring_A_Home_goals_per_match, H2H_Total_Matches, etc.).
    Para Form, se incluye como string. Para H2H, se pivotea el 'Value' a columnas.
    """
    unified_data = {}

    # General A - Form
    for _, row in form_a.iterrows():
        unified_data[f'A_{row["Form"]}_Results'] = row['Results']
        unified_data[f'A_{row["Form"]}_PPG'] = row['PPG']

    # General A - Stats (usar 'Stats' para clave)
    for _, row in stats_a.iterrows():
        stat_name = row['Stats'].replace(' ', '_').replace('%', 'pct').replace('/', '')  # Limpiar nombre
        unified_data[f'A_Overall_{stat_name}'] = row['Overall']
        unified_data[f'A_Home_{stat_name}'] = row['Home']
        unified_data[f'A_Away_{stat_name}'] = row['Away']

    # General B - Form
    for _, row in form_b.iterrows():
        unified_data[f'B_{row["Form"]}_Results'] = row['Results']
        unified_data[f'B_{row["Form"]}_PPG'] = row['PPG']

    # General B - Stats
    for _, row in stats_b.iterrows():
        stat_name = row['Stats'].replace(' ', '_').replace('%', 'pct').replace('/', '')  # Limpiar nombre
        unified_data[f'B_Overall_{stat_name}'] = row['Overall']
        unified_data[f'B_Home_{stat_name}'] = row['Home']
        unified_data[f'B_Away_{stat_name}'] = row['Away']

    # Scoring
    for key, val in scoring_dict.items():
        unified_data[key] = val

    
    h2h_pivot = pd.Series(h2h_df['Value'].values, index=h2h_df['Metric']).to_dict()
    for key, val in h2h_pivot.items():
        clean_key = re.sub(r'[^a-zA-Z0-9\s]', '_', key).replace(' ', '_')  
        unified_data[f'H2H_{clean_key}'] = val

    unified_df = pd.DataFrame([unified_data])  # Una fila con todas las columnas
    return unified_df


def process_csv(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df['liveA'] = pd.to_numeric(df['liveA'], errors='coerce')
    df['liveB'] = pd.to_numeric(df['liveB'], errors='coerce')

    TEAM_A_ID, TEAM_A_NAME, TEAM_B_ID, TEAM_B_NAME = detect_teams(df)

    team_a_df = prepare_team_data(df, TEAM_A_ID, TEAM_A_NAME)
    team_b_df = prepare_team_data(df, TEAM_B_ID, TEAM_B_NAME)

    form_a, stats_a = calculate_general_stats(team_a_df, TEAM_A_NAME)
    form_b, stats_b = calculate_general_stats(team_b_df, TEAM_B_NAME)

    scoring_dict, metrics_a_home, metrics_b_away = display_scoring_metrics(team_a_df, team_b_df, TEAM_A_NAME, TEAM_B_NAME)

    h2h_df = calculate_h2h_metrics_df(df, TEAM_A_ID, TEAM_A_NAME, TEAM_B_ID, TEAM_B_NAME)

    unified_df = create_unified_metrics_df(form_a, stats_a, form_b, stats_b, scoring_dict, h2h_df, TEAM_A_NAME, TEAM_B_NAME)

    return TEAM_A_NAME, TEAM_B_NAME, form_a, stats_a, form_b, stats_b, metrics_a_home, metrics_b_away, h2h_df, unified_df


# ==============================================================================
# === NUEVA CONFIGURACIÓN DEL DASHBOARD (V3 - PORTFOLIO GRID) ===
# ==============================================================================

app = dash.Dash(__name__)

# Paleta de colores y estilos
colors = {
    'background': '#1E1E1E', # Un gris oscuro, menos intenso que el negro
    'text': '#E0E0E0',
    'grid': '#444444',
    'primary': '#00A3FF', # Un azul más brillante y moderno
    'secondary': '#FFC107',
    'card': '#2A2A2A' # Color para los contenedores de gráficos
}

# Estilo para las tablas
table_style = {
    'style_cell': {
        'backgroundColor': colors['card'],
        'color': colors['text'],
        'border': f'1px solid {colors["grid"]}',
        'textAlign': 'center',
        'padding': '8px',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '13px'
    },
    'style_header': {
        'backgroundColor': '#004A7C',
        'color': '#FFFFFF',
        'fontWeight': 'bold',
        'border': f'1px solid {colors["grid"]}',
        'textAlign': 'center',
        'fontSize': '14px'
    },
    'style_data_conditional': [
        {
            'if': {'column_id': 'Metric'},
            'textAlign': 'left',
            'fontWeight': 'bold',
            'backgroundColor': '#333333'
        },
        {
            'if': {'column_id': 'Stats'},
            'textAlign': 'left',
            'fontWeight': 'bold',
            'backgroundColor': '#333333'
        },
         {
            'if': {'column_id': 'Form'},
            'textAlign': 'left',
            'fontWeight': 'bold',
            'backgroundColor': '#333333'
        }
    ]
}

# Estilos CSS para la cuadrícula
style_graph_container = {
    'backgroundColor': colors['card'],
    'borderRadius': '5px',
    'padding': '10px',
    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
}

# Layout de la App
app.layout = html.Div(style={'backgroundColor': colors['background'], 'color': colors['text'], 'padding': '20px', 'fontFamily': 'Arial, sans-serif'}, children=[
    html.H1("Dashboard de Análisis de Partidos", style={'textAlign': 'center', 'color': colors['primary'], 'marginBottom': '20px'}),
    
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Arrastra o ', html.A('Selecciona tu Archivo CSV')]),
        style={
            'width': '95%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '20px auto',
            'borderColor': colors['primary'],
            'backgroundColor': colors['card']
        },
        multiple=False
    ),
    
    # El contenedor principal para todos los resultados (gráficos y tablas)
    dcc.Loading(
        id="loading",
        type="default",
        children=html.Div(id='output-container')
    )
])

# Figura vacía mejorada
def create_empty_fig(message="Carga un CSV para ver los datos"):
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['card'],
        font_color=colors['text'],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[dict(text=message, showarrow=False, font_size=16, xref="paper", yref="paper", x=0.5, y=0.5)],
        height=300 # Altura fija
    )
    return fig


def create_compact_stats_fig(df, columns_to_plot, title, y_title, y_range=None):
    """
    Crea un gráfico de barras agrupado (Overall, Home, Away) para un
    conjunto específico de estadísticas.
    """
    fig = go.Figure()
    
    colors_bars = {'Overall': '#00A3FF', 'Home': '#007AB8', 'Away': '#69CFFF'}
    
    for col in columns_to_plot:
        fig.add_trace(go.Bar(
            name=col,
            x=df['Stats'],
            y=df[col],
            marker_color=colors_bars[col],
            hovertemplate=f'<b>%{{x}}</b><br>{col}: %{{y:.2f}}<extra></extra>'
        ))
        
    fig.update_layout(
        title_text=title,
        template='plotly_dark',
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['card'],
        font_color=colors['text'],
        barmode='group',
        height=350, # Altura fija
        margin=dict(l=40, r=20, t=60, b=40), # Márgenes reducidos
        title_font_size=18,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(title=y_title, range=y_range, gridcolor=colors['grid']),
        xaxis=dict(title=None, gridcolor=colors['grid'])
    )
    return fig


@app.callback(
    Output('output-container', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        # Layout inicial con gráficos vacíos
        return html.Div([
            html.H2("Análisis: (Esperando CSV)", style={'textAlign': 'center'}),
            html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '15px', 'justifyContent': 'center'}, children=[
                html.Div(dcc.Graph(figure=create_empty_fig("Resultados H2H")), style={**style_graph_container, 'flexBasis': '48%'}),
                html.Div(dcc.Graph(figure=create_empty_fig("Métricas H2H")), style={**style_graph_container, 'flexBasis': '48%'}),
                html.Div(dcc.Graph(figure=create_empty_fig("Goles Anotados")), style={**style_graph_container, 'flexBasis': '48%'}),
                html.Div(dcc.Graph(figure=create_empty_fig("Goles Encajados")), style={**style_graph_container, 'flexBasis': '48%'})
            ])
        ])

    try:
        team_a_name, team_b_name, form_a, stats_a, form_b, stats_b, metrics_a_home, metrics_b_away, h2h_df, unified_df = process_csv(contents, filename)
    except Exception as e:
        print(f"Error procesando el CSV: {e}")
        return html.Div([
            html.H3("Error al procesar el archivo.", style={'color': 'red', 'textAlign': 'center'}),
            html.P("Asegúrate de que el CSV tenga el formato esperado.", style={'textAlign': 'center'})
        ])

    # --- 1. Preparar Gráficos H2H ---
    h2h_values = h2h_df.set_index('Metric')['Value']
    a_wins = pd.to_numeric(h2h_values.get(f'{team_a_name} Wins'), errors='coerce')
    b_wins = pd.to_numeric(h2h_values.get(f'{team_b_name} Wins'), errors='coerce')
    draws = pd.to_numeric(h2h_values.get('Draws'), errors='coerce')
    
    fig_h2h_wins = go.Figure(data=[go.Pie(
        labels=[team_a_name, team_b_name, 'Empates'],
        values=[a_wins, b_wins, draws],
        hole=.4, # Convertido a Donut
        marker_colors=['#00A3FF', '#FFC107', '#999999'],
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Victorias: %{value}<br>%{percent}<extra></extra>'
    )])
    fig_h2h_wins.update_layout(
        title_text='H2H - Victorias (Últ. 8)',
        template='plotly_dark',
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['card'],
        font_color=colors['text'],
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        title_font_size=18,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1)
    )

    o15 = pd.to_numeric(h2h_values.get('Over 1.5 %', 'N/A').replace('%', ''), errors='coerce')
    o25 = pd.to_numeric(h2h_values.get('Over 2.5 %', 'N/A').replace('%', ''), errors='coerce')
    btts = pd.to_numeric(h2h_values.get('BTTS %', 'N/A').replace('%', ''), errors='coerce')

    fig_h2h_metrics = go.Figure(data=[go.Bar(
        x=['Over 1.5%', 'Over 2.5%', 'BTTS %'],
        y=[o15, o25, btts],
        marker_color=[colors['primary'], '#007AB8', '#69CFFF'],
        text=[f'{y:.0f}%' if not pd.isna(y) else 'N/A' for y in [o15, o25, btts]],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Porcentaje: %{y:.0f}%<extra></extra>'
    )])
    fig_h2h_metrics.update_layout(
        title_text='H2H - Métricas de Goles (%)',
        template='plotly_dark',
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['card'],
        font_color=colors['text'],
        height=300,
        margin=dict(l=40, r=20, t=60, b=40),
        title_font_size=18,
        yaxis=dict(title=None, range=[0, 100], gridcolor=colors['grid']),
        xaxis=dict(title=None, gridcolor=colors['grid'])
    )

    # --- 2. Preparar Gráficos de Scoring (Home vs Away) ---
    scoring_labels = ['Goles por Partido', 'Prom. Goles 1T', 'Prom. Goles 2T']
    a_scoring_vals = [metrics_a_home['goals_per_match'], metrics_a_home['avg_1h'], metrics_a_home['avg_2h']]
    b_scoring_vals = [metrics_b_away['goals_per_match'], metrics_b_away['avg_1h'], metrics_b_away['avg_2h']]

    fig_scoring = go.Figure(data=[
        go.Bar(name=f'{team_a_name} (Local)', x=scoring_labels, y=a_scoring_vals, marker_color=colors['primary'], hovertemplate=f'<b>%{{x}}</b><br>{team_a_name}: %{{y:.2f}}<extra></extra>'),
        go.Bar(name=f'{team_b_name} (Vis.)', x=scoring_labels, y=b_scoring_vals, marker_color=colors['secondary'], hovertemplate=f'<b>%{{x}}</b><br>{team_b_name}: %{{y:.2f}}<extra></extra>')
    ])
    fig_scoring.update_layout(
        title_text='Métricas de Goles Anotados (Local vs Vis.)',
        barmode='group',
        template='plotly_dark',
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['card'],
        font_color=colors['text'],
        height=350,
        margin=dict(l=40, r=20, t=60, b=40),
        title_font_size=18,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Promedio de Goles", gridcolor=colors['grid']),
        xaxis=dict(gridcolor=colors['grid'])
    )

    conceded_labels = ['Goles Encajados', '% Portería a Cero']
    a_conceded_vals = [metrics_a_home['conceded_per_match'], metrics_a_home['clean_sheets_pct']]
    b_conceded_vals = [metrics_b_away['conceded_per_match'], metrics_b_away['clean_sheets_pct']]

    fig_conceded = go.Figure(data=[
        go.Bar(name=f'{team_a_name} (Local)', x=conceded_labels, y=a_conceded_vals, marker_color=colors['primary'], hovertemplate=f'<b>%{{x}}</b><br>{team_a_name}: %{{y:.2f}}<extra></extra>'),
        go.Bar(name=f'{team_b_name} (Vis.)', x=conceded_labels, y=b_conceded_vals, marker_color=colors['secondary'], hovertemplate=f'<b>%{{x}}</b><br>{team_b_name}: %{{y:.2f}}<extra></extra>')
    ])
    fig_conceded.update_layout(
        title_text='Métricas de Goles Encajados (Local vs Vis.)',
        barmode='group',
        template='plotly_dark',
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['card'],
        font_color=colors['text'],
        height=350,
        margin=dict(l=40, r=20, t=60, b=40),
        title_font_size=18,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Valor", gridcolor=colors['grid']),
        xaxis=dict(gridcolor=colors['grid'])
    )

    # --- 3. Preparar Gráficos de Stats Generales (SIMPLIFICADOS) ---
    def clean_stats_df(df):
       df_num = df.copy()
       for col in ['Overall', 'Home', 'Away']:
           df_num[col] = pd.to_numeric(df_num[col].astype(str).str.replace('%', ''), errors='coerce')
       return df_num.set_index('Stats').transpose() # Pivotar para fácil acceso

    stats_a_num_pivoted = clean_stats_df(stats_a)
    stats_b_num_pivoted = clean_stats_df(stats_b)
    
    stats_to_plot_pct = ['Win %', 'BTTS', 'CS', 'FTS']
    stats_to_plot_avg = ['AVG', 'Scored', 'Conceded', 'xG', 'xGA']
    
    # Crear DFs para los nuevos gráficos
    df_stats_a_pct = stats_a_num_pivoted[stats_to_plot_pct].transpose().reset_index().rename(columns={'index':'Stats'})
    df_stats_a_avg = stats_a_num_pivoted[stats_to_plot_avg].transpose().reset_index().rename(columns={'index':'Stats'})
    df_stats_b_pct = stats_b_num_pivoted[stats_to_plot_pct].transpose().reset_index().rename(columns={'index':'Stats'})
    df_stats_b_avg = stats_b_num_pivoted[stats_to_plot_avg].transpose().reset_index().rename(columns={'index':'Stats'})

    # Generar los 4 nuevos gráficos de estadísticas
    fig_stats_a_pct = create_compact_stats_fig(df_stats_a_pct, ['Overall', 'Home', 'Away'], f'{team_a_name} - Estadísticas (%)', 'Porcentaje (%)', y_range=[0,100])
    fig_stats_a_avg = create_compact_stats_fig(df_stats_a_avg, ['Overall', 'Home', 'Away'], f'{team_a_name} - Promedios (Goles)', 'Promedio')
    
    fig_stats_b_pct = create_compact_stats_fig(df_stats_b_pct, ['Overall', 'Home', 'Away'], f'{team_b_name} - Estadísticas (%)', 'Porcentaje (%)', y_range=[0,100])
    fig_stats_b_avg = create_compact_stats_fig(df_stats_b_avg, ['Overall', 'Home', 'Away'], f'{team_b_name} - Promedios (Goles)', 'Promedio')

    # --- 4. Crear Tablas (Formateo original) ---
    def fmt_pct(x):
        return 'N/A' if pd.isna(x) else f"{round(x)}%"
    def fmt_avg(x):
        return 'N/A' if pd.isna(x) else f"{x:.2f}"

    scoring_scored_df = pd.DataFrame({
        'Metric': ['Goals per Match', 'Over 0.5 %', 'Over 1.5 %', 'Over 2.5 %', 'Failed To Score %',
                   'Scored in 1H %', 'Scored in 2H %', 'Avg 1H', 'Avg 2H'],
        f'{team_a_name} (Home)': [fmt_avg(metrics_a_home['goals_per_match']), fmt_pct(metrics_a_home['over_0_5_pct']),
                                  fmt_pct(metrics_a_home['over_1_5_pct']), fmt_pct(metrics_a_home['over_2_5_pct']),
                                  fmt_pct(metrics_a_home['fts_pct']), fmt_pct(metrics_a_home['scored_1h_pct']), 
                                  fmt_pct(metrics_a_home['scored_2h_pct']), fmt_avg(metrics_a_home['avg_1h']),
                                  fmt_avg(metrics_a_home['avg_2h'])],
        f'{team_b_name} (Away)': [fmt_avg(metrics_b_away['goals_per_match']), fmt_pct(metrics_b_away['over_0_5_pct']),
                                  fmt_pct(metrics_b_away['over_1_5_pct']), fmt_pct(metrics_b_away['over_2_5_pct']),
                                  fmt_pct(metrics_b_away['fts_pct']), fmt_pct(metrics_b_away['scored_1h_pct']), 
                                  fmt_pct(metrics_b_away['scored_2h_pct']), fmt_avg(metrics_b_away['avg_1h']),
                                  fmt_avg(metrics_b_away['avg_2h'])]
    })

    scoring_conceded_df = pd.DataFrame({
        'Metric': ['Conceded per Match', 'Con Over 0.5 %', 'Con Over 1.5 %', 'Clean Sheets %',
                   '1H Clean Sheet %', '2H Clean Sheet %', 'Avg Con 1H', 'Avg Con 2H'],
        f'{team_a_name} (Home)': [fmt_avg(metrics_a_home['conceded_per_match']), fmt_pct(metrics_a_home['con_over_0_5_pct']),
                                  fmt_pct(metrics_a_home['con_over_1_5_pct']), fmt_pct(metrics_a_home['clean_sheets_pct']),
                                  fmt_pct(metrics_a_home['clean_1h_pct']), fmt_pct(metrics_a_home['clean_2h_pct']),
                                  fmt_avg(metrics_a_home['avg_con_1h']), fmt_avg(metrics_a_home['avg_con_2h'])],
        f'{team_b_name} (Away)': [fmt_avg(metrics_b_away['conceded_per_match']), fmt_pct(metrics_b_away['con_over_0_5_pct']),
                                  fmt_pct(metrics_b_away['con_over_1_5_pct']), fmt_pct(metrics_b_away['clean_sheets_pct']),
                                  fmt_pct(metrics_b_away['clean_1h_pct']), fmt_pct(metrics_b_away['clean_2h_pct']),
                                  fmt_avg(metrics_b_away['avg_con_1h']), fmt_avg(metrics_b_away['avg_con_2h'])]
    })
    
    # --- 5. Ensamblar el Layout Final (EL NUEVO GRID) ---
    
    output_layout = html.Div([
        html.H2(f"Análisis: {team_a_name} (Local) vs. {team_b_name} (Visitante)", style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '20px'}),
        
        # Contenedor del grid principal
        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '15px', 'justifyContent': 'center'}, children=[
            
            # --- Fila 1: H2H y Comparativa ---
            html.Div([
                html.H3("Análisis Head-to-Head (H2H)", style={'textAlign': 'center', 'color': colors['primary'], 'borderBottom': f'1px solid {colors["grid"]}', 'paddingBottom': '5px'}),
                html.Div(dcc.Graph(figure=fig_h2h_wins), style=style_graph_container),
                html.Div(dcc.Graph(figure=fig_h2h_metrics), style={**style_graph_container, 'marginTop': '15px'}),
                html.H4("Detalle H2H (Últ. 8)", style={'marginTop': '15px', 'textAlign': 'center'}),
                dash_table.DataTable(
                    h2h_df.to_dict('records'), [{"name": i, "id": i} for i in h2h_df.columns],
                    **table_style
                ),
            ], style={'flexBasis': '48%', 'minWidth': '400px', 'backgroundColor': colors['card'], 'padding': '15px', 'borderRadius': '5px'}),
            
            html.Div([
                html.H3("Comparativa de Goles (Local vs. Vis.)", style={'textAlign': 'center', 'color': colors['primary'], 'borderBottom': f'1px solid {colors["grid"]}', 'paddingBottom': '5px'}),
                html.Div(dcc.Graph(figure=fig_scoring), style=style_graph_container),
                html.Div(dcc.Graph(figure=fig_conceded), style={**style_graph_container, 'marginTop': '15px'}),
                html.H4("Detalle Goles Anotados", style={'marginTop': '15px', 'textAlign': 'center'}),
                dash_table.DataTable(
                    scoring_scored_df.to_dict('records'), [{"name": i, "id": i} for i in scoring_scored_df.columns],
                    **table_style
                ),
                html.H4("Detalle Goles Encajados", style={'marginTop': '15px', 'textAlign': 'center'}),
                dash_table.DataTable(
                    scoring_conceded_df.to_dict('records'), [{"name": i, "id": i} for i in scoring_conceded_df.columns],
                    **table_style
                ),
            ], style={'flexBasis': '48%', 'minWidth': '400px', 'backgroundColor': colors['card'], 'padding': '15px', 'borderRadius': '5px'}),
            
            # --- Fila 2: Análisis por Equipo ---
            html.Div([
                html.H3(f"Análisis Detallado - {team_a_name} (Local)", style={'textAlign': 'center', 'color': colors['primary'], 'borderBottom': f'1px solid {colors["grid"]}', 'paddingBottom': '5px'}),
                html.Div(dcc.Graph(figure=fig_stats_a_pct), style=style_graph_container),
                html.Div(dcc.Graph(figure=fig_stats_a_avg), style={**style_graph_container, 'marginTop': '15px'}),
                html.H4("Forma y PPG", style={'marginTop': '15px', 'textAlign': 'center'}),
                dash_table.DataTable(form_a.to_dict('records'), [{"name": i, "id": i} for i in form_a.columns], **table_style),
                html.H4("Estadísticas", style={'marginTop': '15px', 'textAlign': 'center'}),
                dash_table.DataTable(stats_a.to_dict('records'), [{"name": i, "id": i} for i in stats_a.columns], **table_style)
            ], style={'flexBasis': '48%', 'minWidth': '400px', 'backgroundColor': colors['card'], 'padding': '15px', 'borderRadius': '5px'}),

            html.Div([
                html.H3(f"Análisis Detallado - {team_b_name} (Visitante)", style={'textAlign': 'center', 'color': colors['secondary'], 'borderBottom': f'1px solid {colors["grid"]}', 'paddingBottom': '5px'}),
                html.Div(dcc.Graph(figure=fig_stats_b_pct), style=style_graph_container),
                html.Div(dcc.Graph(figure=fig_stats_b_avg), style={**style_graph_container, 'marginTop': '15px'}),
                html.H4("Forma y PPG", style={'marginTop': '15px', 'textAlign': 'center'}),
                dash_table.DataTable(form_b.to_dict('records'), [{"name": i, "id": i} for i in form_b.columns], **table_style),
                html.H4("Estadísticas (Texto)", style={'marginTop': '15px', 'textAlign': 'center'}),
                dash_table.DataTable(stats_b.to_dict('records'), [{"name": i, "id": i} for i in stats_b.columns], **table_style)
            ], style={'flexBasis': '48%', 'minWidth': '400px', 'backgroundColor': colors['card'], 'padding': '15px', 'borderRadius': '5px'}),

        ]),
        
        # --- DF Unificado (Oculto) ---
        html.Div(id='unified-df', children=unified_df.to_json(), style={'display': 'none'})
    ])

    return output_layout

if __name__ == '__main__':
    app.run(debug=True)