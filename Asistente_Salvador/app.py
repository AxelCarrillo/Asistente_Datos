from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import requests
from io import StringIO
import os
from werkzeug.utils import secure_filename
import logging
import re
import numpy as np
from datetime import datetime


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Configuración desde variables de entorno (más seguro)
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Variables globales
df = None
df_info = {}
df_analysis = {}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    global df, df_info, df_analysis
    
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'success': False, 'message': 'No se subió archivo'})
        
        # Validar que sea un archivo CSV
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'message': 'El archivo debe ser un CSV'})
        
        # Leer el contenido del archivo
        content = file.stream.read().decode('utf-8-sig')  # utf-8-sig para manejar BOM
        
        # Intentar diferentes separadores
        separators = [',', ';', '\t', '|']
        df = None
        
        for sep in separators:
            try:
                temp_df = pd.read_csv(StringIO(content), separator=sep)
                if len(temp_df.columns) > 1:  # Si tiene más de una columna, probablemente es el separador correcto
                    df = temp_df
                    break
            except:
                continue
        
        if df is None:
            df = pd.read_csv(StringIO(content))  # Intento final con separador por defecto
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()
        
        # Análisis inteligente del DataFrame
        df_analysis = analyze_dataframe(df)
        
        # Guardar información del DataFrame
        df_info = {
            'columns': df.columns.tolist(),
            'rows': len(df),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'analysis': df_analysis
        }
        
        logger.info(f"CSV cargado exitosamente: {len(df)} filas, {len(df.columns)} columnas")
        logger.info(f"Tipo de datos detectado: {df_analysis.get('data_type', 'genérico')}")
        
        return jsonify({
            'success': True, 
            'message': 'CSV cargado correctamente',
            'columns': df_info['columns'],
            'rows': df_info['rows'],
            'data_type': df_analysis.get('data_type', 'genérico'),
            'preview': df.head(3).to_dict(orient='records')
        })
        
    except UnicodeDecodeError:
        return jsonify({'success': False, 'message': 'Error de codificación. Asegúrate de que el archivo esté en UTF-8'})
    except pd.errors.EmptyDataError:
        return jsonify({'success': False, 'message': 'El archivo CSV está vacío'})
    except Exception as e:
        logger.error(f"Error cargando CSV: {str(e)}")
        return jsonify({'success': False, 'message': f'Error leyendo CSV: {str(e)}'})

def analyze_dataframe(df):
    """Analiza el DataFrame para entender qué tipo de datos contiene"""
    analysis = {
        'data_type': 'genérico',
        'numeric_columns': [],
        'categorical_columns': [],
        'date_columns': [],
        'text_columns': [],
        'possible_id_columns': [],
        'key_insights': []
    }
    
    # Analizar tipos de columnas
    for column in df.columns:
        col_data = df[column].dropna()
        if len(col_data) == 0:
            continue
            
        # Detectar columnas numéricas
        if pd.api.types.is_numeric_dtype(df[column]):
            analysis['numeric_columns'].append(column)
            
        # Detectar fechas
        elif any(keyword in column.lower() for keyword in ['date', 'fecha', 'time', 'año', 'year', 'mes', 'month']):
            analysis['date_columns'].append(column)
            
        # Detectar posibles IDs
        elif any(keyword in column.lower() for keyword in ['id', 'codigo', 'code', 'key']) or df[column].nunique() == len(df):
            analysis['possible_id_columns'].append(column)
            
        # Detectar columnas categóricas (pocas categorías únicas)
        elif df[column].nunique() < len(df) * 0.5 and df[column].nunique() < 50:
            analysis['categorical_columns'].append(column)
            
        # El resto son columnas de texto
        else:
            analysis['text_columns'].append(column)
    
    # Detectar tipo de datos basado en nombres de columnas
    column_names = [col.lower() for col in df.columns]
    
    # Detectar diferentes tipos de datasets
    if any(keyword in ' '.join(column_names) for keyword in ['movie', 'film', 'title', 'genre', 'director', 'actor', 'pelicula', 'género']):
        analysis['data_type'] = 'películas/entretenimiento'
    elif any(keyword in ' '.join(column_names) for keyword in ['product', 'price', 'cost', 'sales', 'producto', 'precio', 'venta']):
        analysis['data_type'] = 'productos/ventas'
    elif any(keyword in ' '.join(column_names) for keyword in ['student', 'grade', 'score', 'estudiante', 'calificacion', 'nota']):
        analysis['data_type'] = 'educación/calificaciones'
    elif any(keyword in ' '.join(column_names) for keyword in ['employee', 'salary', 'department', 'empleado', 'salario', 'departamento']):
        analysis['data_type'] = 'recursos humanos'
    elif any(keyword in ' '.join(column_names) for keyword in ['customer', 'client', 'cliente', 'age', 'gender', 'edad', 'género']):
        analysis['data_type'] = 'clientes/demografía'
    elif any(keyword in ' '.join(column_names) for keyword in ['stock', 'inventory', 'quantity', 'inventario', 'cantidad']):
        analysis['data_type'] = 'inventario/stock'
    
    # Generar insights clave
    analysis['key_insights'] = generate_key_insights(df, analysis)
    
    return analysis

def generate_key_insights(df, analysis):
    """Genera insights clave sobre el dataset"""
    insights = []
    
    # Información básica
    insights.append(f"Dataset con {len(df)} registros y {len(df.columns)} columnas")
    
    # Columnas numéricas
    if analysis['numeric_columns']:
        insights.append(f"Columnas numéricas: {', '.join(analysis['numeric_columns'])}")
    
    # Columnas categóricas principales
    if analysis['categorical_columns']:
        cat_info = []
        for col in analysis['categorical_columns'][:3]:  # Solo las primeras 3
            unique_count = df[col].nunique()
            cat_info.append(f"{col} ({unique_count} categorías)")
        insights.append(f"Principales columnas categóricas: {', '.join(cat_info)}")
    
    # Detectar valores faltantes
    missing_cols = df.columns[df.isnull().sum() > 0].tolist()
    if missing_cols:
        insights.append(f"Columnas con datos faltantes: {', '.join(missing_cols[:3])}")
    
    return insights

@app.route('/ask', methods=['POST'])
def ask():
    global df, df_info, df_analysis
    
    if df is None:
        return jsonify({'success': False, 'answer': 'No hay CSV cargado'})

    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'success': False, 'answer': 'Pregunta no válida'})
        
        user_question = data.get('question', '').strip()
        if not user_question:
            return jsonify({'success': False, 'answer': 'La pregunta no puede estar vacía'})
        
        # Procesar la pregunta
        answer_html = process_question(user_question.lower(), user_question)
        
        return jsonify({'success': True, 'answer': answer_html})
        
    except Exception as e:
        logger.error(f"Error procesando pregunta: {str(e)}")
        return jsonify({'success': False, 'answer': f'Error procesando la pregunta: {str(e)}'})

def process_question(user_question_lower, original_question):
    """Procesa la pregunta del usuario de manera genérica"""
    global df, df_analysis
    
    # CONSULTAS DE RANGO DE FILAS - CORREGIDO
    row_range_result = handle_row_range_query(user_question_lower, original_question)
    if row_range_result:  # Si la función devuelve algo (no False), usar esa respuesta
        return row_range_result
    
    # CONSULTAS BÁSICAS GENÉRICAS
    if any(keyword in user_question_lower for keyword in ["primeras", "primeros", "top", "head", "inicio"]):
        numbers = re.findall(r'\d+', user_question_lower)
        n = int(numbers[0]) if numbers else 5
        return f"<p><strong>📊 Mostrando las primeras {n} filas:</strong></p>" + df.head(n).to_html(index=False, classes='tabla-resultados', escape=False)
    
    elif any(keyword in user_question_lower for keyword in ["últimas", "ultimas", "últimos", "ultimos", "tail", "final"]):
        numbers = re.findall(r'\d+', user_question_lower)
        n = int(numbers[0]) if numbers else 5
        return f"<p><strong>📊 Mostrando las últimas {n} filas:</strong></p>" + df.tail(n).to_html(index=False, classes='tabla-resultados', escape=False)
    
    # MANEJO DE COLUMNAS ESPECÍFICAS
    elif any(keyword in user_question_lower for keyword in ["columna", "column"]):
        return handle_column_request(user_question_lower)
    
    elif any(keyword in user_question_lower for keyword in ["columnas", "columns", "campos", "fields"]):
        return show_columns_info()
    
    # INFORMACIÓN Y RESUMEN
    elif any(keyword in user_question_lower for keyword in ["resumen", "info", "información", "describe", "summary"]):
        return generate_summary()
    
    # ESTADÍSTICAS
    elif any(keyword in user_question_lower for keyword in ["estadisticas", "estadísticas", "stats", "statistics"]):
        return generate_statistics()
    
    # VALORES ÚNICOS
    elif any(keyword in user_question_lower for keyword in ["únicos", "unicos", "unique", "distinct", "diferentes"]):
        return handle_unique_values(user_question_lower)
    
    # FILTROS GENÉRICOS
    elif any(keyword in user_question_lower for keyword in ["filtrar", "filter", "donde", "where", "buscar", "search"]):
        return handle_generic_filter(original_question)
    
    # CONTEOS
    elif any(keyword in user_question_lower for keyword in ["contar", "count", "cuántos", "cuantos", "cantidad"]):
        return handle_count_query(original_question)
    
    # FILTRO ESPECIAL: jugadores que patean con derecha
    elif any(keyword in user_question_lower for keyword in ["patean con derecha", "pie derecho", "foot right", "derecha"]):
        if 'Foot' in df.columns:
            filtered_df = df[df['Foot'].str.lower() == 'right']
            if filtered_df.empty:
                return "<p>No se encontraron jugadores que pateen con la derecha.</p>"
            return filtered_df.to_html(index=False, classes='tabla-resultados', escape=False)
        else:
            return "<p>No se encontró una columna 'Foot' en el dataset.</p>"
        
    # FILTRO ESPECIAL: jugadores que patean con izquierda
    elif any(keyword in user_question_lower for keyword in ["patean con izquierda", "pie izquierdo", "foot left", "izquierda"]):
        if 'Foot' in df.columns:
            filtered_df = df[df['Foot'].str.lower() == 'left']
            if filtered_df.empty:
                return "<p>No se encontraron jugadores que pateen con la izquierda.</p>"
            return filtered_df.to_html(index=False, classes='tabla-resultados', escape=False)
        else:
            return "<p>No se encontró una columna 'Foot' en el dataset.</p>"
        
    # FILTRO ESPECIAL: jugadores que patean con izquierda
    elif any(keyword in user_question_lower for keyword in ["jueguen en FC Barcelona", "equipo FC Barcelona", "club FC Barcelona", "FC Barcelona"]):
        if 'Club' in df.columns:
            filtered_df = df[df['Club'].str.lower() == 'FC Barcelona']
            if filtered_df.empty:
                return "<p>No se encontraron jugadores que jueguen en FC Barcelona</p>"
            return filtered_df.to_html(index=False, classes='tabla-resultados', escape=False)
        else:
            return "<p>No se encontró una columna 'Club' en el dataset.</p>"

    
    # Si no es una consulta específica, usar IA
    return ask_ai_generic(original_question)


def handle_row_range_query(user_question_lower, original_question):
    """Maneja consultas de rango de filas de manera inteligente"""
    global df
    
    # Patrones para detectar consultas de rango - MEJORADOS
    range_patterns = [
        r'(?:fila|row)\s*(\d+)\s*(?:a|al|to|-)\s*(?:fila|row)?\s*(\d+)',
        r'(?:filas|rows)\s*(\d+)\s*(?:a|al|to|-)\s*(\d+)',
        r'(?:del|from)\s*(\d+)\s*(?:a|al|to|-)\s*(\d+)',
        r'(?:desde|from)\s*(?:fila|row)?\s*(\d+)\s*(?:hasta|to)\s*(?:fila|row)?\s*(\d+)',
        r'(?:muestra|show|mostrar|dame)\s*(?:de|from|los)?\s*(?:la|the|datos)?\s*(?:fila|row)?\s*(\d+)\s*(?:a|al|to|-)\s*(?:la|the)?\s*(?:fila|row)?\s*(\d+)',
        r'(?:entre|between)\s*(?:fila|row)?\s*(\d+)\s*(?:y|and)\s*(?:fila|row)?\s*(\d+)',
        r'(?:datos|data|registros)\s*(?:de|from|del)?\s*(?:la|the)?\s*(?:fila|row)\s*(\d+)\s*(?:a|al|to|-)\s*(?:la|the)?\s*(?:fila|row)?\s*(\d+)'
    ]
    
    for pattern in range_patterns:
        match = re.search(pattern, user_question_lower)
        if match:
            start_row = int(match.group(1))
            end_row = int(match.group(2))
            
            # Validar el rango
            if start_row > end_row:
                start_row, end_row = end_row, start_row
            
            total_rows = len(df)
            
            # Verificar si las filas están en el rango válido
            if start_row < 1 or end_row > total_rows:
                return f"""
                <div class="status-message status-error">
                    <p><strong>❌ Rango de filas inválido</strong></p>
                    <p>Solicitaste filas {start_row} a {end_row}, pero el dataset solo tiene {total_rows:,} filas.</p>
                    <p>Rango válido: 1 a {total_rows:,}</p>
                </div>
                """
            
            # Ajustar índices (pandas usa indexación basada en 0)
            start_idx = start_row - 1
            end_idx = end_row
            
            # Obtener el subconjunto de datos
            subset_df = df.iloc[start_idx:end_idx]
            
            # Crear la respuesta HTML
            num_rows_shown = len(subset_df)
            
            response_html = f"""
            <div class="query-result">
                <p><strong>📊 Mostrando filas {start_row} a {end_row} ({num_rows_shown:,} registros)</strong></p>
                <div class="table-info">
                    <p><strong>Rango solicitado:</strong> Fila {start_row} a {end_row}</p>
                    <p><strong>Total del dataset:</strong> {total_rows:,} filas</p>
                </div>
                {subset_df.to_html(index=True, classes='tabla-resultados', escape=False)}
            </div>
            
            <style>
            .query-result {{
                margin: 10px 0;
            }}
            .table-info {{
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                border-left: 4px solid #007bff;
            }}
            .tabla-resultados {{
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
            }}
            .tabla-resultados th, .tabla-resultados td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .tabla-resultados th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            .tabla-resultados tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .status-message {{
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }}
            .status-error {{
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }}
            </style>
            """
            
            return response_html
    
    # Si no encuentra ningún patrón, devuelve False
    return False

def handle_column_request(user_question):
    """Maneja solicitudes específicas de columnas de manera genérica"""
    global df
    
    column_numbers = re.findall(r'columna\s*(\d+)', user_question)
    
    if not column_numbers:
        return show_columns_info()
    
    requested_indices = [int(num) - 1 for num in column_numbers]
    available_columns = len(df.columns)
    
    valid_indices = [idx for idx in requested_indices if 0 <= idx < available_columns]
    invalid_indices = [idx + 1 for idx in requested_indices if idx < 0 or idx >= available_columns]
    
    if not valid_indices:
        return f"""
        <div class="status-message status-error">
            <p><strong>❌ Columnas solicitadas no existen</strong></p>
            <p>Pediste: columna(s) {', '.join(column_numbers)}</p>
            <p>Pero tu CSV solo tiene <strong>{available_columns} columnas</strong></p>
        </div>
        {show_columns_info()}
        """
    
    selected_columns = [df.columns[idx] for idx in valid_indices]
    result_df = df[selected_columns]
    
    response_html = f"<p><strong>📊 Mostrando columna(s) solicitada(s):</strong></p>"
    
    if invalid_indices:
        response_html += f"""
        <div class="status-message status-info">
            <p><strong>ℹ️ Nota:</strong> Las columnas {', '.join(map(str, invalid_indices))} no existen en tu CSV.</p>
        </div>
        """
    
    response_html += f"""
    <p><strong>Columnas mostradas:</strong> {', '.join([f'Columna {valid_indices[i]+1} ({col})' for i, col in enumerate(selected_columns)])}</p>
    {result_df.to_html(index=False, classes='tabla-resultados', escape=False)}
    """
    
    return response_html

def show_columns_info():
    """Muestra información detallada de las columnas"""
    global df, df_analysis
    
    html = "<p><strong>📋 Información de las columnas:</strong></p><ul>"
    
    for i, col in enumerate(df.columns):
        col_type = "texto"
        if col in df_analysis['numeric_columns']:
            col_type = "numérica"
        elif col in df_analysis['categorical_columns']:
            col_type = "categórica"
        elif col in df_analysis['date_columns']:
            col_type = "fecha"
        
        unique_values = df[col].nunique()
        html += f'<li><strong>Columna {i+1}:</strong> {col} ({col_type}, {unique_values} valores únicos)</li>'
    
    html += "</ul>"
    return html

def generate_summary():
    """Genera un resumen inteligente del DataFrame"""
    global df, df_info, df_analysis
    
    summary_html = f"""
    <div>
        <h3>📋 Resumen del Dataset</h3>
        <p><strong>Tipo de datos detectado:</strong> {df_analysis.get('data_type', 'genérico')}</p>
        <p><strong>Filas:</strong> {df_info['rows']:,}</p>
        <p><strong>Columnas:</strong> {len(df_info['columns'])}</p>
        
        <h4>🔍 Insights clave:</h4>
        <ul>
            {"".join([f"<li>{insight}</li>" for insight in df_analysis.get('key_insights', [])])}
        </ul>
        
        <h4>💡 Consultas útiles que puedes probar:</h4>
        <ul>
            <li><strong>Rango de filas:</strong> "muestra de la fila 10 a la 20"</li>
            <li><strong>Primeras filas:</strong> "primeras 5 filas"</li>
            <li><strong>Estadísticas:</strong> "estadísticas del dataset"</li>
            <li><strong>Columnas específicas:</strong> "columna 1 y 3"</li>
        </ul>
        
        <h4>📊 Vista previa (primeras 3 filas):</h4>
        {df.head(3).to_html(index=False, classes='tabla-resultados', escape=False)}
    </div>
    """
    
    return summary_html

def generate_statistics():
    """Genera estadísticas del DataFrame"""
    global df, df_analysis
    
    html = "<h3>📈 Estadísticas del Dataset</h3>"
    
    # Estadísticas numéricas
    if df_analysis['numeric_columns']:
        html += "<h4>Columnas Numéricas:</h4>"
        numeric_stats = df[df_analysis['numeric_columns']].describe()
        html += numeric_stats.to_html(classes='tabla-resultados', escape=False)
    
    # Estadísticas categóricas
    if df_analysis['categorical_columns']:
        html += "<h4>Columnas Categóricas (valores más frecuentes):</h4>"
        for col in df_analysis['categorical_columns'][:3]:  # Solo las primeras 3
            top_values = df[col].value_counts().head(5)
            html += f"<p><strong>{col}:</strong></p>"
            html += top_values.to_frame().to_html(classes='tabla-resultados', escape=False)
    
    return html

def handle_unique_values(user_question):
    """Maneja consultas sobre valores únicos"""
    global df
    
    # Buscar si mencionan una columna específica
    for col in df.columns:
        if col.lower() in user_question:
            unique_vals = df[col].unique()
            html = f"<p><strong>Valores únicos en '{col}':</strong></p>"
            if len(unique_vals) <= 20:
                html += f"<ul>{''.join([f'<li>{val}</li>' for val in unique_vals if pd.notna(val)])}</ul>"
            else:
                html += f"<p>Total de valores únicos: {len(unique_vals)}</p>"
                html += f"<p>Primeros 20 valores:</p>"
                html += f"<ul>{''.join([f'<li>{val}</li>' for val in unique_vals[:20] if pd.notna(val)])}</ul>"
            return html
    
    # Si no especifica columna, mostrar resumen de todas
    html = "<p><strong>Valores únicos por columna:</strong></p><ul>"
    for col in df.columns:
        unique_count = df[col].nunique()
        html += f"<li><strong>{col}:</strong> {unique_count} valores únicos</li>"
    html += "</ul>"
    
    return html

def handle_generic_filter(original_question):
    """Maneja filtros genéricos usando IA"""
    return ask_ai_generic(f"El usuario quiere filtrar datos con esta consulta: {original_question}")

def handle_count_query(original_question):
    """Maneja consultas de conteo"""
    global df
    
    # Contar registros totales
    if any(keyword in original_question.lower() for keyword in ["total", "registros", "filas", "rows"]):
        return f"<p><strong>Total de registros:</strong> {len(df):,}</p>"
    
    # Usar IA para consultas más complejas
    return ask_ai_generic(f"El usuario quiere contar algo específico: {original_question}")

def ask_ai_generic(user_question):
    """Consulta a la IA de manera genérica e inteligente"""
    global df, df_analysis
    
    # Preparar una muestra más inteligente
    sample_size = min(5, len(df))
    sample_rows = df.head(sample_size).to_dict(orient="records")
    
    # Crear contexto inteligente basado en el análisis
    context = f"""
Eres un asistente experto en análisis de datos. Tienes acceso a un dataset CSV con la siguiente información:

INFORMACIÓN DEL DATASET:
- Tipo de datos: {df_analysis.get('data_type', 'genérico')}
- Total de filas: {len(df):,}
- Total de columnas: {len(df.columns)}

ANÁLISIS DE COLUMNAS:
- Columnas numéricas: {df_analysis.get('numeric_columns', [])}
- Columnas categóricas: {df_analysis.get('categorical_columns', [])}
- Columnas de fecha: {df_analysis.get('date_columns', [])}
- Columnas de texto: {df_analysis.get('text_columns', [])}

COLUMNAS DISPONIBLES:
{chr(10).join([f'{i+1}. {col}' for i, col in enumerate(df.columns)])}

INSIGHTS CLAVE:
{chr(10).join([f'- {insight}' for insight in df_analysis.get('key_insights', [])])}

MUESTRA DE DATOS (primeras {sample_size} filas):
{sample_rows}

PREGUNTA DEL USUARIO: "{user_question}"

INSTRUCCIONES:
1. Responde basándote ÚNICAMENTE en los datos del CSV proporcionado
2. Si necesitas mostrar datos tabulares, usa formato HTML con <table>, <tr>, <th>, <td> y class="tabla-resultados"
3. Sé específico y usa los nombres reales de las columnas
4. Si la pregunta no puede responderse con los datos disponibles, explícalo claramente
5. Proporciona insights útiles y relevantes
6. Si hay números, usa formato con comas para miles (ej: 1,000)
7. Mantén las respuestas concisas
8. Si el usuario pide rangos específicos de filas, sugiere usar consultas como "muestra de la fila X a la Y"
9. NO muestres la consulta ni expliques cómo obtuviste los datos, solo presenta los resultados directamente
10. NO uses frases como "Jugadores con más de 10 goles:" antes de mostrar la tabla, ve directo a los resultados
11. NUNCA mostrar código, sintaxis de plantillas, bucles for, if, ni elementos técnicos como '{{% %}}' o '{{{{ }}}}'
12. Solo muestra el resultado final en formato HTML limpio sin mostrar el proceso
13. No incluyas texto como "al analizar los datos" ni "encontré que...". Solo responde con los datos directamente, sin explicaciones.
14. Si hay más de 100 resultados, puedes mostrarlos todos en una tabla. No muestres "solo ejemplos", muestra todos los datos filtrados.
15. Si hay más de 15 resultados, puedes mostrarlos todos en una tabla. No muestres "solo ejemplos", muestra todos los datos filtrados.

"""

    return call_groq_api(context)

def call_groq_api(context, max_retries=3):
    """Llama a la API de Groq con reintentos y manejo robusto de errores"""
    import time

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Eres un asistente experto en análisis de datos. Responde de forma precisa, concisa y útil."},
            {"role": "user", "content": context}
        ],
        "max_tokens": 1000,
        "temperature": 0.1
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'choices' in data and len(data['choices']) > 0:
                raw_response = data['choices'][0]['message']['content']
                
                # 🧽 Limpieza: eliminar bloques como {{ ... }} o {% ... %}
                cleaned_response = re.sub(r"{[{%].*?[}%]}", "", raw_response, flags=re.DOTALL)

                return cleaned_response
            else:
                return '<p>No se recibió respuesta válida de la IA.</p>'

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503:
                logger.warning(f"API Groq no disponible (503). Intento {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Backoff exponencial: 1s, 2s, 4s
                    continue
                else:
                    return generate_fallback_response(context)
            elif e.response.status_code == 429:
                logger.warning(f"Límite de velocidad alcanzado (429). Intento {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    return '<p>⚠️ Límite de consultas alcanzado. Intenta nuevamente en unos minutos.</p>'
            else:
                logger.error(f"Error HTTP en API Groq: {e}")
                return f'<p>Error en la API: {e.response.status_code}</p>'

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout en API Groq. Intento {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                continue
            else:
                return '<p>⏱️ Tiempo de espera agotado. Intenta con una pregunta más específica.</p>'

        except requests.exceptions.RequestException as e:
            logger.error(f"Error de conexión con API Groq: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                return generate_fallback_response(context)

        except Exception as e:
            logger.error(f"Error inesperado en Groq: {str(e)}")
            return f'<p>Error inesperado: {str(e)}</p>'

    # Si llegamos aquí, todos los intentos fallaron
    return generate_fallback_response(context)

def generate_fallback_response(context):
    """Genera respuesta de respaldo cuando la IA no está disponible"""
    global df, df_analysis
    
    # Extraer la pregunta del contexto
    question_match = re.search(r'PREGUNTA DEL USUARIO: "([^"]*)"', context)
    user_question = question_match.group(1).lower() if question_match else ""
    
    logger.info(f"Generando respuesta de respaldo para: {user_question}")
    
    # Respuestas básicas sin IA
    if any(keyword in user_question for keyword in ["promedio", "average", "media"]):
        numeric_cols = df_analysis.get('numeric_columns', [])
        if numeric_cols:
            html = "<h4>📊 Promedios de columnas numéricas:</h4><ul>"
            for col in numeric_cols[:5]:  # Máximo 5 columnas
                avg_val = df[col].mean()
                html += f"<li><strong>{col}:</strong> {avg_val:,.2f}</li>"
            html += "</ul>"
            return html
    
    elif any(keyword in user_question for keyword in ["máximo", "max", "mayor"]):
        numeric_cols = df_analysis.get('numeric_columns', [])
        if numeric_cols:
            html = "<h4>📈 Valores máximos:</h4><ul>"
            for col in numeric_cols[:5]:
                max_val = df[col].max()
                html += f"<li><strong>{col}:</strong> {max_val:,}</li>"
            html += "</ul>"
            return html
    
    elif any(keyword in user_question for keyword in ["mínimo", "min", "menor"]):
        numeric_cols = df_analysis.get('numeric_columns', [])
        if numeric_cols:
            html = "<h4>📉 Valores mínimos:</h4><ul>"
            for col in numeric_cols[:5]:
                min_val = df[col].min()
                html += f"<li><strong>{col}:</strong> {min_val:,}</li>"
            html += "</ul>"
            return html
    
    elif any(keyword in user_question for keyword in ["nulos", "null", "faltantes", "missing"]):
        missing_data = df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        if len(missing_cols) > 0:
            html = "<h4>❓ Datos faltantes por columna:</h4><ul>"
            for col, count in missing_cols.items():
                percentage = (count / len(df)) * 100
                html += f"<li><strong>{col}:</strong> {count:,} valores faltantes ({percentage:.1f}%)</li>"
            html += "</ul>"
            return html
        else:
            return "<p>✅ No hay datos faltantes en el dataset.</p>"
    
    # Respuesta genérica cuando la IA no está disponible
    return f"""
    <div class="api-fallback">
        <h4>🔧 Servicio de IA temporalmente no disponible</h4>
        <p>La API de análisis inteligente está experimentando problemas. Mientras tanto, puedes usar estos comandos directos:</p>
        
        <div class="commands-help">
            <h5>📋 Comandos disponibles:</h5>
            <ul>
                <li><strong>"primeras 10"</strong> - Muestra las primeras filas</li>
                <li><strong>"de la fila 5 a la 15"</strong> - Muestra rango específico</li>
                <li><strong>"estadísticas"</strong> - Estadísticas generales</li>
                <li><strong>"columnas"</strong> - Lista todas las columnas</li>
                <li><strong>"resumen"</strong> - Resumen del dataset</li>
                <li><strong>"únicos"</strong> - Valores únicos</li>
            </ul>
        </div>
        
        <div class="dataset-summary">
            <h5>📊 Información actual del dataset:</h5>
            <ul>
                <li><strong>Filas:</strong> {len(df):,}</li>
                <li><strong>Columnas:</strong> {len(df.columns)}</li>
                <li><strong>Tipo de datos:</strong> {df_analysis.get('data_type', 'genérico')}</li>
            </ul>
        </div>
    </div>
    
    <style>
    .api-fallback {{
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }}
    .commands-help {{
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    .dataset-summary {{
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    </style>
    """

def check_api_health():
    """Verifica si la API de Groq está disponible"""
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        # Hacer una petición simple para verificar conectividad
        response = requests.get("https://api.groq.com/openai/v1/models", 
                              headers=headers, timeout=5)
        return response.status_code == 200
    except:
        return False
def too_large(e):
    return jsonify({'success': False, 'message': 'El archivo es demasiado grande'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'message': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)