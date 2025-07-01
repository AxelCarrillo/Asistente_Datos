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
import secrets
import hashlib
from functools import wraps
import time
from flask_cors import CORS

# Configurar logging seguro
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

app = Flask(__name__, static_folder='static', static_url_path='/static')

# 🔒 SEGURIDAD: Configuración segura de Flask
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 100MB límite de archivo
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# 🔒 SEGURIDAD: API Key desde variables de entorno ÚNICAMENTE
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY no está configurada en las variables de entorno")
    raise ValueError("GROQ_API_KEY es requerida")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# 🔒 SEGURIDAD: Rate limiting
REQUEST_LIMIT = 100  # Requests por minuto
request_counts = {}

def rate_limit_check(ip_address):
    """Implementa rate limiting básico"""
    current_time = time.time()
    minute_ago = current_time - 60
    
    # Limpiar registros antiguos
    if ip_address in request_counts:
        request_counts[ip_address] = [
            timestamp for timestamp in request_counts[ip_address] 
            if timestamp > minute_ago
        ]
    else:
        request_counts[ip_address] = []
    
    # Verificar límite
    if len(request_counts[ip_address]) >= REQUEST_LIMIT:
        return False
    
    # Agregar request actual
    request_counts[ip_address].append(current_time)
    return True

def require_rate_limit(f):
    """Decorador para aplicar rate limiting"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        if not rate_limit_check(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return jsonify({'success': False, 'message': 'Límite de requests excedido. Intenta en un minuto.'}), 429
        return f(*args, **kwargs)
    return decorated_function

# Variables globales
df = None
df_info = {}
df_analysis = {}

# 🔒 SEGURIDAD: Lista de extensiones permitidas
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_content(content):
    """Valida el contenido del archivo CSV de manera segura - MEJORADA"""
    # Verificar que no esté vacío
    if not content or len(content.strip()) == 0:
        return False, "El archivo está vacío"
    
    # Verificar tamaño
    if len(content.encode('utf-8')) > MAX_FILE_SIZE:
        return False, "El archivo es demasiado grande"
    
    # ✅ CORREGIDO: Verificar patrones maliciosos más específicos para CSV
    suspicious_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'<iframe[^>]*>.*?</iframe>',
        r'<?php.*?>',
        r'<%.*%>',
        r'data:text/html'
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            return False, f"Contenido no válido detectado"
    
    return True, "Válido"

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/upload_csv', methods=['POST'])
@require_rate_limit
def upload_csv():
    global df, df_info, df_analysis
    
    try:
        logger.info(f"Intento de subida desde IP: {request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)}")
        
        # 🔒 SEGURIDAD: Validar que existe el archivo
        if 'file' not in request.files:
            logger.warning("No se proporcionó archivo en request.files")
            return jsonify({'success': False, 'message': 'No se proporcionó archivo'}), 400
        
        file = request.files['file']
        logger.info(f"Archivo recibido: {file.filename}")
        
        # 🔒 SEGURIDAD: Validar nombre de archivo
        if file.filename == '':
            logger.warning("Nombre de archivo vacío")
            return jsonify({'success': False, 'message': 'No se seleccionó archivo'}), 400
        
        if not allowed_file(file.filename):
            logger.warning(f"Extensión no permitida: {file.filename}")
            return jsonify({'success': False, 'message': 'Solo se permiten archivos CSV'}), 400
        
        # 🔒 SEGURIDAD: Leer el contenido de manera segura - MEJORADO
        try:
            raw_content = file.stream.read()
            logger.info(f"Archivo leído: {len(raw_content)} bytes")
            
            if len(raw_content) > MAX_FILE_SIZE:
                logger.warning(f"Archivo demasiado grande: {len(raw_content)} bytes")
                return jsonify({'success': False, 'message': 'El archivo es demasiado grande (máximo 10MB)'}), 413
            
            # ✅ CORREGIDO: Intentar más codificaciones
            content = None
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    content = raw_content.decode(encoding)
                    logger.info(f"Archivo decodificado exitosamente con {encoding}")
                    break
                except UnicodeDecodeError as e:
                    logger.debug(f"Fallo decodificación con {encoding}: {str(e)}")
                    continue
            
            if content is None:
                logger.error("No se pudo decodificar el archivo")
                return jsonify({'success': False, 'message': 'No se pudo decodificar el archivo. Verifica que sea un CSV válido en UTF-8'}), 400
                
        except Exception as e:
            logger.error(f"Error leyendo archivo: {str(e)}")
            return jsonify({'success': False, 'message': f'Error leyendo el archivo: {str(e)}'}), 400
        
        # 🔒 SEGURIDAD: Validar contenido
        is_valid, validation_message = validate_file_content(content)
        if not is_valid:
            logger.warning(f"Archivo rechazado: {validation_message}")
            return jsonify({'success': False, 'message': validation_message}), 400
        
        logger.info("Contenido validado, procesando CSV...")
        
        # ✅ CORREGIDO: Procesar CSV con mejor manejo de errores
        separators = [',', ';', '\t', '|']
        df = None
        processing_errors = []
        
        for sep in separators:
            try:
                temp_df = pd.read_csv(
                    StringIO(content), 
                    sep=sep, 
                    nrows=50000,
                    dtype=str,
                    na_filter=False
                )
                if len(temp_df.columns) > 1:
                    df = temp_df
                    logger.info(f"CSV procesado con separador '{sep}': {len(df)} filas, {len(df.columns)} columnas")
                    break
            except Exception as e:
                processing_errors.append(f"Sep '{sep}': {str(e)[:50]}")
                logger.debug(f"Separador {sep} falló: {str(e)}")
                continue
        
        if df is None:
            try:
                df = pd.read_csv(
                    StringIO(content), 
                    nrows=50000,
                    dtype=str,
                    na_filter=False
                )
                logger.info(f"CSV procesado con separador por defecto: {len(df)} filas, {len(df.columns)} columnas")
            except Exception as e:
                logger.error(f"Error final procesando CSV: {str(e)}")
                return jsonify({
                    'success': False, 
                    'message': f'Formato CSV inválido. Verifica separadores y estructura.'
                }), 400
        
        # Verificar que el DataFrame no esté vacío
        if df.empty:
            logger.warning("DataFrame vacío después del procesamiento")
            return jsonify({'success': False, 'message': 'El archivo CSV está vacío o no tiene datos válidos'}), 400
        
        # 🔒 SEGURIDAD: Limpiar y validar nombres de columnas
        df.columns = [sanitize_column_name(col) for col in df.columns]
        
        # Análisis seguro del DataFrame
        df_analysis = analyze_dataframe_safe(df)
        
        # Guardar información del DataFrame
        df_info = {
            'columns': df.columns.tolist()[:100],
            'rows': len(df),
            'dtypes': {k: str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
            'analysis': df_analysis
        }
        
        logger.info(f"CSV procesado exitosamente: {len(df)} filas, {len(df.columns)} columnas")
        
        return jsonify({
            'success': True, 
            'message': 'CSV cargado correctamente',
            'columns': df_info['columns'],
            'rows': min(df_info['rows'], 50000),
            'data_type': df_analysis.get('data_type', 'genérico'),
            'preview': df.head(3).to_dict(orient='records')
        })
        
    except Exception as e:
        logger.error(f"Error inesperado en upload_csv: {str(e)}")
        return jsonify({'success': False, 'message': f'Error interno: {str(e)[:100]}'}), 500
        

def sanitize_column_name(col_name):
    """Sanitiza nombres de columnas para evitar inyecciones"""
    if not isinstance(col_name, str):
        col_name = str(col_name)
    
    # Remover caracteres peligrosos
    col_name = re.sub(r'[<>"\'/\\]', '', col_name)
    
    # Limitar longitud
    col_name = col_name[:100]
    
    # Limpiar espacios
    col_name = col_name.strip()
    
    # Si queda vacío, usar nombre por defecto
    if not col_name:
        col_name = "Column"
    
    return col_name

def analyze_dataframe_safe(df):
    """Analiza el DataFrame de manera segura"""
    analysis = {
        'data_type': 'genérico',
        'numeric_columns': [],
        'categorical_columns': [],
        'date_columns': [],
        'text_columns': [],
        'possible_id_columns': [],
        'key_insights': []
    }
    
    try:
        # Limitar análisis para evitar timeouts
        max_columns_to_analyze = 50
        columns_to_analyze = df.columns[:max_columns_to_analyze]
        
        for column in columns_to_analyze:
            try:
                col_data = df[column].dropna()
                if len(col_data) == 0:
                    continue
                
                # Análisis seguro de tipos de datos
                sample_size = min(1000, len(col_data))
                col_sample = col_data.head(sample_size)
                
                # Detectar columnas numéricas de manera segura
                try:
                    pd.to_numeric(col_sample, errors='coerce')
                    if not col_sample.isna().all():
                        analysis['numeric_columns'].append(column)
                        continue
                except:
                    pass
                
                # Detectar fechas de manera segura
                if any(keyword in column.lower() for keyword in ['date', 'fecha', 'time', 'año', 'year']):
                    analysis['date_columns'].append(column)
                    continue
                
                # Detectar posibles IDs
                unique_ratio = col_sample.nunique() / len(col_sample)
                if unique_ratio > 0.95 or any(keyword in column.lower() for keyword in ['id', 'codigo', 'code']):
                    analysis['possible_id_columns'].append(column)
                    continue
                
                # Detectar columnas categóricas
                if col_sample.nunique() < len(col_sample) * 0.5 and col_sample.nunique() < 50:
                    analysis['categorical_columns'].append(column)
                else:
                    analysis['text_columns'].append(column)
                    
            except Exception as e:
                logger.debug(f"Error analizando columna {column}: {str(e)}")
                continue
        
        # Detectar tipo de datos de manera segura
        analysis['data_type'] = detect_data_type_safe(df.columns.tolist()[:20])
        analysis['key_insights'] = generate_key_insights_safe(df, analysis)
        
    except Exception as e:
        logger.error(f"Error en análisis de DataFrame: {str(e)}")
    
    return analysis

def detect_data_type_safe(column_names):
    """Detecta el tipo de datos de manera segura"""
    try:
        column_text = ' '.join([str(col).lower() for col in column_names[:20]])
        
        type_patterns = {
            'películas/entretenimiento': ['movie', 'film', 'title', 'genre', 'director', 'pelicula'],
            'productos/ventas': ['product', 'price', 'cost', 'sales', 'producto', 'precio'],
            'educación/calificaciones': ['student', 'grade', 'score', 'estudiante', 'calificacion'],
            'recursos humanos': ['employee', 'salary', 'department', 'empleado', 'salario'],
            'clientes/demografía': ['customer', 'client', 'cliente', 'age', 'gender', 'edad'],
            'inventario/stock': ['stock', 'inventory', 'quantity', 'inventario', 'cantidad']
        }
        
        for data_type, keywords in type_patterns.items():
            if any(keyword in column_text for keyword in keywords):
                return data_type
                
    except Exception as e:
        logger.debug(f"Error detectando tipo de datos: {str(e)}")
    
    return 'genérico'

def generate_key_insights_safe(df, analysis):
    """Genera insights de manera segura"""
    insights = []
    
    try:
        insights.append(f"Dataset con {min(len(df), 50000):,} registros y {len(df.columns)} columnas")
        
        if analysis['numeric_columns']:
            num_cols = len(analysis['numeric_columns'])
            insights.append(f"{num_cols} columnas numéricas detectadas")
        
        if analysis['categorical_columns']:
            cat_cols = len(analysis['categorical_columns'])
            insights.append(f"{cat_cols} columnas categóricas detectadas")
        
        # Verificar datos faltantes de manera segura
        try:
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                insights.append(f"Se detectaron {missing_count:,} valores faltantes")
        except:
            pass
            
    except Exception as e:
        logger.debug(f"Error generando insights: {str(e)}")
        insights = ["Análisis básico completado"]
    
    return insights

@app.route('/ask', methods=['POST'])
@require_rate_limit
def ask():
    global df, df_info, df_analysis
    
    if df is None:
        return jsonify({'success': False, 'answer': 'No hay CSV cargado'}), 400

    try:
        # 🔒 SEGURIDAD: Validar entrada JSON
        if not request.is_json:
            return jsonify({'success': False, 'answer': 'Formato de datos inválido'}), 400
        
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'success': False, 'answer': 'Pregunta no proporcionada'}), 400
        
        user_question = data.get('question', '').strip()
        
        # 🔒 SEGURIDAD: Validar y sanitizar la pregunta
        if not user_question:
            return jsonify({'success': False, 'answer': 'La pregunta no puede estar vacía'}), 400
        
        if len(user_question) > 500:
            return jsonify({'success': False, 'answer': 'La pregunta es demasiado larga (máximo 500 caracteres)'}), 400
        
        # 🔒 SEGURIDAD: Detectar patrones maliciosos en la pregunta
        malicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'<iframe[^>]*>.*?</iframe>',
            r'eval\s*\(',
            r'exec\s*\(',
            r'import\s+os',
            r'import\s+subprocess',
            r'__import__',
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'UPDATE\s+.*\s+SET'
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, user_question, re.IGNORECASE):
                logger.warning(f"Pregunta maliciosa detectada: {user_question[:100]}")
                return jsonify({'success': False, 'answer': 'Pregunta no válida'}), 400
        
        # Procesar la pregunta de manera segura
        answer_html = process_question_safe(user_question.lower(), user_question)
        
        return jsonify({'success': True, 'answer': answer_html})
        
    except Exception as e:
        logger.error(f"Error procesando pregunta: {str(e)}")
        return jsonify({'success': False, 'answer': 'Error interno del servidor'}), 500

def process_question_safe(user_question_lower, original_question):
    """Procesa la pregunta de manera segura"""
    global df, df_analysis
    
    try:
        # CONSULTAS DE RANGO DE FILAS - SEGURO
        row_range_result = handle_row_range_query_safe(user_question_lower, original_question)
        if row_range_result:
            return row_range_result
        
        # CONSULTAS BÁSICAS SEGURAS
        if any(keyword in user_question_lower for keyword in ["primeras", "primeros", "top", "head"]):
            numbers = re.findall(r'\d+', user_question_lower)
            n = min(int(numbers[0]) if numbers else 5, 100)  # 🔒 Limitar a 100 filas
            return f"<p><strong>📊 Mostrando las primeras {n} filas:</strong></p>" + \
                   escape_html_output(df.head(n).to_html(index=False, classes='tabla-resultados', escape=True))
        
        elif any(keyword in user_question_lower for keyword in ["últimas", "ultimas", "tail"]):
            numbers = re.findall(r'\d+', user_question_lower)
            n = min(int(numbers[0]) if numbers else 5, 100)  # 🔒 Limitar a 100 filas
            return f"<p><strong>📊 Mostrando las últimas {n} filas:</strong></p>" + \
                   escape_html_output(df.tail(n).to_html(index=False, classes='tabla-resultados', escape=True))
        
        # INFORMACIÓN SEGURA
        elif any(keyword in user_question_lower for keyword in ["resumen", "info", "información"]):
            return generate_summary_safe()
        
        elif any(keyword in user_question_lower for keyword in ["columnas", "columns"]):
            return show_columns_info_safe()
        
        elif any(keyword in user_question_lower for keyword in ["estadisticas", "estadísticas"]):
            return generate_statistics_safe()
        
        # Si no es una consulta específica, usar IA de manera segura
        return ask_ai_safe(original_question)
        
    except Exception as e:
        logger.error(f"Error procesando pregunta: {str(e)}")
        return "<p>Error procesando la consulta</p>"

def handle_row_range_query_safe(user_question_lower, original_question):
    """Maneja consultas de rango de filas de manera segura"""
    global df
    
    try:
        range_patterns = [
            r'(?:fila|row)\s*(\d+)\s*(?:a|al|to|-)\s*(?:fila|row)?\s*(\d+)',
            r'(?:del|from)\s*(\d+)\s*(?:a|al|to|-)\s*(\d+)',
            r'(?:entre|between)\s*(?:fila|row)?\s*(\d+)\s*(?:y|and)\s*(?:fila|row)?\s*(\d+)'
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, user_question_lower)
            if match:
                start_row = int(match.group(1))
                end_row = int(match.group(2))
                
                # 🔒 SEGURIDAD: Validar y limitar rango
                if start_row > end_row:
                    start_row, end_row = end_row, start_row
                
                # Limitar rango máximo
                max_range = 1000
                if (end_row - start_row) > max_range:
                    return f"<p>❌ Rango demasiado grande. Máximo {max_range} filas por consulta.</p>"
                
                total_rows = len(df)
                
                if start_row < 1 or end_row > total_rows:
                    return f"<p>❌ Rango inválido. Dataset tiene {total_rows:,} filas.</p>"
                
                # Obtener datos de manera segura
                start_idx = start_row - 1
                end_idx = min(end_row, start_idx + max_range)
                subset_df = df.iloc[start_idx:end_idx]
                
                return f"<p><strong>📊 Filas {start_row} a {end_idx}:</strong></p>" + \
                       escape_html_output(subset_df.to_html(index=True, classes='tabla-resultados', escape=True))
        
    except Exception as e:
        logger.error(f"Error en rango de filas: {str(e)}")
        return "<p>Error procesando rango de filas</p>"
    
    return False

def generate_summary_safe():
    """Genera resumen de manera segura"""
    global df, df_info, df_analysis
    
    try:
        summary_html = f"""
        <div>
            <h3>📋 Resumen del Dataset</h3>
            <p><strong>Tipo:</strong> {escape_html_content(df_analysis.get('data_type', 'genérico'))}</p>
            <p><strong>Filas:</strong> {min(df_info['rows'], 50000):,}</p>
            <p><strong>Columnas:</strong> {len(df_info['columns'])}</p>
            
            <h4>🔍 Características:</h4>
            <ul>
                <li>Columnas numéricas: {len(df_analysis.get('numeric_columns', []))}</li>
                <li>Columnas categóricas: {len(df_analysis.get('categorical_columns', []))}</li>
                <li>Columnas de texto: {len(df_analysis.get('text_columns', []))}</li>
            </ul>
            
            <h4>📊 Vista previa:</h4>
            {escape_html_output(df.head(3).to_html(index=False, classes='tabla-resultados', escape=True))}
        </div>
        """
        return summary_html
    except Exception as e:
        logger.error(f"Error generando resumen: {str(e)}")
        return "<p>Error generando resumen</p>"

def show_columns_info_safe():
    """Muestra información de columnas de manera segura"""
    global df, df_analysis
    
    try:
        html = "<p><strong>📋 Columnas del dataset:</strong></p><ul>"
        
        max_columns_to_show = 50
        columns_to_show = df.columns[:max_columns_to_show]
        
        for i, col in enumerate(columns_to_show):
            safe_col_name = escape_html_content(str(col))
            unique_count = min(df[col].nunique(), 1000)  # Limitar cálculo
            html += f'<li><strong>Columna {i+1}:</strong> {safe_col_name} ({unique_count} valores únicos)</li>'
        
        if len(df.columns) > max_columns_to_show:
            html += f"<li><em>... y {len(df.columns) - max_columns_to_show} columnas más</em></li>"
            
        html += "</ul>"
        return html
    except Exception as e:
        logger.error(f"Error mostrando columnas: {str(e)}")
        return "<p>Error mostrando información de columnas</p>"

def generate_statistics_safe():
    """Genera estadísticas de manera segura"""
    global df, df_analysis
    
    try:
        html = "<h3>📈 Estadísticas del Dataset</h3>"
        
        # Solo estadísticas básicas para evitar timeouts
        numeric_cols = df_analysis.get('numeric_columns', [])[:10]  # Máximo 10 columnas
        
        if numeric_cols:
            try:
                # Convertir a numérico de manera segura
                numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                stats = numeric_df.describe()
                html += "<h4>Columnas Numéricas:</h4>"
                html += escape_html_output(stats.to_html(classes='tabla-resultados', escape=True))
            except Exception as e:
                logger.debug(f"Error calculando estadísticas numéricas: {str(e)}")
                html += "<p>No se pudieron calcular estadísticas numéricas</p>"
        else:
            html += "<p>No hay columnas numéricas para analizar</p>"
        
        return html
    except Exception as e:
        logger.error(f"Error generando estadísticas: {str(e)}")
        return "<p>Error generando estadísticas</p>"

def ask_ai_safe(user_question):
    """Consulta a la IA de manera segura"""
    global df, df_analysis
    
    try:
        # Verificar si la pregunta requiere mostrar datos tabulares
        needs_table = any(keyword in user_question.lower() for keyword in [
            "muestra", "show", "ver", "tabla", "datos", "registros", "filas", 
            "ejemplo", "sample", "display", "list", "primeros", "últimos"
        ])
        
        # 🔒 SEGURIDAD: Limitar muestra de datos enviada a la IA
        sample_size = min(5, len(df)) if needs_table else min(3, len(df))
        safe_sample = df.head(sample_size).to_dict(orient="records")
        
        # Sanitizar muestra antes de enviar
        sanitized_sample = []
        for record in safe_sample:
            sanitized_record = {}
            for key, value in record.items():
                safe_key = escape_html_content(str(key)[:50])
                safe_value = escape_html_content(str(value)[:100])
                sanitized_record[safe_key] = safe_value
            sanitized_sample.append(sanitized_record)
        
        # Crear contexto seguro con instrucciones mejoradas
        context = f"""
Analiza este dataset CSV y responde de forma conversacional y natural:

INFORMACIÓN DEL DATASET:
- Total de filas: {min(len(df), 50000):,}
- Total de columnas: {len(df.columns)}
- Tipo de datos: {escape_html_content(df_analysis.get('data_type', 'genérico'))}

COLUMNAS PRINCIPALES:
{chr(10).join([f'• {escape_html_content(str(col)[:50])}' for i, col in enumerate(df.columns[:20])])}

MUESTRA DE DATOS (primeras {sample_size} filas):
{sanitized_sample}

PREGUNTA DEL USUARIO: "{escape_html_content(user_question[:200])}"

INSTRUCCIONES IMPORTANTES:
- Responde como un analista de datos experimentado y amigable
- Usa un lenguaje natural y conversacional 
- Proporciona insights útiles y prácticos
- Si la pregunta pide ver datos, muestra una tabla HTML bien formateada
- Si la pregunta es analítica, responde en párrafos explicativos
- Explica qué significan los patrones que encuentres
- Evita mostrar código de programación
- Sé específico pero fácil de entender
- Si muestras una tabla, agrega una breve explicación de lo que se ve

FORMATO DE RESPUESTA:
- Para mostrar datos: Usa tablas HTML con <table>, <thead>, <tbody>, <tr>, <th>, <td>
- Para explicaciones: Usa párrafos con <p>, <strong>, <em>
- Siempre mantén un tono conversacional y profesional
"""

        return call_groq_api_safe(context)
        
    except Exception as e:
        logger.error(f"Error en consulta IA: {str(e)}")
        return "<p>Error procesando consulta con IA</p>"

def call_groq_api_safe(context, max_retries=2):
    """Llama a la API de Groq de manera segura"""
    import time
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system", 
                "content": """Eres un analista de datos experto que explica información de manera clara y conversacional. 

INSTRUCCIONES IMPORTANTES:
1. Si el usuario pide "columna X" donde X es un número mayor al total de columnas disponibles, NO digas que no tienes datos
2. En su lugar, explica qué columnas SÍ existen y ofrece mostrar las disponibles
3. Si pide columnas específicas por número, tradúcelas a los nombres reales
4. Responde SOLO con datos relevantes del CSV
5. Si necesitas mostrar datos tabulares, usa formato HTML con <table>, <tr>, <th>, <td> y class="tabla-resultados"
6. Sé útil y proactivo: si no puedes mostrar exactamente lo que piden, ofrece alternativas útiles
7. No inventes datos que no estén en el CSV

EJEMPLOS DE RESPUESTAS MEJORADAS:
- Si piden "columna 5" pero solo hay 3 columnas, responde: "Tu CSV tiene 3 columnas: [lista]. ¿Te gustaría ver alguna de estas?"
- Si piden varias columnas, algunas existentes y otras no, muestra las que sí existen"""

            },
            {"role": "user", "content": context}
        ],
        "max_tokens": 1000,  # Aumentar un poco para respuestas más completas
        "temperature": 0.3  # Slightly higher for more natural responses
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                GROQ_API_URL, 
                json=payload, 
                headers=headers, 
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            if 'choices' in data and len(data['choices']) > 0:
                raw_response = data['choices'][0]['message']['content']
                
                # 🔒 SEGURIDAD: Sanitizar respuesta de la IA
                sanitized_response = sanitize_ai_response(raw_response)
                return sanitized_response
            else:
                return '<p>No se recibió respuesta válida del servicio de análisis.</p>'
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = min((attempt + 1) * 5, 30)
                logger.warning(f"Rate limit de API alcanzado. Reintentando en {wait_time} segundos...")
                time.sleep(wait_time)
                continue
            logger.error(f"Error HTTP en API Groq: {str(e)}")
            return '<p>Error en el servicio de análisis de datos. Inténtalo de nuevo más tarde.</p>'
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout en API Groq (intento {attempt + 1})")
            if attempt == max_retries - 1:
                return '<p>El servicio de análisis está tardando demasiado. Inténtalo de nuevo más tarde.</p>'
            time.sleep(2)
            continue
            
        except Exception as e:
            logger.error(f"Error inesperado al llamar API Groq: {str(e)}")
            return '<p>Error al procesar la solicitud con el servicio de análisis.</p>'
    
    return '<p>No se pudo obtener respuesta del servicio de análisis.</p>'

def sanitize_ai_response(raw_response):
    """Sanitiza la respuesta de la IA para evitar XSS y contenido malicioso"""
    try:
        # Permitir etiquetas HTML necesarias para tablas y texto
        allowed_tags = {
            'p', 'strong', 'em', 'br', 'h3', 'h4',
            'table', 'thead', 'tbody', 'tr', 'th', 'td'
        }
        
        # Patrones maliciosos a eliminar
        malicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+="[^"]+"',
            r'on\w+=\'[^\']+\'',
            r'on\w+=[^\s>]+',
            r'expression\s*\([^)]*\)',
            r'vbscript:',
            r'<iframe[^>]*>.*?</iframe>',
            r'<meta[^>]*>',
            r'<link[^>]*>',
            r'<style[^>]*>.*?</style>',
            r'<code[^>]*>.*?</code>',  # Eliminar etiquetas de código
            r'<pre[^>]*>.*?</pre>',   # Eliminar preformateado
            r'```[^`]*```',           # Eliminar bloques de código markdown
            r'`[^`]*`'                # Eliminar código inline
        ]
        
        # Eliminar patrones maliciosos
        sanitized = raw_response
        for pattern in malicious_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Permitir ciertos atributos seguros para tablas
        safe_attributes = ['class', 'border']
        
        # Limpiar etiquetas manteniendo solo atributos seguros
        def clean_tag(match):
            tag_name = match.group(2).lower()
            if tag_name in allowed_tags:
                if tag_name == 'table':
                    return f'<{match.group(1)}table class="tabla-resultados" border="1">'
                else:
                    return f'<{match.group(1)}{tag_name}>'
            return ''
        
        sanitized = re.sub(r'<(/?)(\w+)[^>]*>', clean_tag, sanitized)
        
        # Si la respuesta está vacía después de la sanitización, proporcionar una por defecto
        if not sanitized.strip() or len(sanitized.strip()) < 10:
            return '<p>La respuesta del análisis no está disponible en este momento.</p>'
        
        return sanitized
        
    except Exception as e:
        logger.error(f"Error sanitizando respuesta IA: {str(e)}")
        return '<p>Respuesta no disponible por problemas de seguridad.</p>'

def escape_html_content(content):
    """Escapa contenido HTML para prevenir XSS"""
    if not content:
        return ""
    
    return str(content).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')

def escape_html_output(html_content):
    """Escapa contenido HTML pero permite etiquetas seguras"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Lista de etiquetas permitidas
        allowed_tags = ['table', 'thead', 'tbody', 'tr', 'th', 'td', 'p', 'strong', 'em', 'ul', 'ol', 'li', 'br']
        
        for tag in soup.find_all(True):
            if tag.name not in allowed_tags:
                tag.unwrap()  # Elimina la etiqueta pero mantiene el contenido
            else:
                # Elimina todos los atributos
                tag.attrs = {}
        
        return str(soup)
    except Exception as e:
        logger.error(f"Error escapando HTML: {str(e)}")
        return escape_html_content(html_content)

@app.route('/download_sample', methods=['GET'])
@require_rate_limit
def download_sample():
    """Descarga una muestra segura del dataset actual"""
    global df
    
    try:
        if df is None:
            return jsonify({'success': False, 'message': 'No hay datos disponibles'}), 400
        
        # 🔒 SEGURIDAD: Limitar tamaño de muestra
        sample_size = min(1000, len(df))
        safe_sample = df.head(sample_size)
        
        # 🔒 SEGURIDAD: Crear archivo temporal seguro
        from tempfile import NamedTemporaryFile
        import shutil
        
        temp_file = NamedTemporaryFile(prefix='sample_', suffix='.csv', delete=False)
        try:
            safe_sample.to_csv(temp_file.name, index=False, encoding='utf-8')
            temp_file.close()
            
            # Validar que el archivo no exceda el tamaño máximo permitido
            file_size = os.path.getsize(temp_file.name)
            if file_size > MAX_FILE_SIZE:
                os.unlink(temp_file.name)
                return jsonify({'success': False, 'message': 'Muestra demasiado grande'}), 400
            
            # Enviar archivo con cabeceras de seguridad
            response = send_from_directory(
                directory=os.path.dirname(temp_file.name),
                path=os.path.basename(temp_file.name),
                as_attachment=True,
                download_name='muestra_dataset.csv'
            )
            
            # 🔒 SEGURIDAD: Cabeceras de protección
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['Content-Security-Policy'] = "default-src 'none'"
            response.headers['Content-Disposition'] = f'attachment; filename="muestra_dataset.csv"'
            
            # Programar eliminación del archivo temporal
            @response.call_on_close
            def remove_temp_file():
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            
            return response
            
        except Exception as e:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            logger.error(f"Error generando muestra: {str(e)}")
            return jsonify({'success': False, 'message': 'Error generando archivo'}), 500
            
    except Exception as e:
        logger.error(f"Error en descarga de muestra: {str(e)}")
        return jsonify({'success': False, 'message': 'Error interno'}), 500

@app.route('/clear_data', methods=['POST'])
@require_rate_limit
def clear_data():
    """Limpia los datos cargados de manera segura"""
    global df, df_info, df_analysis
    
    try:
        df = None
        df_info = {}
        df_analysis = {}
        
        # 🔒 SEGURIDAD: Limpiar memoria
        import gc
        gc.collect()
        
        return jsonify({'success': True, 'message': 'Datos limpiados correctamente'})
        
    except Exception as e:
        logger.error(f"Error limpiando datos: {str(e)}")
        return jsonify({'success': False, 'message': 'Error limpiando datos'}), 500

# 🔒 SEGURIDAD: Middleware para cabeceras de seguridad
@app.after_request
def add_security_headers(response):
    """Agrega cabeceras de seguridad HTTP - CORREGIDAS"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    # ✅ CORREGIDO: CSP más permisivo para uploads
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

# 🔒 SEGURIDAD: Manejador de errores
@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"Intento de acceso a ruta no existente: {request.path}")
    return jsonify({'success': False, 'message': 'Recurso no encontrado'}), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Error interno en {request.path}: {str(e)}")
    return jsonify({'success': False, 'message': 'Error interno del servidor'}), 500

@app.errorhandler(413)
def request_entity_too_large(e):
    logger.warning(f"Intento de subir archivo demasiado grande desde {request.remote_addr}")
    return jsonify({'success': False, 'message': 'Archivo demasiado grande (máximo 10MB)'}), 413

# 🔒 SEGURIDAD: Deshabilitar caché para rutas sensibles
@app.after_request
def disable_caching(response):
    if request.path in ['/upload_csv', '/ask', '/download_sample', '/clear_data']:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    # ✅ CORREGIDO: Configuración condicional para desarrollo/producción
    if os.getenv('FLASK_ENV') == 'development':
        logger.info("Iniciando en modo desarrollo")
        app.run(
            host='127.0.0.1',  # Solo localhost en desarrollo
            port=int(os.getenv('PORT', 5000)),
            debug=True,  # Debug habilitado en desarrollo
            threaded=True
        )
    else:
        logger.info("Iniciando en modo producción")
        # Configuración segura del servidor Flask para producción
        ssl_context = None
        ssl_cert = os.getenv('SSL_CERT_PATH')
        ssl_key = os.getenv('SSL_KEY_PATH')
        
        if ssl_cert and ssl_key and os.path.exists(ssl_cert) and os.path.exists(ssl_key):
            ssl_context = (ssl_cert, ssl_key)
        
        app.run(
            host='0.0.0.0',
            port=int(os.getenv('PORT', 5000)),
            ssl_context=ssl_context,
            threaded=True,
            debug=False
        )