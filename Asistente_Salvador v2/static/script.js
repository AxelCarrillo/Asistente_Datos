// script.js - Versión mejorada y optimizada con formato elegante
let csvLoaded = false;
let csvInfo = null;

// Configuración
const CONFIG = {
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_EXTENSIONS: ['.csv'],
    API_TIMEOUT: 30000 // 30 segundos
};

/**
 * Sube y procesa un archivo CSV
 */
async function subirCSV() {
    const input = document.getElementById('csvInput');
    const resultadoSubida = document.getElementById('resultadoSubida');
    const uploadBtn = document.getElementById('uploadBtn');
    const askBtn = document.getElementById('askBtn');
    
    if (!input.files || input.files.length === 0) {
        showMessage('Selecciona un archivo CSV', 'error');
        return;
    }

    const file = input.files[0];
    console.log("Archivo seleccionado:", file.name, file.size, file.type);
    
    // Validaciones
    if (!validateFile(file)) return;

    const formData = new FormData();
    formData.append('file', file);
    
    setUploadingState(uploadBtn, resultadoSubida, true);

    try {
        console.log("Enviando archivo...");
        const res = await fetchWithTimeout('/upload_csv', {
            method: 'POST',
            body: formData,
        }, CONFIG.API_TIMEOUT);

        const data = await res.json();
        console.log("Datos recibidos:", data);
        
        if (data.success) {
            handleUploadSuccess(data, askBtn);
        } else {
            handleUploadError(data.message || 'Error desconocido');
        }
    } catch (error) {
        console.error("Error en subirCSV:", error);
        handleUploadError(getErrorMessage(error));
    } finally {
        setUploadingState(uploadBtn, resultadoSubida, false);
    }
}

/**
 * Envía una pregunta al servidor y muestra la respuesta
 */
async function enviarPregunta() {
    const preguntaInput = document.getElementById('pregunta');
    const respuestaDiv = document.getElementById('respuesta');
    const askBtn = document.getElementById('askBtn');

    const pregunta = preguntaInput.value.trim();
    
    // Validaciones
    if (!validateQuestion(pregunta)) return;
    if (!csvLoaded) {
        showMessage('Primero debes cargar un archivo CSV', 'error');
        return;
    }

    // UI de carga
    setAskingState(askBtn, respuestaDiv, true);

    try {
        const res = await fetchWithTimeout('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: pregunta })
        }, CONFIG.API_TIMEOUT);

        const data = await res.json();

        if (data.success) {
            handleQuestionSuccess(data, respuestaDiv, preguntaInput);
        } else {
            handleQuestionError(data.answer || 'Error en la consulta', respuestaDiv);
        }
    } catch (error) {
        console.error("Error en enviarPregunta:", error);
        handleQuestionError(getErrorMessage(error), respuestaDiv);
    } finally {
        setAskingState(askBtn, respuestaDiv, false);
    }
}

/**
 * Formatea la respuesta del servidor con estilos elegantes
 */
function formatResponse(htmlContent) {
    // Crear un contenedor temporal para procesar el HTML
    const temp = document.createElement('div');
    temp.innerHTML = htmlContent;
    
    // Procesar tablas
    const tables = temp.querySelectorAll('table');
    tables.forEach(table => {
        table.className = 'response-table';
        
        // Agregar clases a headers
        const headers = table.querySelectorAll('th');
        headers.forEach(th => th.className = 'table-header');
        
        // Agregar clases a celdas
        const cells = table.querySelectorAll('td');
        cells.forEach(td => td.className = 'table-cell');
        
        // Si no hay headers, convertir primera fila
        if (headers.length === 0) {
            const firstRowCells = table.querySelectorAll('tr:first-child td');
            firstRowCells.forEach(td => {
                td.tagName = 'th';
                td.className = 'table-header';
            });
        }
    });
    
    // Procesar listas
    const lists = temp.querySelectorAll('ul, ol');
    lists.forEach(list => {
        list.className = 'response-list';
        const items = list.querySelectorAll('li');
        items.forEach(item => item.className = 'list-item');
    });
    
    // Procesar párrafos
    const paragraphs = temp.querySelectorAll('p');
    paragraphs.forEach(p => {
        if (p.textContent.trim()) {
            p.className = 'response-paragraph';
        }
    });
    
    // Procesar headers
    const headers = temp.querySelectorAll('h1, h2, h3, h4, h5, h6');
    headers.forEach(h => h.className = 'response-header');
    
    // Procesar texto en negrita
    const boldText = temp.querySelectorAll('strong, b');
    boldText.forEach(b => b.className = 'response-bold');
    
    // Procesar código inline
    const codeBlocks = temp.querySelectorAll('code');
    codeBlocks.forEach(code => {
        if (!code.parentElement.tagName === 'PRE') {
            code.className = 'inline-code';
        }
    });
    
    // Procesar bloques de código
    const preBlocks = temp.querySelectorAll('pre');
    preBlocks.forEach(pre => {
        pre.className = 'code-block';
        const code = pre.querySelector('code');
        if (code) code.className = 'code-content';
    });
    
    return temp.innerHTML;
}

/**
 * Detecta si el contenido contiene una tabla y la formatea especialmente
 */
function formatTableResponse(content) {
    // Detectar si es una respuesta que debería ser tabla
    const tablePatterns = [
        /\|.*\|/g,  // Markdown tables
        /^\s*\w+\s*:\s*.+$/gm,  // Key-value pairs
        /^\d+\.\s+/gm,  // Numbered lists that could be table rows
    ];
    
    let hasTableStructure = tablePatterns.some(pattern => pattern.test(content));
    
    if (hasTableStructure && !content.includes('<table>')) {
        // Convertir texto estructurado a tabla HTML
        return convertTextToTable(content);
    }
    
    return content;
}

/**
 * Convierte texto estructurado en tabla HTML
 */
function convertTextToTable(text) {
    const lines = text.split('\n').filter(line => line.trim());
    
    // Detectar si son pares clave-valor
    const keyValuePattern = /^(.+?):\s*(.+)$/;
    const keyValueLines = lines.filter(line => keyValuePattern.test(line));
    
    if (keyValueLines.length > 0 && keyValueLines.length === lines.length) {
        let tableHTML = '<table class="response-table"><tbody>';
        
        keyValueLines.forEach(line => {
            const match = line.match(keyValuePattern);
            if (match) {
                tableHTML += `
                    <tr>
                        <td class="table-header" style="font-weight: bold;">${match[1].trim()}</td>
                        <td class="table-cell">${match[2].trim()}</td>
                    </tr>
                `;
            }
        });
        
        tableHTML += '</tbody></table>';
        return tableHTML;
    }
    
    return text;
}

/**
 * Agrega estilos CSS dinámicamente si no existen
 */
function addResponseStyles() {
    const styleId = 'response-styles';
    if (document.getElementById(styleId)) return;
    
    const styles = `
        <style id="${styleId}">
        .response-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #2d3748;
            max-width: 100%;
            overflow-x: auto;
        }
        
        .response-table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .table-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-weight: 600;
            padding: 12px 16px;
            text-align: left;
            border: none;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .table-cell {
            padding: 12px 16px;
            border-bottom: 1px solid #e2e8f0;
            background: white;
            transition: background-color 0.2s ease;
        }
        
        .response-table tr:nth-child(even) .table-cell {
            background: #f8f9fa;
        }
        
        .response-table tr:hover .table-cell {
            background: #e3f2fd;
        }
        
        .response-list {
            margin: 16px 0;
            padding-left: 0;
            list-style: none;
        }
        
        .list-item {
            padding: 8px 0;
            border-left: 3px solid #667eea;
            padding-left: 16px;
            margin: 4px 0;
            background: linear-gradient(90deg, rgba(102,126,234,0.1) 0%, transparent 100%);
        }
        
        .list-item:before {
            content: "▶";
            color: #667eea;
            margin-right: 8px;
            font-size: 12px;
        }
        
        .response-paragraph {
            margin: 12px 0;
            text-align: justify;
            background: #f8f9ff;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }
        
        .response-header {
            color: #4c51bf;
            font-weight: 700;
            margin: 20px 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .response-bold {
            color: #2d3748;
            font-weight: 600;
            background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        .inline-code {
            background: #2d3748;
            color: #e2e8f0;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 13px;
        }
        
        .code-block {
            background: #1a202c;
            color: #e2e8f0;
            padding: 16px;
            border-radius: 8px;
            margin: 16px 0;
            overflow-x: auto;
            border-left: 4px solid #667eea;
        }
        
        .code-content {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 13px;
            line-height: 1.5;
        }
        
        .status-success {
            background: linear-gradient(135deg, #48bb78, #38a169);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 8px 0;
            font-weight: 500;
        }
        
        .status-error {
            background: linear-gradient(135deg, #f56565, #e53e3e);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 8px 0;
            font-weight: 500;
        }
        
        .status-info {
            background: linear-gradient(135deg, #4299e1, #3182ce);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 8px 0;
            font-weight: 500;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
            font-style: italic;
        }
        
        .loading:after {
            content: "...";
            animation: dots 1.5s infinite;
        }
        
        @keyframes dots {
            0%, 20% { content: "..."; }
            40% { content: ".."; }
            60% { content: "."; }
            80%, 100% { content: ""; }
        }
        </style>
    `;
    
    document.head.insertAdjacentHTML('beforeend', styles);
}

/**
 * Validaciones
 */
function validateFile(file) {
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        showMessage(`El archivo es demasiado grande. Máximo ${CONFIG.MAX_FILE_SIZE / 1024 / 1024}MB permitido.`, 'error');
        return false;
    }

    const extension = '.' + file.name.split('.').pop().toLowerCase();
    if (!CONFIG.ALLOWED_EXTENSIONS.includes(extension)) {
        showMessage('Por favor, selecciona un archivo CSV válido.', 'error');
        return false;
    }

    if (file.size === 0) {
        showMessage('El archivo está vacío.', 'error');
        return false;
    }

    return true;
}

function validateQuestion(pregunta) {
    if (!pregunta) {
        showMessage('Por favor, escribe una pregunta', 'error');
        return false;
    }
    
    if (pregunta.length > 500) {
        showMessage('La pregunta es demasiado larga. Máximo 500 caracteres.', 'error');
        return false;
    }
    
    return true;
}

/**
 * Manejadores de éxito
 */
function handleUploadSuccess(data, askBtn) {
    csvLoaded = true;
    csvInfo = data;
    askBtn.disabled = false;
    
    const message = `✅ CSV cargado exitosamente<br>
        📊 <strong>${data.rows}</strong> filas, <strong>${data.columns.length}</strong> columnas<br>
        📋 Columnas: ${data.columns.join(', ')}`;
    
    showMessage(message, 'success');
    
    document.getElementById('respuesta').innerHTML = 
        `<div class="response-container">
            <p class="response-paragraph" style="color: #4c51bf; font-weight: 500;">
                🎉 ¡Perfecto! Tu CSV está listo. Ahora puedes hacer preguntas sobre tus datos.
            </p>
        </div>`;
}

function handleQuestionSuccess(data, respuestaDiv, preguntaInput) {
    // Agregar estilos si no existen
    addResponseStyles();
    
    // Formatear respuesta
    let formattedContent = data.answer;
    
    // Detectar y formatear tablas
    formattedContent = formatTableResponse(formattedContent);
    
    // Aplicar formato general
    formattedContent = formatResponse(formattedContent);
    
    // Crear contenedor con estilos
    const responseHTML = `
        <div class="response-container">
            ${formattedContent}
        </div>
    `;
    
    respuestaDiv.innerHTML = responseHTML;
    preguntaInput.value = '';
    
    // Scroll suave a la respuesta
    setTimeout(() => {
        respuestaDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

/**
 * Manejadores de error
 */
function handleUploadError(message) {
    csvLoaded = false;
    csvInfo = null;
    showMessage(`❌ Error: ${message}`, 'error');
}

function handleQuestionError(message, respuestaDiv) {
    addResponseStyles();
    respuestaDiv.innerHTML = `<div class="response-container"><div class="status-message status-error">❌ ${message}</div></div>`;
}

/**
 * Estados de UI
 */
function setUploadingState(uploadBtn, resultadoSubida, isUploading) {
    uploadBtn.disabled = isUploading;
    uploadBtn.textContent = isUploading ? '⏳ Subiendo...' : '⬆️ Subir CSV';
    
    if (isUploading) {
        resultadoSubida.innerHTML = '<div class="status-message status-info">📤 Subiendo archivo...</div>';
    }
}

function setAskingState(askBtn, respuestaDiv, isAsking) {
    askBtn.disabled = isAsking;
    askBtn.textContent = isAsking ? '🤔 Pensando...' : '🔍 Preguntar';
    
    if (isAsking) {
        addResponseStyles();
        respuestaDiv.innerHTML = '<div class="response-container"><div class="loading">🧠 Analizando tus datos</div></div>';
    }
}

/**
 * Utilidades
 */
function showMessage(message, type) {
    addResponseStyles();
    const resultadoSubida = document.getElementById('resultadoSubida');
    const className = type === 'success' ? 'status-success' : 
                     type === 'error' ? 'status-error' : 'status-info';
    
    resultadoSubida.innerHTML = `<div class="status-message ${className}">${message}</div>`;
    
    if (type === 'info') {
        setTimeout(() => {
            if (resultadoSubida.innerHTML.includes(message)) {
                resultadoSubida.innerHTML = '';
            }
        }, 5000);
    }
}

async function fetchWithTimeout(url, options, timeout) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return response;
    } finally {
        clearTimeout(timeoutId);
    }
}

function getErrorMessage(error) {
    if (error.name === 'AbortError') {
        return 'Tiempo de espera agotado. Intenta de nuevo.';
    }
    if (error.message.includes('Failed to fetch')) {
        return 'Error de conexión. Verifica tu conexión a internet.';
    }
    return error.message || 'Error desconocido';
}

/**
 * Event Listeners
 */
document.addEventListener('DOMContentLoaded', function() {
    // Agregar estilos al cargar la página
    addResponseStyles();
    
    // Enter para enviar pregunta
    const preguntaInput = document.getElementById('pregunta');
    if (preguntaInput) {
        preguntaInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                enviarPregunta();
            }
        });
    }

    // Actualizar label del archivo cuando se selecciona
    const csvInput = document.getElementById('csvInput');
    if (csvInput) {
        csvInput.addEventListener('change', function() {
            const label = document.querySelector('.file-label');
            if (this.files.length > 0) {
                const fileName = this.files[0].name;
                const fileSize = (this.files[0].size / 1024 / 1024).toFixed(2);
                label.textContent = `📄 ${fileName} (${fileSize} MB)`;
            } else {
                label.textContent = '📄 Seleccionar archivo CSV';
            }
        });
    }

    // Drag and drop para archivos
    const uploadSection = document.querySelector('.upload-section');
    if (uploadSection) {
        uploadSection.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.borderColor = '#4c51bf';
            this.style.background = '#f0f1ff';
        });

        uploadSection.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.borderColor = '#c3c6e8';
            this.style.background = '#f8f9ff';
        });

        uploadSection.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.borderColor = '#c3c6e8';
            this.style.background = '#f8f9ff';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                csvInput.files = files;
                csvInput.dispatchEvent(new Event('change'));
            }
        });
    }
});

// Exportar funciones para uso global
window.subirCSV = subirCSV;
window.enviarPregunta = enviarPregunta;