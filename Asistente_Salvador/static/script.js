// script.js - Versión mejorada y optimizada
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
    const uploadBtn = document.getElementById('uploadBtn');
    const askBtn = document.getElementById('askBtn');
    
    // Validaciones iniciales
    if (!validateFileInput(input)) return;
    
    const file = input.files[0];
    
    // Validaciones del archivo
    if (!validateFile(file)) return;
    
    const formData = new FormData();
    formData.append('file', file);

    // UI de carga
    setUploadingState(uploadBtn, true);
    showMessage('📤 Subiendo archivo...', 'info');

    try {
        const response = await fetchWithTimeout('/upload_csv', {
            method: 'POST',
            body: formData
        }, CONFIG.API_TIMEOUT);

        const data = await response.json();
        
        if (data.success) {
            handleUploadSuccess(data, askBtn);
        } else {
            handleUploadError(data.message);
        }
    } catch (error) {
        handleUploadError(getErrorMessage(error));
    } finally {
        setUploadingState(uploadBtn, false);
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
        const response = await fetchWithTimeout('/ask', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ question: pregunta })
        }, CONFIG.API_TIMEOUT);

        const data = await response.json();

        if (data.success) {
            handleQuestionSuccess(data, respuestaDiv, preguntaInput);
        } else {
            handleQuestionError(data.answer, respuestaDiv);
        }
    } catch (error) {
        handleQuestionError(getErrorMessage(error), respuestaDiv);
    } finally {
        setAskingState(askBtn, respuestaDiv, false);
    }
}

/**
 * Validaciones
 */
function validateFileInput(input) {
    if (!input || input.files.length === 0) {
        showMessage('Selecciona un archivo CSV', 'error');
        return false;
    }
    return true;
}

function validateFile(file) {
    // Validar tamaño
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        showMessage(`El archivo es demasiado grande. Máximo ${CONFIG.MAX_FILE_SIZE / 1024 / 1024}MB permitido.`, 'error');
        return false;
    }

    // Validar extensión
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    if (!CONFIG.ALLOWED_EXTENSIONS.includes(extension)) {
        showMessage('Por favor, selecciona un archivo CSV válido.', 'error');
        return false;
    }

    // Validar que no esté vacío
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
    
    // Actualizar respuesta inicial
    document.getElementById('respuesta').innerHTML = 
        `<p style="color: #4c51bf; font-weight: 500;">
          ¡Perfecto! Tu CSV está listo. Ahora puedes hacer preguntas sobre tus datos.
        </p>`;
}

function handleQuestionSuccess(data, respuestaDiv, preguntaInput) {
    // Sanitizar y mostrar respuesta
    respuestaDiv.innerHTML = sanitizeHtml(data.answer);
    
    // Limpiar input
    preguntaInput.value = '';
    
    // Scroll suave a la respuesta
    respuestaDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
    respuestaDiv.innerHTML = `<div class="status-message status-error">❌ ${message}</div>`;
}

/**
 * Estados de UI
 */
function setUploadingState(uploadBtn, isUploading) {
    uploadBtn.disabled = isUploading;
    uploadBtn.textContent = isUploading ? '⏳ Subiendo...' : '⬆️ Subir CSV';
}

function setAskingState(askBtn, respuestaDiv, isAsking) {
    askBtn.disabled = isAsking;
    askBtn.textContent = isAsking ? '🤔 Pensando...' : '🔍 Preguntar';
    
    if (isAsking) {
        respuestaDiv.innerHTML = '<div class="loading">Analizando tus datos</div>';
    }
}

/**
 * Utilidades
 */
function showMessage(message, type) {
    const resultadoSubida = document.getElementById('resultadoSubida');
    const className = type === 'success' ? 'status-success' : 
                     type === 'error' ? 'status-error' : 'status-info';
    
    resultadoSubida.innerHTML = `<div class="status-message ${className}">${message}</div>`;
    
    // Auto-hide después de 5 segundos para mensajes de info
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

function sanitizeHtml(html) {
    // Permitir solo elementos HTML seguros
    const allowedTags = ['p', 'div', 'span', 'strong', 'em', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'tr', 'td', 'th', 'br'];
    const allowedAttributes = ['class', 'style'];
    
    // Esta es una implementación básica. En producción, usar una librería como DOMPurify
    const temp = document.createElement('div');
    temp.innerHTML = html;
    
    // Remover scripts y eventos
    const scripts = temp.querySelectorAll('script');
    scripts.forEach(script => script.remove());
    
    const elements = temp.querySelectorAll('*');
    elements.forEach(el => {
        // Remover atributos no permitidos
        Array.from(el.attributes).forEach(attr => {
            if (!allowedAttributes.includes(attr.name.toLowerCase())) {
                el.removeAttribute(attr.name);
            }
        });
    });
    
    return temp.innerHTML;
}

/**
 * Event Listeners
 */
document.addEventListener('DOMContentLoaded', function() {
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

/**
 * Funciones de utilidad adicionales
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Exportar funciones para uso global (si es necesario)
window.subirCSV = subirCSV;
window.enviarPregunta = enviarPregunta;