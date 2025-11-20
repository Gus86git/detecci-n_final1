import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
import time
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import csv
from io import BytesIO

# =============================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================
st.set_page_config(
    page_title="SafeBuild - Monitoreo de Seguridad con IA",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CSS PERSONALIZADO
# =============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .alert-high {
        background-color: #1F2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 6px solid #DC2626;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
    }
    .alert-medium {
        background-color: #1F2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 6px solid #D97706;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
    }
    .alert-ok {
        background-color: #1F2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 6px solid #059669;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
    }
    .metric-card {
        background-color: #1F2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #374151;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: white;
    }
    .detection-box {
        background-color: #1F2937;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #374151;
        margin: 0.5rem 0;
        color: white;
    }
    .detection-detail-box {
        background-color: #1F2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #374151;
        margin: 0.5rem 0;
        color: white;
    }
    .sidebar-section {
        background-color: #1F2937;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: white;
    }
    .stButton button {
        width: 100%;
        background-color: #1E40AF;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #1E3A8A;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #1E3A8A;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
        color: white;
    }
    .historial-box {
        background-color: #1F2937;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #374151;
        margin: 0.5rem 0;
        color: white;
    }
    .config-section {
        background-color: #1F2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: white;
    }
    .panel-section {
        background-color: #1F2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: white;
    }
    .analysis-section {
        background-color: #1F2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white;
    }
    .detections-section {
        background-color: #1F2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white;
    }
    .expert-analysis-section {
        background-color: #1F2937;
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white;
        border: 2px solid #374151;
    }
    .stats-section {
        background-color: #1F2937;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: white;
    }
    .alert-title {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .alert-message {
        color: white;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        background-color: #374151;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-action {
        color: white;
        background-color: #4B5563;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-priority {
        color: white;
        background-color: #6B7280;
        padding: 0.75rem;
        border-radius: 0.5rem;
        display: inline-block;
        margin-right: 1rem;
    }
    .alert-compliance {
        color: white;
        background-color: #6B7280;
        padding: 0.75rem;
        border-radius: 0.5rem;
        display: inline-block;
    }
    .export-section {
        background-color: #1F2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 2px solid #374151;
        color: white;
    }
    .export-button {
        background: linear-gradient(45deg, #059669, #10B981);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
        margin: 0.5rem;
    }
    .export-button:hover {
        background: linear-gradient(45deg, #047857, #059669);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# SISTEMA EXPERTO DE SEGURIDAD MEJORADO
# =============================================
class SafetyExpertSystem:
    def __init__(self):
        self.rules = {
            'height_risk_critical': {
                'condition': lambda stats: stats['persons_high_risk'] > 0 and stats['helmets_high_risk'] == 0,
                'message': "CRÍTICO: Personal en zona de altura sin ningún casco",
                'level': "ALTA",
                'action': "🚫 SUSPENDER trabajos en altura - Implementar andamios y redes",
                'priority': 1
            },
            'height_risk_partial': {
                'condition': lambda stats: stats['persons_high_risk'] > 0 and stats['helmets_high_risk'] < stats['persons_high_risk'],
                'message': "ALTO RIESGO: Personal en zona elevada sin protección completa",
                'level': "ALTA", 
                'action': "📏 DELIMITAR área de riesgo - Proveer EPP inmediatamente",
                'priority': 2
            },
            'no_ppe_complete': {
                'condition': lambda stats: stats['persons'] > 0 and stats['full_ppe'] == 0,
                'message': "PROTECCIÓN INCOMPLETA: Ningún trabajador con EPP completo",
                'level': "ALTA",
                'action': "🛑 DETENER actividades - Verificar dotación de EPP completo",
                'priority': 3
            },
            'no_helmet_critical': {
                'condition': lambda stats: stats['persons'] > 0 and stats['helmets'] == 0,
                'message': "CRÍTICO: Ningún trabajador usa casco de seguridad",
                'level': "ALTA",
                'action': "DETENER actividades inmediatamente y notificar al supervisor de seguridad",
                'priority': 4
            },
            'no_helmet_partial': {
                'condition': lambda stats: stats['persons'] > 0 and stats['helmets'] < stats['persons'],
                'message': "ALTA: Trabajadores detectados sin casco de seguridad",
                'level': "ALTA", 
                'action': "Aislar el área y proveer EPP inmediatamente",
                'priority': 5
            },
            'no_vest_critical': {
                'condition': lambda stats: stats['persons'] > 0 and stats['vests'] == 0,
                'message': "MEDIA: Ningún trabajador usa chaleco reflectante",
                'level': "MEDIA",
                'action': "Notificar al supervisor y proveer chalecos de seguridad",
                'priority': 6
            },
            'no_vest_partial': {
                'condition': lambda stats: stats['persons'] > 0 and stats['vests'] < stats['persons'],
                'message': "MEDIA: Trabajadores detectados sin chaleco reflectante",
                'level': "MEDIA",
                'action': "Recordar uso obligatorio de chaleco en reunión de seguridad",
                'priority': 7
            },
            'proper_equipment': {
                'condition': lambda stats: stats['persons'] > 0 and stats['helmets'] >= stats['persons'] and stats['vests'] >= stats['persons'],
                'message': "OK: Todo el personal cuenta con Equipo de Protección Personal completo",
                'level': "OK",
                'action': "Continuar monitoreo y mantener los estándares de seguridad",
                'priority': 8
            },
            'no_persons': {
                'condition': lambda stats: stats['persons'] == 0,
                'message': "OK: No se detectaron trabajadores en el área analizada",
                'level': "OK", 
                'action': "Continuar con el monitoreo rutinario del área",
                'priority': 9
            }
        }
    
    def analyze_detections(self, detections, confidence_threshold=0.5, image_size=None):
        """Analiza las detecciones y aplica las reglas del sistema experto"""
        # Estadísticas básicas
        person_count = sum(1 for det in detections if det['class'] in ['person', 'worker'] and det['confidence'] >= confidence_threshold)
        helmet_count = sum(1 for det in detections if det['class'] in ['helmet', 'hardhat', 'hard-hat'] and det['confidence'] >= confidence_threshold)
        vest_count = sum(1 for det in detections if det['class'] in ['safety_vest', 'vest', 'safety-vest'] and det['confidence'] >= confidence_threshold)
        
        # Nuevas estadísticas de contexto
        context_stats = self._analyze_context(detections, image_size, confidence_threshold)
        
        detection_stats = {
            'persons': person_count,
            'helmets': helmet_count,
            'vests': vest_count,
            'total_detections': len(detections),
            'persons_high_risk': context_stats['persons_high_risk'],
            'helmets_high_risk': context_stats['helmets_high_risk'],
            'full_ppe': context_stats['full_ppe']
        }
        
        # Aplicar reglas en orden de prioridad
        for rule_name, rule in sorted(self.rules.items(), key=lambda x: x[1]['priority']):
            if rule['condition'](detection_stats):
                return {
                    'alert_level': rule['level'],
                    'alert_message': rule['message'],
                    'recommended_action': rule['action'],
                    'statistics': detection_stats,
                    'compliance_rate': self._calculate_compliance(detection_stats),
                    'rule_triggered': rule_name
                }
        
        return {
            'alert_level': "OK",
            'alert_message': "Condiciones normales de seguridad detectadas",
            'recommended_action': "Continuar con el monitoreo rutinario",
            'statistics': detection_stats,
            'compliance_rate': 100.0,
            'rule_triggered': 'default'
        }
    
    def _analyze_context(self, detections, image_size, confidence_threshold):
        """Analiza el contexto de las detecciones para reglas avanzadas"""
        if image_size is None:
            return {
                'persons_high_risk': 0,
                'helmets_high_risk': 0,
                'full_ppe': 0
            }
        
        height, width = image_size
        persons_high_risk = 0
        helmets_high_risk = 0
        full_ppe_count = 0
        
        # Obtener todas las personas
        persons = [d for d in detections if d['class'] in ['person', 'worker'] and d['confidence'] >= confidence_threshold]
        
        for person in persons:
            # Calcular posición vertical (y_center)
            y_center = (person['bbox'][1] + person['bbox'][3]) / 2
            
            # Persona en zona de alto riesgo (parte superior de la imagen)
            if y_center < height * 0.4:  # 40% superior de la imagen
                persons_high_risk += 1
                
                # Verificar si tiene casco en zona de riesgo
                if self._has_helmet_nearby(person, detections):
                    helmets_high_risk += 1
            
            # Verificar EPP completo (casco + chaleco)
            if self._has_full_ppe(person, detections):
                full_ppe_count += 1
        
        return {
            'persons_high_risk': persons_high_risk,
            'helmets_high_risk': helmets_high_risk,
            'full_ppe': full_ppe_count
        }
    
    def _has_helmet_nearby(self, person_det, all_detections):
        """Verifica si una persona tiene casco cerca (misma zona)"""
        person_bbox = person_det['bbox']
        
        for det in all_detections:
            if det['class'] in ['helmet', 'hardhat', 'hard-hat']:
                helmet_bbox = det['bbox']
                # Verificar superposición en eje Y (misma altura)
                if self._bboxes_overlap_vertical(person_bbox, helmet_bbox):
                    return True
        return False
    
    def _has_full_ppe(self, person_det, all_detections):
        """Verifica si una persona tiene EPP completo (casco + chaleco)"""
        person_bbox = person_det['bbox']
        has_helmet = False
        has_vest = False
        
        for det in all_detections:
            if det['class'] in ['helmet', 'hardhat', 'hard-hat']:
                if self._bboxes_overlap_vertical(person_bbox, det['bbox']):
                    has_helmet = True
            elif det['class'] in ['safety_vest', 'vest', 'safety-vest']:
                if self._bboxes_overlap_vertical(person_bbox, det['bbox']):
                    has_vest = True
        
        return has_helmet and has_vest
    
    def _bboxes_overlap_vertical(self, bbox1, bbox2, threshold=0.3):
        """Verifica si dos bounding boxes se superponen verticalmente"""
        y1_1, y2_1 = bbox1[1], bbox1[3]
        y1_2, y2_2 = bbox2[1], bbox2[3]
        
        # Calcular superposición vertical
        overlap = min(y2_1, y2_2) - max(y1_1, y1_2)
        height1 = y2_1 - y1_1
        
        return overlap > height1 * threshold
    
    def _calculate_compliance(self, stats):
        """Calcula el porcentaje de cumplimiento de EPP"""
        if stats['persons'] == 0:
            return 100.0
        
        helmet_compliance = (stats['helmets'] / stats['persons']) * 100
        vest_compliance = (stats['vests'] / stats['persons']) * 100
        
        # Penalización por riesgo de altura sin protección
        height_risk_penalty = 0
        if stats['persons_high_risk'] > 0:
            height_protection_rate = stats['helmets_high_risk'] / stats['persons_high_risk'] if stats['persons_high_risk'] > 0 else 1
            height_risk_penalty = (1 - height_protection_rate) * 20  # Hasta 20% de penalización
        
        # Promedio ponderado (casco es más crítico)
        total_compliance = (helmet_compliance * 0.6 + vest_compliance * 0.4) - height_risk_penalty
        return max(0, round(total_compliance, 1))

# =============================================
# SISTEMA DE EXPORTACIÓN DE DATOS
# =============================================
def generate_export_data():
    """Genera DataFrames para exportación"""
    
    # DataFrame principal de análisis
    analysis_data = []
    for i, record in enumerate(st.session_state.analysis_history):
        analysis_data.append({
            'ID_Análisis': i + 1,
            'Fecha': record['timestamp'].strftime('%Y-%m-%d'),
            'Hora': record['timestamp'].strftime('%H:%M:%S'),
            'Archivo': record['filename'],
            'Nivel_Alerta': record['alert_level'],
            'Regla_Activada': record.get('rule_triggered', 'N/A'),
            'Cumplimiento_EPP': f"{record.get('compliance_rate', 0):.1f}%",
            'Total_Personas': record['statistics']['persons'],
            'Cascos_Detectados': record['statistics']['helmets'],
            'Chalecos_Detectados': record['statistics']['vests'],
            'EPP_Completo': record['statistics']['full_ppe'],
            'Personas_Riesgo_Altura': record['statistics']['persons_high_risk'],
            'Total_Detecciones': record['statistics']['total_detections']
        })
    
    df_analysis = pd.DataFrame(analysis_data)
    
    # DataFrame de estadísticas resumidas
    if analysis_data:
        summary_data = {
            'Métrica': [
                'Total de Análisis Realizados',
                'Alertas de Alto Riesgo',
                'Alertas de Riesgo Medio', 
                'Condiciones Seguras',
                'Cumplimiento Promedio EPP',
                'Personas Detectadas (Total)',
                'Cascos Detectados (Total)',
                'Chalecos Detectados (Total)',
                'EPP Completo (Total)',
                'Tasa de Cumplimiento de Cascos',
                'Tasa de Cumplimiento de Chalecos'
            ],
            'Valor': [
                len(analysis_data),
                sum(1 for r in analysis_data if r['Nivel_Alerta'] == 'ALTA'),
                sum(1 for r in analysis_data if r['Nivel_Alerta'] == 'MEDIA'),
                sum(1 for r in analysis_data if r['Nivel_Alerta'] == 'OK'),
                f"{np.mean([float(r['Cumplimiento_EPP'].replace('%', '')) for r in analysis_data]):.1f}%",
                sum(r['Total_Personas'] for r in analysis_data),
                sum(r['Cascos_Detectados'] for r in analysis_data),
                sum(r['Chalecos_Detectados'] for r in analysis_data),
                sum(r['EPP_Completo'] for r in analysis_data),
                f"{(sum(r['Cascos_Detectados'] for r in analysis_data) / sum(r['Total_Personas'] for r in analysis_data) * 100) if sum(r['Total_Personas'] for r in analysis_data) > 0 else 0:.1f}%",
                f"{(sum(r['Chalecos_Detectados'] for r in analysis_data) / sum(r['Total_Personas'] for r in analysis_data) * 100) if sum(r['Total_Personas'] for r in analysis_data) > 0 else 0:.1f}%"
            ]
        }
        df_summary = pd.DataFrame(summary_data)
    else:
        df_summary = pd.DataFrame({'Métrica': ['No hay datos'], 'Valor': ['N/A']})
    
    return df_analysis, df_summary

def export_to_csv():
    """Exporta todos los datos a CSV"""
    df_analysis, df_summary = generate_export_data()
    
    # Crear un buffer en memoria
    output = BytesIO()
    
    # Crear un escritor Excel con múltiples hojas
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_analysis.to_excel(writer, sheet_name='Análisis_Detallado', index=False)
        df_summary.to_excel(writer, sheet_name='Estadísticas_Resumen', index=False)
        
        # Formato para Excel
        workbook = writer.book
        worksheet_analysis = writer.sheets['Análisis_Detallado']
        worksheet_summary = writer.sheets['Estadísticas_Resumen']
        
        # Autoajustar columnas
        for column in df_analysis:
            column_width = max(df_analysis[column].astype(str).map(len).max(), len(column))
            col_idx = df_analysis.columns.get_loc(column)
            worksheet_analysis.set_column(col_idx, col_idx, column_width)
        
        for column in df_summary:
            column_width = max(df_summary[column].astype(str).map(len).max(), len(column))
            col_idx = df_summary.columns.get_loc(column)
            worksheet_summary.set_column(col_idx, col_idx, column_width)
    
    output.seek(0)
    return output

def export_to_excel():
    """Exporta todos los datos a Excel"""
    return export_to_csv()  # Ya estamos usando Excel con múltiples hojas

# =============================================
# DETECTOR YOLO Y FUNCIONES DE DETECCIÓN
# =============================================
@st.cache_resource
def load_yolo_model():
    """Carga el modelo YOLO (cachea para evitar recargas)"""
    try:
        # Intenta cargar modelo personalizado si existe
        if os.path.exists('models/best.pt'):
            model = YOLO('models/best.pt')
            st.sidebar.success("✅ Modelo personalizado cargado")
        else:
            # Usa YOLOv8n como modelo base
            model = YOLO('yolov8n.pt')
            st.sidebar.info("ℹ️ Usando YOLOv8n base")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {str(e)}")
        return None

def detect_helmet_by_color(region):
    """
    Detecta cascos basándose en colores característicos
    Cascos comunes: blanco, amarillo, naranja, rojo, azul
    """
    try:
        if region.size == 0:
            return False
        
        # Convertir a HSV para mejor detección de colores
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Definir rangos de colores para cascos
        # Amarillo
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        # Naranja
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([15, 255, 255])
        
        # Rojo (dos rangos)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Azul
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        # Blanco (alta luminosidad)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Crear máscaras
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combinar máscaras
        combined_mask = mask_yellow | mask_orange | mask_red | mask_blue | mask_white
        
        # Calcular porcentaje de píxeles que coinciden
        total_pixels = region.shape[0] * region.shape[1]
        colored_pixels = np.count_nonzero(combined_mask)
        percentage = colored_pixels / total_pixels
        
        # Si más del 15% de la región tiene estos colores, probablemente es un casco
        return percentage > 0.15
    except:
        return False

def detect_vest_by_color(region):
    """
    Detecta chalecos reflectantes basándose en colores característicos
    Chalecos: amarillo fluorescente, naranja fluorescente, verde lima
    """
    try:
        if region.size == 0:
            return False
        
        # Convertir a HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Amarillo fluorescente (muy común en chalecos)
        lower_yellow = np.array([20, 100, 150])
        upper_yellow = np.array([35, 255, 255])
        
        # Naranja fluorescente
        lower_orange = np.array([5, 150, 150])
        upper_orange = np.array([20, 255, 255])
        
        # Verde lima (menos común pero usado)
        lower_lime = np.array([35, 100, 100])
        upper_lime = np.array([85, 255, 255])
        
        # Crear máscaras
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_lime = cv2.inRange(hsv, lower_lime, upper_lime)
        
        # Combinar máscaras
        combined_mask = mask_yellow | mask_orange | mask_lime
        
        # Calcular porcentaje
        total_pixels = region.shape[0] * region.shape[1]
        colored_pixels = np.count_nonzero(combined_mask)
        percentage = colored_pixels / total_pixels
        
        # Si más del 20% de la región tiene estos colores, probablemente es un chaleco
        return percentage > 0.20
    except:
        return False

def enhance_ppe_detection(image, detections):
    """
    Mejora las detecciones de EPP usando análisis de color y posición
    YOLOv8 base no está entrenado para EPP, así que inferimos basado en:
    - Cascos: objetos en la parte superior de personas (región de cabeza) con colores típicos
    - Chalecos: objetos en torso con colores reflectantes (amarillo, naranja, verde)
    """
    enhanced = detections.copy()
    
    # Separar personas del resto
    persons = [d for d in detections if d['class'] == 'person']
    
    for person in persons:
        x1, y1, x2, y2 = person['bbox']
        person_width = x2 - x1
        person_height = y2 - y1
        
        # Región de la cabeza (20% superior del cuerpo)
        head_region = image[y1:int(y1 + person_height * 0.2), x1:x2]
        
        # Región del torso (30-70% del cuerpo)
        torso_region = image[int(y1 + person_height * 0.3):int(y1 + person_height * 0.7), x1:x2]
        
        if head_region.size > 0:
            # Detectar casco basado en colores característicos
            helmet_detected = detect_helmet_by_color(head_region)
            if helmet_detected:
                # Agregar detección de casco
                helmet_bbox = [
                    x1,
                    y1,
                    x2,
                    int(y1 + person_height * 0.25)
                ]
                enhanced.append({
                    'class': 'helmet',
                    'confidence': 0.70,  # Confianza media-alta para inferencia
                    'bbox': helmet_bbox,
                    'area': (helmet_bbox[2]-helmet_bbox[0]) * (helmet_bbox[3]-helmet_bbox[1]),
                    'inferred': True
                })
        
        if torso_region.size > 0:
            # Detectar chaleco basado en colores reflectantes
            vest_detected = detect_vest_by_color(torso_region)
            if vest_detected:
                # Agregar detección de chaleco
                vest_bbox = [
                    x1,
                    int(y1 + person_height * 0.25),
                    x2,
                    int(y1 + person_height * 0.75)
                ]
                enhanced.append({
                    'class': 'safety_vest',
                    'confidence': 0.65,  # Confianza media para inferencia
                    'bbox': vest_bbox,
                    'area': (vest_bbox[2]-vest_bbox[0]) * (vest_bbox[3]-vest_bbox[1]),
                    'inferred': True
                })
    
    return enhanced

def detect_objects(image, model, confidence_threshold=0.5):
    """Realiza detección de objetos en la imagen con parámetros optimizados"""
    try:
        # Convertir imagen PIL a formato OpenCV
        img_array = np.array(image)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Realizar inferencia con parámetros optimizados
        results = model.predict(
            img_rgb,
            conf=confidence_threshold,
            iou=0.45,  # Umbral de IoU para NMS (Non-Maximum Suppression)
            imgsz=640,  # Tamaño de imagen optimizado
            augment=True,  # Test Time Augmentation para mejor precisión
            agnostic_nms=False,  # NMS por clase
            max_det=300,  # Máximo de detecciones
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls].lower()
                
                # Mapear nombres de clases similares
                # YOLOv8 base puede detectar 'person' pero no EPP específico
                # Necesitamos inferir EPP basado en características de región
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'area': (x2-x1) * (y2-y1)
                })
        
        # Post-procesamiento: Inferir EPP basado en detecciones de personas
        enhanced_detections = enhance_ppe_detection(img_rgb, detections)
        
        return enhanced_detections, results
    except Exception as e:
        st.error(f"❌ Error en detección: {str(e)}")
        return [], None

def draw_detections(image, detections, confidence_threshold=0.5):
    """Dibuja las detecciones en la imagen"""
    img_array = np.array(image)
    img_draw = img_array.copy()
    
    # Colores para diferentes clases
    colors = {
        'person': (255, 0, 0),      # Rojo
        'worker': (255, 0, 0),      # Rojo
        'helmet': (0, 255, 0),      # Verde
        'hardhat': (0, 255, 0),     # Verde
        'hard-hat': (0, 255, 0),    # Verde
        'safety_vest': (0, 255, 255), # Amarillo/Cyan
        'vest': (0, 255, 255),      # Amarillo/Cyan
        'safety-vest': (0, 255, 255)  # Amarillo/Cyan
    }
    
    for det in detections:
        if det['confidence'] >= confidence_threshold:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            is_inferred = det.get('inferred', False)
            
            color = colors.get(class_name, (255, 255, 0))
            
            # Si es inferido, usar línea punteada (simulada con línea más delgada)
            thickness = 2 if is_inferred else 3
            
            # Dibujar rectángulo
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
            
            # Preparar texto
            label = f"{class_name}: {confidence:.2f}"
            if is_inferred:
                label += " (IA)"
            
            # Fondo para el texto
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_draw, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), color, -1)
            
            # Texto
            cv2.putText(img_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return Image.fromarray(img_draw)

# =============================================
# INICIALIZACIÓN
# =============================================
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

expert_system = SafetyExpertSystem()

# =============================================
# SIDEBAR
# =============================================
st.sidebar.markdown('<div class="config-section">', unsafe_allow_html=True)
st.sidebar.header("⚙️ Configuración del Detector")
confidence_threshold = st.sidebar.slider(
    "Confianza Mínima de Detección", 
    min_value=0.1, 
    max_value=0.95, 
    value=0.4, 
    step=0.05,
    help="Umbral mínimo de confianza para considerar una detección válida. Valor más bajo = más detecciones pero más falsos positivos"
)

show_boxes = st.sidebar.checkbox("Mostrar Bounding Boxes", True)
show_labels = st.sidebar.checkbox("Mostrar Etiquetas", True)

st.sidebar.markdown("---")
st.sidebar.markdown("**🎨 Detección de EPP por Color**")
st.sidebar.info("""
El sistema usa análisis de color para detectar:
- 🪖 **Cascos**: Blanco, amarillo, naranja, rojo, azul
- 🦺 **Chalecos**: Amarillo/naranja fluorescente, verde lima

Las detecciones marcadas con **(IA)** son inferidas por análisis de color.
""")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="config-section">', unsafe_allow_html=True)
st.sidebar.header("📊 Información del Modelo")
model = load_yolo_model()
if model:
    st.sidebar.success("🤖 Modelo YOLO cargado")
    st.sidebar.info(f"📦 Clases detectables: {len(model.names)}")
else:
    st.sidebar.error("❌ Modelo no disponible")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# =============================================
# HEADER PRINCIPAL
# =============================================
st.markdown('<h1 class="main-header">🦺 SafeBuild AI</h1>', unsafe_allow_html=True)
st.markdown("### Sistema Inteligente de Detección de EPP con YOLO")
st.markdown("---")

# =============================================
# CONTENIDO PRINCIPAL
# =============================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Análisis de Imagen con IA")
    
    st.markdown("""
    <div class="info-box">
        <strong>🎯 ¿Cómo funciona?</strong><br>
        1. Sube una imagen de tu obra de construcción<br>
        2. El sistema YOLO detectará automáticamente personas y EPP<br>
        3. El sistema experto evaluará el cumplimiento de seguridad<br>
        4. Recibirás alertas y recomendaciones en tiempo real
    </div>
    """, unsafe_allow_html=True)
    
    # Widget para subir imagen
    uploaded_file = st.file_uploader(
        "📁 Selecciona una imagen de la obra:",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Formatos soportados: JPG, JPEG, PNG, BMP (máx 200MB)"
    )
    
    if uploaded_file is not None:
        # Mostrar información de la imagen
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ **Imagen cargada:** {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        # Cargar imagen
        image = Image.open(uploaded_file)
        original_size = image.size
        
        # Mostrar imagen original
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("**📷 Imagen Original**")
            st.image(image, use_container_width=True)
        
        # Botón para analizar
        if st.button("🔍 Analizar Seguridad con YOLO", use_container_width=True):
            if model is None:
                st.error("❌ No se pudo cargar el modelo YOLO. Por favor, recarga la página.")
            else:
                with st.spinner("🤖 Analizando imagen con YOLO..."):
                    # Barra de progreso
                    progress_bar = st.progress(0)
                    for i in range(30):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Detectar objetos
                    detections, yolo_results = detect_objects(image, model, confidence_threshold)
                    
                    progress_bar.progress(60)
                    
                    # Analizar con sistema experto (pasando tamaño de imagen para contexto)
                    image_array = np.array(image)
                    analysis = expert_system.analyze_detections(
                        detections, 
                        confidence_threshold,
                        image_size=image_array.shape[:2]  # (height, width)
                    )
                    
                    progress_bar.progress(100)
                    time.sleep(0.2)
                    progress_bar.empty()
                
                st.success("✅ Análisis completado")
                
                # Dibujar detecciones si está habilitado
                if show_boxes and detections:
                    annotated_image = draw_detections(image, detections, confidence_threshold)
                    with col_img2:
                        st.markdown("**🎯 Detecciones YOLO**")
                        st.image(annotated_image, use_container_width=True)
                
                # Guardar en historial
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'filename': uploaded_file.name,
                    'detections': len(detections),
                    'alert_level': analysis['alert_level'],
                    'statistics': analysis['statistics'],
                    'compliance_rate': analysis['compliance_rate'],
                    'rule_triggered': analysis.get('rule_triggered', 'default')
                })
                
                # Mostrar información de detecciones
                st.markdown("---")
                st.markdown('<div class="detections-section">', unsafe_allow_html=True)
                st.subheader("🔍 Detecciones Realizadas")
                
                if detections:
                    col_det1, col_det2, col_det3, col_det4, col_det5 = st.columns(5)
                    with col_det1:
                        st.metric("📦 Total Detecciones", len(detections))
                    with col_det2:
                        st.metric("👥 Personas", analysis['statistics']['persons'])
                    with col_det3:
                        st.metric("🪖 Cascos", analysis['statistics']['helmets'])
                    with col_det4:
                        st.metric("🦺 Chalecos", analysis['statistics']['vests'])
                    with col_det5:
                        st.metric("🛡️ EPP Completo", analysis['statistics']['full_ppe'])
                    
                    # Mostrar estadísticas de contexto si existen
                    if analysis['statistics']['persons_high_risk'] > 0:
                        st.info(f"⚠️ **Zona de Altura:** {analysis['statistics']['persons_high_risk']} persona(s) en área de riesgo elevado")
                    
                    # Tabla de detecciones
                    with st.expander("📋 Ver detalle de todas las detecciones"):
                        for i, det in enumerate(detections, 1):
                            if det['confidence'] >= confidence_threshold:
                                st.markdown(f"""
                                <div class="detection-detail-box">
                                    <strong>Detección #{i}</strong><br>
                                    🏷️ Clase: {det['class']}<br>
                                    📊 Confianza: {det['confidence']:.2%}<br>
                                    📍 Ubicación: {det['bbox']}
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ No se detectaron objetos con la confianza mínima establecida")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Mostrar análisis del sistema experto
                st.markdown("---")
                st.markdown('<div class="expert-analysis-section">', unsafe_allow_html=True)
                st.subheader("🧠 Análisis del Sistema Experto")
                
                alert_level = analysis['alert_level']
                
                if alert_level == "ALTA":
                    st.markdown(f"""
                    <div class="alert-high">
                        <div class="alert-title">🚨 ALERTA CRÍTICA DE SEGURIDAD</div>
                        <div class="alert-message">
                            <strong>{analysis['alert_message']}</strong>
                        </div>
                        <div class="alert-action">
                            <strong>📋 Acción Recomendada:</strong><br>
                            {analysis['recommended_action']}
                        </div>
                        <div>
                            <span class="alert-priority">
                                <strong>⏰ Prioridad:</strong> Resolución Inmediata
                            </span>
                            <span class="alert-compliance">
                                <strong>📊 Cumplimiento EPP:</strong> {analysis['compliance_rate']:.1f}%
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif alert_level == "MEDIA":
                    st.markdown(f"""
                    <div class="alert-medium">
                        <div class="alert-title">⚠️ ALERTA DE SEGURIDAD</div>
                        <div class="alert-message">
                            <strong>{analysis['alert_message']}</strong>
                        </div>
                        <div class="alert-action">
                            <strong>📋 Acción Recomendada:</strong><br>
                            {analysis['recommended_action']}
                        </div>
                        <div>
                            <span class="alert-priority">
                                <strong>⏰ Prioridad:</strong> Resolución en 1 hora
                            </span>
                            <span class="alert-compliance">
                                <strong>📊 Cumplimiento EPP:</strong> {analysis['compliance_rate']:.1f}%
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown(f"""
                    <div class="alert-ok">
                        <div class="alert-title">✅ CONDICIONES SEGURAS</div>
                        <div class="alert-message">
                            <strong>{analysis['alert_message']}</strong>
                        </div>
                        <div class="alert-action">
                            <strong>📋 Acción Recomendada:</strong><br>
                            {analysis['recommended_action']}
                        </div>
                        <div>
                            <span class="alert-priority">
                                <strong>⏰ Estado:</strong> Operaciones Normales
                            </span>
                            <span class="alert-compliance">
                                <strong>📊 Cumplimiento EPP:</strong> {analysis['compliance_rate']:.1f}%
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.info("👆 **Sube una imagen para comenzar el análisis de seguridad**")
        st.markdown("""
        <div class="analysis-section">
        <strong>📸 Recomendaciones para mejores resultados:</strong><br>
        - Usa imágenes con buena iluminación<br>
        - Asegúrate que los trabajadores sean visibles<br>
        - Evita imágenes muy borrosas o de baja calidad<br>
        - El modelo detecta: personas, cascos y chalecos reflectantes
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel-section">', unsafe_allow_html=True)
    st.subheader("📊 Panel de Control")
    
    # Mostrar estadísticas actuales
    if 'analysis' in locals() and analysis:
        stats = analysis['statistics']
        compliance = analysis['compliance_rate']
    else:
        stats = {'persons': 0, 'helmets': 0, 'vests': 0, 'total_detections': 0, 'full_ppe': 0, 'persons_high_risk': 0}
        compliance = 0
    
    # Métricas principales
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("👥 Trabajadores Detectados", stats['persons'])
    st.metric("🪖 Cascos Detectados", stats['helmets'])
    st.metric("🦺 Chalecos Detectados", stats['vests'])
    st.metric("🛡️ EPP Completo", stats['full_ppe'])
    st.metric("📈 Cumplimiento EPP", f"{compliance:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Estado actual
    st.subheader("🚦 Estado Actual")
    if stats['persons'] > 0:
        if stats['helmets'] < stats['persons']:
            missing_helmets = stats['persons'] - stats['helmets']
            st.error(f"❌ {missing_helmets} trabajador(es) sin casco")
        else:
            st.success("✅ Todos con casco")
        
        if stats['vests'] < stats['persons']:
            missing_vests = stats['persons'] - stats['vests']
            st.warning(f"⚠️ {missing_vests} trabajador(es) sin chaleco")
        else:
            st.success("✅ Todos con chaleco")
            
        if stats['full_ppe'] == 0:
            st.error("❌ Ningún trabajador con EPP completo")
        else:
            st.success(f"✅ {stats['full_ppe']} trabajador(es) con EPP completo")
            
        if stats['persons_high_risk'] > 0:
            st.warning(f"⚠️ {stats['persons_high_risk']} persona(s) en zona de altura")
    else:
        st.info("👀 No hay trabajadores detectados")
    
    # Historial de análisis
    st.subheader("📋 Historial Reciente")
    if st.session_state.analysis_history:
        for i, record in enumerate(reversed(st.session_state.analysis_history[-5:]), 1):
            status_emoji = "🚨" if record['alert_level'] == "ALTA" else "⚠️" if record['alert_level'] == "MEDIA" else "✅"
            st.markdown(f"""
            <div class="historial-box">
                {status_emoji} <strong>Análisis #{len(st.session_state.analysis_history) - i + 1}</strong><br>
                📸 {record['filename'][:20]}...<br>
                🕐 {record['timestamp'].strftime('%H:%M:%S')}<br>
                👥 {record['statistics']['persons']} personas
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📝 Aún no hay análisis realizados")
    
    # Botón para limpiar historial
    if st.session_state.analysis_history:
        if st.button("🗑️ Limpiar Historial"):
            st.session_state.analysis_history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================
# SECCIÓN DE EXPORTACIÓN DE DATOS
# =============================================
st.markdown("---")
st.markdown('<div class="export-section">', unsafe_allow_html=True)
st.subheader("📤 Exportar Datos de Análisis")

if st.session_state.analysis_history:
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        st.markdown("### 📊 Exportar a Excel")
        st.markdown("""
        Descarga todos los análisis en un archivo Excel con:
        - 📋 **Análisis Detallado**: Todos los análisis realizados
        - 📈 **Estadísticas Resumen**: Métricas consolidadas
        """)
        
        excel_data = export_to_excel()
        st.download_button(
            label="📥 Descargar Excel (.xlsx)",
            data=excel_data,
            file_name=f"safebuild_analisis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col_export2:
        st.markdown("### 📄 Exportar a CSV")
        st.markdown("""
        Descarga los datos en formato CSV:
        - 🗂️ **Archivo único**: Todos los análisis
        - 💾 **Formato universal**: Compatible con cualquier software
        """)
        
        df_analysis, df_summary = generate_export_data()
        csv_data = df_analysis.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Descargar CSV (.csv)",
            data=csv_data,
            file_name=f"safebuild_analisis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_export3:
        st.markdown("### 📋 Vista Previa de Datos")
        st.markdown(f"""
        **Resumen de exportación:**
        - 📊 **Total de análisis:** {len(st.session_state.analysis_history)}
        - 🚨 **Alertas críticas:** {sum(1 for r in st.session_state.analysis_history if r['alert_level'] == 'ALTA')}
        - ✅ **Condiciones seguras:** {sum(1 for r in st.session_state.analysis_history if r['alert_level'] == 'OK')}
        - 👥 **Personas analizadas:** {sum(r['statistics']['persons'] for r in st.session_state.analysis_history)}
        """)
        
        # Mostrar vista previa de los datos
        with st.expander("👀 Ver vista previa de datos"):
            st.dataframe(df_analysis.head(10), use_container_width=True)
    
    # Información adicional sobre la exportación
    st.info("""
    💡 **Nota:** Los archivos exportados incluyen todos los análisis realizados en esta sesión, 
    con timestamps, niveles de alerta, estadísticas detalladas y tasas de cumplimiento de EPP.
    """)
    
else:
    st.warning("📭 No hay datos para exportar. Realiza al menos un análisis para habilitar la exportación.")
    st.info("👆 Sube una imagen y haz clic en 'Analizar Seguridad con YOLO' para comenzar.")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================
# ESTADÍSTICAS GLOBALES
# =============================================
st.markdown("---")
st.markdown('<div class="stats-section">', unsafe_allow_html=True)
st.subheader("📈 Estadísticas de la Sesión")

col3, col4, col5, col6 = st.columns(4)

total_analyses = len(st.session_state.analysis_history)
total_alerts = sum(1 for r in st.session_state.analysis_history if r['alert_level'] in ['ALTA', 'MEDIA'])
avg_persons = np.mean([r['statistics']['persons'] for r in st.session_state.analysis_history]) if st.session_state.analysis_history else 0
avg_compliance = np.mean([r.get('compliance_rate', 0) for r in st.session_state.analysis_history]) if st.session_state.analysis_history else 0

with col3:
    st.metric("🔍 Análisis Realizados", total_analyses)
with col4:
    st.metric("🚨 Alertas Generadas", total_alerts)
with col5:
    st.metric("👥 Promedio Trabajadores", f"{avg_persons:.1f}")
with col6:
    st.metric("📊 Cumplimiento Promedio", f"{avg_compliance:.1f}%")
st.markdown('</div>', unsafe_allow_html=True)

# =============================================
# FOOTER E INFORMACIÓN
# =============================================
st.markdown("---")
st.sidebar.markdown('<div class="config-section">', unsafe_allow_html=True)
st.sidebar.subheader("ℹ️ Acerca de SafeBuild AI")
st.sidebar.info("""
**SafeBuild AI v2.0**  

🤖 **Tecnología:**  
• YOLOv8 para detección de objetos
• Sistema Experto basado en reglas
• Análisis en tiempo real

🎯 **Detecta:**  
• Trabajadores (personas)
• Cascos de seguridad
• Chalecos reflectantes

📊 **Características:**  
• Análisis automático de cumplimiento
• Alertas por niveles de riesgo
• Historial de análisis
• Exportación de datos
• Métricas en tiempo real

🎓 **Desarrollo:**  
Trabajo Práctico Integrador  
Sistemas de Inteligencia Artificial
""")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>SafeBuild AI v2.0</strong> - Sistema de Detección de EPP con YOLO</p>
    <p>🤖 Powered by YOLOv8 + Sistema Experto 🤖</p>
    <p style="font-size: 0.9rem;">Desarrollado como TP Integrador - IA</p>
</div>
""", unsafe_allow_html=True)
