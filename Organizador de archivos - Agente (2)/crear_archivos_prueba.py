"""
Script para crear archivos de prueba en una carpeta 'desordenada'.
Estos archivos servirán para probar el agente organizador.
"""
import os

# Nombre de la carpeta de prueba
CARPETA = "desordenada"

# Archivos de prueba con diferentes tipos
ARCHIVOS = [
    # Documentos
    "informe_final_proyecto.pdf",
    "tarea_matematicas_semana4.docx",
    "notas_de_clase_fisica.txt",
    "presentacion_grupo.pptx",
    "curriculum_vitae_2026.pdf",
    # Imágenes
    "foto_vacaciones_playa.jpg",
    "screenshot_error_codigo.png",
    "diagrama_circuito_motor.png",
    "selfie_graduacion.jpeg",
    # Código
    "mi_primer_script.py",
    "utilidades_matematicas.py",
    "pagina_portafolio.html",
    "estilos_web.css",
    # Audio y Video
    "podcast_tecnologia_ep5.mp3",
    "tutorial_python_basico.mp4",
    "grabacion_clase_virtual.wav",
    # Datos
    "datos_ventas_marzo.csv",
    "respuestas_encuesta.xlsx",
    "configuracion_servidor.json",
    # Comprimidos
    "backup_proyecto_final.zip",
    "recursos_diseno.rar",
    # Otros
    "readme_instrucciones",
    "notas_rapidas",
]

def crear_archivos():
    """Crea la carpeta y los archivos de prueba."""
    os.makedirs(CARPETA, exist_ok=True)
    
    for archivo in ARCHIVOS:
        ruta = os.path.join(CARPETA, archivo)
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(f"Archivo de prueba: {archivo}\n")
            f.write("Este archivo fue creado para probar el agente organizador.\n")
    
    print(f"✅ Se crearon {len(ARCHIVOS)} archivos en la carpeta '{CARPETA}/'")
    print(f"\nArchivos creados:")
    for archivo in sorted(ARCHIVOS):
        print(f"   📄 {archivo}")

if __name__ == "__main__":
    crear_archivos()