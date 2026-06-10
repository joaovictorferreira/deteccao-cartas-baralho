from ultralytics import YOLO
import cv2
import os

# Definição do modelo
MODEL_PATH = "best.pt"

# Caminho para o vídeo a ser aplicado o modelo
VIDEO_INPUT = "C:/Users/João/Documents/faculdade/cd/AV3/modelo-cartas/videos/s_vs_c.mp4"

# Resultado gravado e armazenado
OUTPUT_DIR = "resultados"
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "video_cartas_detectado.mp4")

#Nível de confiança escolhido para o modelo 
CONFIDENCE = 0.40

#Tamanho do display do vídeo gerado
DISPLAY_SCALE = 0.5


#Carrega o modelo
os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)


# Configurações para abrir o vídeo do modelo sendo aplicado

cap = cv2.VideoCapture(VIDEO_INPUT)

if not cap.isOpened():
    raise FileNotFoundError(f"Não foi possível abrir o vídeo: {VIDEO_INPUT}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Gravar no formato .mp4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Processamento dos frames do vídeo
while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        verbose=False
    )

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    # Redimencionamento do vídeo para exibição
    frame_display = cv2.resize(
        annotated_frame,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE
    )

    cv2.imshow("Deteccao de Cartas", frame_display)

    # Adicionada a opção de parar o vídeo com a tecla Q caso não queira assistir ate o final
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Terminar a execução
cap.release()
out.release()
cv2.destroyAllWindows()