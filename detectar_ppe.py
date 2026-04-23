from ultralytics import YOLO
import cv2

MODEL_PATH = "C:/ppe_dataset/runs/entrenamiento_ppe/weights/best.pt"
CONFIANZA  = 0.5

print("🦺 Iniciando detección de PPE...")
print("   Presiona Q para salir\n")

model = YOLO(MODEL_PATH)

results = model.predict(
    source=0,
    show=True,
    conf=CONFIANZA,
    device=0,
    stream=True
)

for r in results:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("✅ Detección finalizada")