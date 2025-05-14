import cv2
import mediapipe as mp
import pickle
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model dosyasını yükle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Model ayarları
base = python.BaseOptions(model_asset_path="face_landmarker_v2_with_blendshapes.task")
ayarlar = vision.FaceLandmarkerOptions(
    base_options=base,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
sistem = vision.FaceLandmarker.create_from_options(ayarlar)

kamera = cv2.VideoCapture(0)

while kamera.isOpened():
    okundu, kare = kamera.read()
    if not okundu:
        continue

    rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
    mp_goruntu = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    sonuc = sistem.detect(mp_goruntu)

    if sonuc.face_landmarks:
        landmark = sonuc.face_landmarks[0]
        veri = []
        for n in landmark:
            veri.extend([round(n.x, 4), round(n.y, 4)])
        
        # Feature name hatasına karşı çözüm
        sutunlar = [f"f{i}" for i in range(956)]
        veri_df = pd.DataFrame([veri], columns=sutunlar)

        try:
            tahmin = model.predict(veri_df)[0]
            cv2.putText(kare, f"Tahmin: {tahmin.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        except Exception as e:
            print("Tahmin hatası:", e)

    cv2.imshow("Ifade Tahmini", kare)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
