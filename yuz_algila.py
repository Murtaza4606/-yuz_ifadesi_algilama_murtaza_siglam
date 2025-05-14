import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Etiket: toplamak istediğiniz ifade adı
etiket = "happy"  # Her ifade için bu satırı değiştir (sad, angry, surprised)

# Sadece ilk ifade için başlıklar yazılır
if etiket == "happy":
    with open("veriseti.csv", "w") as f:
        baslik = ",".join([f"x{i},y{i}" for i in range(1, 479)]) + ",Etiket\n"
        f.write(baslik)

# Model ayarları
base = python.BaseOptions(model_asset_path="face_landmarker_v2_with_blendshapes.task")
secenek = vision.FaceLandmarkerOptions(
    base_options=base,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
tanimlayici = vision.FaceLandmarker.create_from_options(secenek)

kamera = cv2.VideoCapture(0)
sayac = 0

while kamera.isOpened():
    ret, kare = kamera.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
    mp_goruntu = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    sonuc = tanimlayici.detect(mp_goruntu)

    if sonuc.face_landmarks:
        landmark = sonuc.face_landmarks[0]
        satir = ""
        for nokta in landmark:
            satir += f"{round(nokta.x, 4)},{round(nokta.y, 4)},"
        satir += f"{etiket}\n"

        with open("veriseti.csv", "a") as f:
            f.write(satir)

        sayac += 1
        print(f"{sayac} veri eklendi ({etiket})")

    cv2.putText(kare, f"Ifade: {etiket.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Kamera", kare)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
