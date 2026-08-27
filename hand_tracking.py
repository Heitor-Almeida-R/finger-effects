import cv2
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker

base_options = mp.tasks.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

tempo_inicial = cv2.getTickCount()

while True:
    sucesso, frame = camera.read()

    if not sucesso:
        print("Não achou")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    imagem = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    tempo_atual = cv2.getTickCount()

    timestamp = int(
        (tempo_atual - tempo_inicial) / cv2.getTickFrequency() * 1000
    )

    resultado = detector.detect_for_video(imagem, timestamp)

    pontos = []

    if resultado.hand_landmarks:
        for mao in resultado.hand_landmarks:

            indicador = mao[8]
            polegar = mao[4]

            altura, largura, _ = frame.shape

            x_indicador = int(indicador.x * largura)
            y_indicador = int(indicador.y * altura)

            x_polegar = int(polegar.x * largura)
            y_polegar = int(polegar.y * altura)

            pontos.append((x_indicador, y_indicador))
            pontos.append((x_polegar, y_polegar))

            cv2.circle(
                frame,
                (x_indicador, y_indicador),
                10,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                frame,
                (x_polegar, y_polegar),
                10,
                (0, 255, 0),
                -1
            )

        if len(pontos) == 4:
            cv2.line(frame, pontos[0], pontos[2], (0, 255, 0), 2)
            cv2.line(frame, pontos[2], pontos[3], (0, 255, 0), 2)
            cv2.line(frame, pontos[3], pontos[1], (0, 255, 0), 2)
            cv2.line(frame, pontos[1], pontos[0], (0, 255, 0), 2)

    cv2.imshow("Minha Webcam", frame)

    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()