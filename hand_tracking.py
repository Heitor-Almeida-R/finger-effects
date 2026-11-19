import cv2
import mediapipe as mp
import numpy as np

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

detector = HandLandmarker.create_from_options(options)


camera = cv2.VideoCapture(0)

tempo_inicial = cv2.getTickCount()


efeito_atual = 0


gesto_mudar_anterior = False


gesto_print_anterior = False

printando = False

tempo_inicio_print = 0

efeito_do_print = 0

DURACAO_PRINT = 1000


efeitos_permanentes = []


palmas_anteriormente_proximas = False

distancia_palmas_anterior = None

tempo_ultima_palmas = 0

INTERVALO_PALMAS = 800


def efeito_normal(frame):

    return frame.copy()


def efeito_preto_branco(frame):
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cinza = cv2.convertScaleAbs(
        cinza,
        alpha=1.15,
        beta=-5
    )

    resultado = cv2.cvtColor(
        cinza,
        cv2.COLOR_GRAY2BGR
    )

    altura, largura = cinza.shape
    tamanho_celula = 8

    bolinhas = np.zeros_like(
        cinza,
        dtype=np.uint8
    )

    for y in range(0, altura, tamanho_celula):
        for x in range(0, largura, tamanho_celula):
            y2 = min(y + tamanho_celula, altura)
            x2 = min(x + tamanho_celula, largura)

            bloco = cinza[y:y2, x:x2]

            if bloco.size == 0:
                continue

            brilho = np.mean(bloco)
            escuridao = 255 - brilho
            raio_maximo = 2

            raio = int(
                (escuridao / 255) * raio_maximo
            )

            centro_x = (x + x2) // 2
            centro_y = (y + y2) // 2

            if raio > 0:
                cv2.circle(
                    bolinhas,
                    (centro_x, centro_y),
                    raio,
                    255,
                    -1,
                    cv2.LINE_AA
                )

    resultado[bolinhas > 0] = 0

    return resultado

def efeito_borrado(frame):

    resultado = cv2.GaussianBlur(
        frame,
        (31, 31),
        0
    )

    return resultado


def efeito_invertido(frame):

    resultado = cv2.bitwise_not(
        frame
    )

    return resultado


def efeito_pixelado(frame):

    altura, largura = frame.shape[:2]


    pequeno = cv2.resize(
        frame,
        (
            max(1, largura // 20),
            max(1, altura // 20)
        ),
        interpolation=cv2.INTER_LINEAR
    )


    resultado = cv2.resize(
        pequeno,
        (
            largura,
            altura
        ),
        interpolation=cv2.INTER_NEAREST
    )


    return resultado


def aplicar_efeito(
    frame,
    mascara,
    efeito
):


    if efeito == 0:

        resultado = efeito_normal(
            frame
        )


    elif efeito == 1:

        resultado = efeito_preto_branco(
            frame
        )


    elif efeito == 2:

        resultado = efeito_borrado(
            frame
        )


    elif efeito == 3:

        resultado = efeito_invertido(
            frame
        )


    elif efeito == 4:

        resultado = efeito_pixelado(
            frame
        )


    else:

        resultado = frame.copy()


    resultado_final = np.where(
        mascara[:, :, np.newaxis] == 255,
        resultado,
        frame
    )


    return resultado_final


def desenhar_borda_quadrinho(
    frame,
    pontos,
    espessura=2
):

    cv2.polylines(
        frame,
        [pontos],
        True,
        (0, 0, 0),
        espessura,
        cv2.LINE_AA
    )


while True:

    sucesso, frame = camera.read()


    if not sucesso:

        print("Não achou")

        break

    frame = cv2.flip(
        frame,
        1
    )

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    imagem = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )


    tempo_atual = cv2.getTickCount()


    timestamp = int(
        (
            tempo_atual
            - tempo_inicial
        )
        / cv2.getTickFrequency()
        * 1000
    )


    resultado = detector.detect_for_video(
        imagem,
        timestamp
    )


    mao_direita = None
    mao_esquerda = None


    if resultado.hand_landmarks:

        for i, mao in enumerate(
            resultado.hand_landmarks
        ):

            indicador = mao[8]

            polegar = mao[4]

            pulso = mao[0]

            palma = mao[9]


            altura, largura, _ = frame.shape


            ponto_indicador = (
                int(indicador.x * largura),
                int(indicador.y * altura)
            )


            ponto_polegar = (
                int(polegar.x * largura),
                int(polegar.y * altura)
            )


            ponto_pulso = (
                int(pulso.x * largura),
                int(pulso.y * altura)
            )


            ponto_palma = (
                int(palma.x * largura),
                int(palma.y * altura)
            )


            if resultado.handedness:

                tipo_mao = (
                    resultado
                    .handedness[i][0]
                    .category_name
                )


                if tipo_mao == "Right":

                    mao_direita = {
                        "indicador": ponto_indicador,
                        "polegar": ponto_polegar,
                        "pulso": ponto_pulso,
                        "palma": ponto_palma
                    }


                elif tipo_mao == "Left":

                    mao_esquerda = {
                        "indicador": ponto_indicador,
                        "polegar": ponto_polegar,
                        "pulso": ponto_pulso,
                        "palma": ponto_palma
                    }


    duas_maos = (
        mao_direita is not None
        and
        mao_esquerda is not None
    )

    if duas_maos:

        indicador_direito = (
            mao_direita["indicador"]
        )

        polegar_direito = (
            mao_direita["polegar"]
        )

        indicador_esquerdo = (
            mao_esquerda["indicador"]
        )

        polegar_esquerdo = (
            mao_esquerda["polegar"]
        )


        distancia_indicadores = np.linalg.norm(
            np.array(indicador_direito)
            -
            np.array(indicador_esquerdo)
        )


        distancia_polegares = np.linalg.norm(
            np.array(polegar_direito)
            -
            np.array(polegar_esquerdo)
        )


        distancia_pinch_direita = np.linalg.norm(
            np.array(indicador_direito)
            -
            np.array(polegar_direito)
        )


        distancia_pinch_esquerda = np.linalg.norm(
            np.array(indicador_esquerdo)
            -
            np.array(polegar_esquerdo)
        )


        centro_direita = np.array(
            mao_direita["palma"]
        )


        centro_esquerda = np.array(
            mao_esquerda["palma"]
        )


        distancia_palmas = np.linalg.norm(
            centro_direita
            -
            centro_esquerda
        )


        gesto_mudar = (
            distancia_indicadores < 70
            and
            distancia_polegares < 70
        )


        gesto_print = (
            distancia_pinch_direita < 10
            and
            distancia_pinch_esquerda < 10
            and
            distancia_palmas > 55
        )


        if gesto_mudar:

            if not gesto_mudar_anterior:

                efeito_atual += 1


                if efeito_atual > 4:

                    efeito_atual = 0


        gesto_mudar_anterior = gesto_mudar


        if gesto_print and not gesto_mudar:

            if (
                not gesto_print_anterior
                and
                not printando
            ):

                printando = True

                tempo_inicio_print = (
                    cv2.getTickCount()
                )

                efeito_do_print = (
                    efeito_atual
                )


        gesto_print_anterior = gesto_print

        pontos_area = np.array(
            [
                indicador_esquerdo,
                indicador_direito,
                polegar_direito,
                polegar_esquerdo
            ],
            dtype=np.int32
        )


        mascara = np.zeros(
            frame.shape[:2],
            dtype=np.uint8
        )


        cv2.fillPoly(
            mascara,
            [pontos_area],
            255
        )


        if efeito_atual != 0:

            frame = aplicar_efeito(
                frame,
                mascara,
                efeito_atual
            )


        if efeito_atual != 0:

            desenhar_borda_quadrinho(
                frame,
                pontos_area,
                2
            )


        if printando:

            tempo_agora = (
                cv2.getTickCount()
            )


            tempo_passado = (
                tempo_agora
                -
                tempo_inicio_print
            ) / cv2.getTickFrequency() * 1000


            if tempo_passado >= DURACAO_PRINT:

                nova_mascara = np.zeros(
                    frame.shape[:2],
                    dtype=np.uint8
                )


                cv2.fillPoly(
                    nova_mascara,
                    [pontos_area],
                    255
                )


                efeitos_permanentes.append(
                    (
                        efeito_do_print,
                        nova_mascara.copy()
                    )
                )


                printando = False


    else:

        gesto_mudar_anterior = False

        gesto_print_anterior = False


    for efeito, mascara_permanente in efeitos_permanentes:

        frame = aplicar_efeito(
            frame,
            mascara_permanente,
            efeito
        )


        contornos, _ = cv2.findContours(
            mascara_permanente,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            frame,
            contornos,
            -1,
            (0, 0, 0),
            2
        )


    if duas_maos:


        palmas_proximas = (
            distancia_palmas < 90
        )


        aproximando = False


        if distancia_palmas_anterior is not None:

            aproximando = (
                distancia_palmas
                <
                distancia_palmas_anterior
            )


        bateu_palmas = (
            palmas_proximas
            and
            not palmas_anteriormente_proximas
            and
            aproximando
        )


        tempo_agora = cv2.getTickCount()


        tempo_desde_palmas = (
            tempo_agora
            -
            tempo_ultima_palmas
        ) / cv2.getTickFrequency() * 1000


        if (
            bateu_palmas
            and
            tempo_desde_palmas > INTERVALO_PALMAS
        ):


            efeitos_permanentes.clear()


            tempo_ultima_palmas = (
                tempo_agora
            )


            printando = False


        distancia_palmas_anterior = (
            distancia_palmas
        )


        palmas_anteriormente_proximas = (
            palmas_proximas
        )


    else:

        distancia_palmas_anterior = None

        palmas_anteriormente_proximas = False


    pontos_maos = []

    if mao_direita is not None:
        pontos_maos.extend([
            mao_direita["indicador"],
            mao_direita["polegar"],
            mao_direita["palma"]
        ])

    if mao_esquerda is not None:
        pontos_maos.extend([
            mao_esquerda["indicador"],
            mao_esquerda["polegar"],
            mao_esquerda["palma"]
        ])

    for ponto in pontos_maos:
        cv2.circle(
            frame,
            ponto,
            2,
            (0, 255, 0),
            -1,
            cv2.LINE_AA
        )

    cv2.imshow(
        "Minha Webcam",
        frame
    )


    if cv2.waitKey(1) == ord("q"):

        break


camera.release()

cv2.destroyAllWindows()
