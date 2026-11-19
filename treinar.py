import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time


SELOS = [
    "cobra",
    "carneiro",
    "macaco",
    "javali",
    "cavalo",
    "tigre",
    "cao",
    "passaro",
    "boi",
    "coelho",
    "rato",
    "dragao"
]


class ColetorSelos:
    def __init__(self, modelo):
        if not os.path.exists(modelo):
            raise FileNotFoundError(
                "\nO arquivo hand_landmarker.task nao foi encontrado.\n"
                "Coloque ele na mesma pasta deste programa.\n"
            )

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        opcoes = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=modelo
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = HandLandmarker.create_from_options(
            opcoes
        )

        self.timestamp = 0

    def detectar(self, frame):
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        imagem = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        self.timestamp += 33

        resultado = self.landmarker.detect_for_video(
            imagem,
            self.timestamp
        )

        if not resultado.hand_landmarks:
            return None, frame

        maos = resultado.hand_landmarks

        for mao in maos:
            self.desenhar_mao(
                frame,
                mao
            )

        if len(maos) != 2:
            return None, frame

        pontos = self.criar_caracteristicas(
            maos
        )

        return pontos, frame

    def criar_caracteristicas(self, maos):
        mao_a = maos[0]
        mao_b = maos[1]

        centro_a = np.array([
            mao_a[0].x,
            mao_a[0].y,
            mao_a[0].z
        ])

        centro_b = np.array([
            mao_b[0].x,
            mao_b[0].y,
            mao_b[0].z
        ])

        caracteristicas = []

        for mao, centro in [
            (mao_a, centro_a),
            (mao_b, centro_b)
        ]:
            for ponto in mao:
                caracteristicas.extend([
                    ponto.x - centro[0],
                    ponto.y - centro[1],
                    ponto.z - centro[2]
                ])

        distancia = np.linalg.norm(
            centro_a - centro_b
        )

        caracteristicas.append(
            distancia
        )

        return caracteristicas

    def desenhar_mao(self, frame, pontos):
        conexoes = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (5, 9),
            (9, 13),
            (13, 17)
        ]

        for ponto in pontos:
            x = int(
                ponto.x * frame.shape[1]
            )

            y = int(
                ponto.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        for a, b in conexoes:
            x1 = int(
                pontos[a].x * frame.shape[1]
            )

            y1 = int(
                pontos[a].y * frame.shape[0]
            )

            x2 = int(
                pontos[b].x * frame.shape[1]
            )

            y2 = int(
                pontos[b].y * frame.shape[0]
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                2
            )

    def fechar(self):
        self.landmarker.close()


def salvar_dataset(dados, arquivo):
    novo_arquivo = not os.path.exists(
        arquivo
    )

    with open(
        arquivo,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:
        escritor = csv.writer(f)

        if novo_arquivo:
            cabecalho = [
                f"x{i}"
                for i in range(
                    len(dados[0]) - 1
                )
            ]

            cabecalho.append("classe")

            escritor.writerow(
                cabecalho
            )

        escritor.writerows(dados)


def coletar_selo(
    coletor,
    camera,
    nome_selo,
    numero_amostras=150
):
    dados = []

    print()
    print("=" * 50)
    print(f"SELO: {nome_selo.upper()}")
    print("=" * 50)
    print()
    print(
        "Posicione as DUAS MAOS "
        "exatamente como o selo."
    )
    print()
    print(
        "Pressione ESPACO para comecar."
    )
    print(
        "Pressione ESC para cancelar."
    )

    while True:
        sucesso, frame = camera.read()

        if not sucesso:
            continue

        frame = cv2.flip(
            frame,
            1
        )

        cv2.putText(
            frame,
            f"SELO: {nome_selo.upper()}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "ESPACO = iniciar",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Treinamento dos Selos",
            frame
        )

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 32:
            break

        if tecla == 27:
            return False

    print()
    print(
        f"Preparando captura de {nome_selo}..."
    )

    inicio = time.time()

    while time.time() - inicio < 2:
        sucesso, frame = camera.read()

        if not sucesso:
            continue

        frame = cv2.flip(
            frame,
            1
        )

        cv2.putText(
            frame,
            "PREPARE-SE...",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Treinamento dos Selos",
            frame
        )

        cv2.waitKey(1)

    print(
        f"Capturando {numero_amostras} amostras..."
    )

    while len(dados) < numero_amostras:
        sucesso, frame = camera.read()

        if not sucesso:
            continue

        frame = cv2.flip(
            frame,
            1
        )

        caracteristicas, frame = coletor.detectar(
            frame
        )

        progresso = int(
            len(dados) /
            numero_amostras *
            100
        )

        cv2.rectangle(
            frame,
            (30, 100),
            (530, 135),
            (50, 50, 50),
            -1
        )

        cv2.rectangle(
            frame,
            (30, 100),
            (
                30 + int(
                    500 *
                    len(dados) /
                    numero_amostras
                ),
                135
            ),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            f"{progresso}%",
            (30, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Amostras: {len(dados)}/{numero_amostras}",
            (30, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if caracteristicas is not None:
            linha = caracteristicas.copy()
            linha.append(nome_selo)

            dados.append(
                linha
            )

            cv2.putText(
                frame,
                "DUAS MAOS DETECTADAS",
                (30, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "MOSTRE AS DUAS MAOS",
                (30, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Treinamento dos Selos",
            frame
        )

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:
            return False

    print(
        f"{nome_selo.upper()} concluido!"
    )

    return dados


def main():
    pasta = os.path.dirname(
        os.path.abspath(__file__)
    )

    modelo = os.path.join(
        pasta,
        "hand_landmarker.task"
    )

    arquivo_dataset = os.path.join(
        pasta,
        "dataset.csv"
    )

    coletor = ColetorSelos(
        modelo
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print(
            "Nao foi possivel abrir a webcam."
        )

        coletor.fechar()
        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    print()
    print("=" * 60)
    print("TREINAMENTO DOS SELOS DE NARUTO")
    print("=" * 60)
    print()
    print(
        "Vamos ensinar os 12 selos ao Machine Learning."
    )
    print()
    print(
        "Cada selo tera 150 amostras."
    )
    print(
        "Total esperado: 1800 amostras."
    )
    print()

    try:
        for i, selo in enumerate(SELOS):
            print()
            print(
                f"[{i + 1}/12] "
                f"Proximo selo: {selo.upper()}"
            )

            dados = coletar_selo(
                coletor,
                camera,
                selo,
                150
            )

            if dados is False:
                print(
                    "Treinamento cancelado."
                )
                break

            salvar_dataset(
                dados,
                arquivo_dataset
            )

            print(
                f"{len(dados)} amostras "
                f"salvas para {selo}."
            )

            if i < len(SELOS) - 1:
                print()
                print(
                    "Pressione qualquer tecla "
                    "para continuar."
                )

                cv2.waitKey(0)

        print()
        print("=" * 60)
        print("TREINAMENTO FINALIZADO!")
        print("=" * 60)
        print()
        print(
            f"Dataset salvo em:\n{arquivo_dataset}"
        )

    finally:
        coletor.fechar()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()