import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

from jutsos import identificar_jutsu


MODELO = "modelo_selos.pkl"
MODELO_MAOS = "hand_landmarker.task"

TEMPO_CONFIRMACAO = 0.5


with open(MODELO, "rb") as arquivo:
    modelo = pickle.load(arquivo)


X_TREINO = modelo["X"]
Y_TREINO = modelo["y"]
K = modelo["k"]


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class PilhaSelos:

    def __init__(self):
        self.itens = []

    def empilhar(self, selo):
        self.itens.append(selo)

    def desempilhar(self):
        if not self.esta_vazia():
            return self.itens.pop()

        return None

    def topo(self):
        if not self.esta_vazia():
            return self.itens[-1]

        return None

    def esta_vazia(self):
        return len(self.itens) == 0

    def tamanho(self):
        return len(self.itens)

    def limpar(self):
        self.itens.clear()

    def mostrar(self):
        return self.itens.copy()


def criar_caracteristicas(maos):

    caracteristicas = []

    for mao in maos:

        pulso = mao[0]

        for ponto in mao:

            x = ponto.x - pulso.x
            y = ponto.y - pulso.y
            z = ponto.z - pulso.z

            caracteristicas.extend([x, y, z])

    pulso1 = maos[0][0]
    pulso2 = maos[1][0]

    distancia = np.sqrt(
        (pulso1.x - pulso2.x) ** 2 +
        (pulso1.y - pulso2.y) ** 2 +
        (pulso1.z - pulso2.z) ** 2
    )

    caracteristicas.append(distancia)

    return np.array(caracteristicas, dtype=np.float32)


def prever(amostra):

    diferencas = X_TREINO - amostra

    distancias = np.sqrt(
        np.sum(diferencas ** 2, axis=1)
    )

    indices = np.argsort(distancias)[:K]

    classes = Y_TREINO[indices]

    valores, contagens = np.unique(
        classes,
        return_counts=True
    )

    indice = np.argmax(contagens)

    return valores[indice]


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODELO_MAOS
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


pilha = PilhaSelos()


selo_atual = None
inicio_selo = None
ultimo_selo_adicionado = None

jutsu_atual = None
tempo_jutsu = 0

timestamp = 0


camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


with HandLandmarker.create_from_options(options) as detector:

    while True:

        sucesso, frame = camera.read()

        if not sucesso:
            break


        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        imagem = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        timestamp += 33

        resultado = detector.detect_for_video(
            imagem,
            timestamp
        )


        selo_reconhecido = None


        if resultado.hand_landmarks and len(resultado.hand_landmarks) == 2:

            maos = resultado.hand_landmarks

            caracteristicas = criar_caracteristicas(maos)

            selo_reconhecido = prever(caracteristicas)


            for mao in maos:

                for ponto in mao:

                    x = int(ponto.x * frame.shape[1])
                    y = int(ponto.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )


            cv2.putText(
                frame,
                f"Selo: {selo_reconhecido}",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            agora = time.time()


            if selo_reconhecido != selo_atual:

                selo_atual = selo_reconhecido
                inicio_selo = agora


            else:

                if inicio_selo is not None:

                    tempo_estavel = agora - inicio_selo

                    progresso = min(
                        tempo_estavel / TEMPO_CONFIRMACAO,
                        1.0
                    )

                    largura = int(300 * progresso)


                    cv2.rectangle(
                        frame,
                        (40, 70),
                        (340, 95),
                        (40, 40, 40),
                        -1
                    )

                    cv2.rectangle(
                        frame,
                        (40, 70),
                        (40 + largura, 95),
                        (0, 255, 0),
                        -1
                    )


                    if (
                        tempo_estavel >= TEMPO_CONFIRMACAO
                        and selo_atual != ultimo_selo_adicionado
                    ):

                        pilha.empilhar(selo_atual)

                        ultimo_selo_adicionado = selo_atual

                        print(
                            "Selo adicionado:",
                            selo_atual
                        )

                        print(
                            "Pilha:",
                            pilha.mostrar()
                        )


                        jutsu = identificar_jutsu(
                            pilha.mostrar()
                        )


                        if jutsu is not None:

                            jutsu_atual = jutsu
                            tempo_jutsu = time.time()

                            print()
                            print(
                                "🔥🔥🔥 JUTSU ENCONTRADO 🔥🔥🔥"
                            )
                            print(
                                "Jutsu:",
                                jutsu
                            )
                            print(
                                "Sequência:",
                                pilha.mostrar()
                            )
                            print()


                            pilha.limpar()

                            selo_atual = None
                            inicio_selo = None
                            ultimo_selo_adicionado = None


        else:

            selo_atual = None
            inicio_selo = None


        if jutsu_atual is not None:

            tempo_passado = time.time() - tempo_jutsu

            if tempo_passado < 3:

                texto = f"JUTSU: {jutsu_atual}"

                tamanho = cv2.getTextSize(
                    texto,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    4
                )[0]

                x = (frame.shape[1] - tamanho[0]) // 2


                cv2.rectangle(
                    frame,
                    (x - 20, 80),
                    (x + tamanho[0] + 20, 140),
                    (0, 0, 0),
                    -1
                )


                cv2.putText(
                    frame,
                    texto,
                    (x, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 255),
                    4
                )

            else:

                jutsu_atual = None


        cv2.rectangle(
            frame,
            (20, 570),
            (600, 700),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            "PILHA",
            (40, 605),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        pilha_visual = pilha.mostrar()


        if pilha_visual:

            texto_pilha = " -> ".join(
                pilha_visual
            )

        else:

            texto_pilha = "vazia"


        cv2.putText(
            frame,
            texto_pilha,
            (40, 645),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            "SPACE = limpar | BACKSPACE = remover | ESC = sair",
            (40, 680),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )


        cv2.imshow(
            "Naruto Jutsu",
            frame
        )


        tecla = cv2.waitKey(1) & 0xFF


        if tecla == 27:

            break


        if tecla == 32:

            pilha.limpar()

            selo_atual = None
            inicio_selo = None
            ultimo_selo_adicionado = None
            jutsu_atual = None

            print("Pilha limpa.")


        if tecla == 8:

            removido = pilha.desempilhar()

            if removido:

                ultimo_selo_adicionado = None

                print(
                    "Selo removido:",
                    removido
                )


camera.release()

cv2.destroyAllWindows()