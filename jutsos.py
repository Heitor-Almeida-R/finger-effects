JUTSUS = {
    "Bola de Fogo": [
        "cobra",
        "carneiro",
        "macaco",
        "javali",
        "cavalo",
        "tigre"
    ],

    "Invocação": [
        "javali",
        "cao",
        "passaro",
        "macaco"
    ],

    "Clone das Sombras": [
        "carneiro",
        "cobra",
        "tigre"
    ],

    "Grande Bola de Fogo": [
        "cobra",
        "carneiro",
        "tigre"
    ],

    "Raiton": [
        "cobra",
        "coelho",
        "macaco",
        "tigre"
    ],

    "Suiton": [
        "tigre",
        "boi",
        "coelho",
        "rato"
    ],

    "Fuuton": [
        "passaro",
        "tigre",
        "cao",
        "cobra"
    ],

    "Doton": [
        "tigre",
        "javali",
        "cao",
        "boi"
    ],

    "Dragão de Fogo": [
        "cobra",
        "cavalo",
        "tigre",
        "dragao"
    ],

    "Jutsu Supremo": [
        "rato",
        "dragao",
        "tigre",
        "cobra",
        "carneiro",
        "macaco"
    ]
}


def identificar_jutsu(sequencia):

    for nome, selos in JUTSUS.items():

        if sequencia == selos:
            return nome

    return None