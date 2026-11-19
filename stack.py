class PilhaSelos:

    def __init__(self):
        self.selos = []

    def empilhar(self, selo):
        self.selos.append(selo)

    def desempilhar(self):
        if self.esta_vazia():
            return None

        return self.selos.pop()

    def topo(self):
        if self.esta_vazia():
            return None

        return self.selos[-1]

    def esta_vazia(self):
        return len(self.selos) == 0

    def tamanho(self):
        return len(self.selos)

    def limpar(self):
        self.selos.clear()

    def mostrar(self):
        return self.selos


pilha = PilhaSelos()

pilha.empilhar("cobra")
pilha.empilhar("carneiro")
pilha.empilhar("macaco")

print(pilha.mostrar())

print("Topo:", pilha.topo())

print("Saiu:", pilha.desempilhar())

print(pilha.mostrar())