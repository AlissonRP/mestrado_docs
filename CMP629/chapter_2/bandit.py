#%%

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


class KArmedBandit:
    """
    Ambiente de K-armed Bandit estacionário.

    Cada ação 'a' gera recompensas segundo uma distribuição normal:
        R(a) ~ N(mu[a], sigma^2)

    Attributes
    ----------
    k : int
        Número de ações (ou braços) disponíveis.
    means : np.ndarray
        Médias verdadeiras de cada ação (se não fornecido, são sorteadas aleatoriamente).
    sigma : float
        Desvio padrão das distribuições de recompensa.
    rng : np.random.Generator
        Gerador de números aleatórios para reprodutibilidade.
    optimal_action : int
        Índice da ação com maior valor esperado (melhor ação).
    """

    def __init__(self, k: int = 10, means: np.ndarray = None, sigma: float = 1.0, seed: int = None):
        """
        Parameters
        ----------
        k : int, default=10
            Número de ações do bandit.
        means : np.ndarray, optional
            Médias verdadeiras das recompensas de cada ação. Se None, são sorteadas.
        sigma : float, default=1.0
            Desvio padrão das distribuições normais de recompensa.
        seed : int, optional
            Semente para reprodutibilidade.
        """
        self.k = k
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

        if means is None:
            self.means = self.rng.normal(0, 1, size=k)
        else:
            self.means = np.array(means)

        self.optimal_action = int(np.argmax(self.means))

    def pull(self, action: int) -> float:
        """
        Executa uma ação e retorna uma recompensa estocástica.

        Parameters
        ----------
        action : int
            Índice da ação escolhida.

        Returns
        -------
        float
            Recompensa obtida ao executar a ação.
        """
        return self.rng.normal(self.means[action], self.sigma)


class EpsilonGreedyAgent:
    """
    Agente que usa política ε-greedy para seleção de ações.

    A cada passo:
        - Com probabilidade (1 - ε), escolhe a ação com maior valor estimado Q(a).
        - Com probabilidade ε, escolhe uma ação aleatória (exploração).

    Attributes
    ----------
    k : int
        Número de ações possíveis.
    Q : np.ndarray
        Estimativas de valor de cada ação.
    N : np.ndarray
        Contador de quantas vezes cada ação foi escolhida.
    epsilon0 : float
        Valor inicial de ε.
    epsilon_min : float
        Valor mínimo que ε pode assumir (em caso de decaimento).
    decay : float
        Taxa de decaimento de ε ao longo do tempo.
    t : int
        Passo de tempo atual.
    rng : np.random.Generator
        Gerador de números aleatórios.
    """

    def __init__(
        self,
        k: int,
        epsilon: float = 0.1,
        epsilon_min: float = None,
        decay: float = 0.0,
        init_value: float = 0.0,
    ):
        """
        Parameters
        ----------
        k : int
            Número de ações possíveis.
        epsilon : float, default=0.1
            Probabilidade inicial de explorar uma ação aleatória.
        epsilon_min : float, optional
            Valor mínimo para ε em caso de decaimento. Se None, assume o mesmo valor de epsilon.
        decay : float, default=0.0
            Taxa de decaimento de ε. Se 0, ε é fixo.
        init_value : float, default=0.0
            Valor inicial otimista/pessimista para as estimativas Q(a).
        """
        self.k = k
        self.Q = np.full(k, init_value, dtype=float)
        self.N = np.zeros(k, dtype=int)
        self.epsilon0 = epsilon
        self.epsilon_min = epsilon if epsilon_min is None else epsilon_min
        self.decay = decay
        self.t = 0
        self.rng = np.random.default_rng()

    def epsilon(self) -> float:
        """
        Calcula o valor atual de ε (fixo ou decaindo).

        Returns
        -------
        float
            Valor de ε no tempo atual.
        """
        eps = self.epsilon0 / (1.0 + self.t * self.decay)
        return max(self.epsilon_min, eps)

    def select_action(self) -> int:
        """
        Seleciona uma ação segundo a política ε-greedy.

        Returns
        -------
        int
            Índice da ação escolhida.
        """
        self.t += 1
        if self.rng.random() < self.epsilon():  # exploração
            return self.rng.integers(self.k)

        max_q = np.max(self.Q)
        candidates = np.flatnonzero(self.Q == max_q)
        return self.rng.choice(candidates)

    def update(self, action: int, reward: float):
        """
        Atualiza a estimativa Q(a) da ação escolhida com média incremental.

        Parameters
        ----------
        action : int
            Índice da ação executada.
        reward : float
            Recompensa observada.
        """
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


def run_bandit(env: KArmedBandit, agent: EpsilonGreedyAgent, steps: int = 1000) -> dict:
    """
    Executa simulação de um agente em um ambiente K-armed bandit.

    Parameters
    ----------
    env : KArmedBandit
        Ambiente de bandit.
    agent : EpsilonGreedyAgent
        Agente ε-greedy.
    steps : int, default=1000
        Número de passos da simulação.

    Returns
    -------
    dict
        Resultados da simulação com as seguintes chaves:
        - "rewards" : np.ndarray
            Recompensas obtidas em cada passo.
        - "actions" : np.ndarray
            Ações escolhidas em cada passo.
        - "optimal" : np.ndarray
            Booleanos indicando se a ação ótima foi escolhida em cada passo.
        - "Q" : np.ndarray
            Estimativas finais Q(a).
        - "N" : np.ndarray
            Contadores de quantas vezes cada ação foi escolhida.
    """
    rewards = np.zeros(steps)
    optimal = np.zeros(steps, dtype=bool)
    actions = np.zeros(steps, dtype=int)

    for t in range(steps):
        action = agent.select_action()
        reward = env.pull(action)
        agent.update(action, reward)

        actions[t] = action
        rewards[t] = reward
        optimal[t] = (action == env.optimal_action)

    return {
        "rewards": rewards,
        "actions": actions,
        "optimal": optimal,
        "Q": agent.Q,
        "N": agent.N,
    }


#%%
# Criar ambiente e agente
env = KArmedBandit(k=6, seed=42)
agent = EpsilonGreedyAgent(k=env.k, epsilon=0.1, epsilon_min=0.01, decay=1e-3)

# Rodar experimento
result = run_bandit(env, agent, steps=5000)

print("Médias verdadeiras:", np.round(env.means, 2))
print("Ação ótima:", env.optimal_action + 1)
print("Recompensa média obtida:", np.mean(result["rewards"]))
print("Taxa de ação ótima:", np.mean(result["optimal"]) * 100, "%")

# Plotar resultados
plt.figure(figsize=(12, 5))

# Recompensa acumulada
plt.plot(np.cumsum(result["rewards"]) / (np.arange(len(result["rewards"])) + 1))
plt.xlabel("Passos")
plt.ylabel("Recompensa média acumulada")
plt.title("Desempenho do agente ε-greedy")
plt.grid(True)
plt.show()
