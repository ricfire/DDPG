import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Tuple
from parametros import *
from collections import Counter
import time


# Função para calcular o Índice de Justiça de Jain
def jain_fairness_index(allocations):
    """Calcula o fairness index para uma lista de alocações.
    :arg
        RICs alocados
    :returns
        Indíce de justiça entre 0  e 1
    """

    if len(allocations) == 0:
        return 0.0
    sum_x = np.sum(allocations)
    sum_x_squared = np.sum(np.square(allocations))
    return (sum_x ** 2) / (len(allocations) * sum_x_squared)


class ResourceAllocationEnv(gym.Env):


    def __init__(self):
        """Inicializa o ambiente com a ação e o espaço de observação"""
        super(ResourceAllocationEnv, self).__init__()
        self.num_slices = 3
        self.slice_counts_over_time = []
        self.discrete_action = False

        # Definir espaço de ação: os primeiros valores M são seleções binárias (0 ou 1),
        # os próximos M valores são alocações de largura de banda (entre b_min e B)
        self.action_space = spaces.Box(
            low=np.concatenate([np.zeros(M), np.full(M, b_min)]),
            high=np.concatenate([np.ones(M), np.full(M, B)]),
            dtype=np.float32
        )

        # Define o espaço de observação
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(5 * M + 2,),
            dtype=np.float32
        )

        self.state = None
        self.reset()

        # Métricas para análise
        self.compute_costs = []
        self.transmission_costs = []
        self.latencies = []
        self.selected_counts = []
        self.fairness_indices = []
        self.execution_times_ms = []  # Novo: Armazena tempos em milissegundos

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        slice_encoded = np.zeros((M, 3))
        for i, s in enumerate(slice_types):
            if s == "uRLLC":
                slice_encoded[i, 0] = 1
            elif s == "eMBB":
                slice_encoded[i, 1] = 1
            else:
                slice_encoded[i, 2] = 1

        self.state = np.concatenate([
            (f_m - 1.0e9) / 0.6e9,
            (D_m - 5e6) / 5e6,
            slice_encoded.flatten(),
            [0.0],
            [1.0]
        ]).astype(np.float32)

        return self.state, {}

    def step(self, action):
        """Executa uma etapa de tempo no ambiente.

            :arg
                action: ação a ser tomada(seleção e alocação de largura de banda)

            :returns
                tuple: (next_state, reward, done, truncated, info)
            """

        start_time = time.perf_counter()  # Medição mais precisa do tempo

        # Manipular a incompatibilidade do tamanho da ação
        if len(action) != 2 * M:
            corrected_action = np.zeros(2 * M, dtype=np.float32)
            size = min(len(action), 2 * M)
            corrected_action[:size] = action[:size]
            action = corrected_action

        # Converte os primeiros valores de ação M em seleções binárias
        a_m = (action[:M] > 0.5).astype(int)
        n_selected = np.sum(a_m)

        # Garantir de que pelo menos um RIC seja selecionado
        if n_selected == 0:
            a_m[np.random.randint(M)] = 1
            n_selected = 1

        # Alocação da largura de banda do processo (últimos M valores de ação)
        raw_b_m = action[M:]
        if np.sum(raw_b_m) > 0:
            b_m = raw_b_m / np.sum(raw_b_m) * B
        else:
            b_m = np.full(M, b_min)

        # Aplicar máscara de seleção e restrição de largura de banda mínima
        b_m = np.where(a_m == 1, np.maximum(b_m, b_min), 0.0)

        if np.sum(b_m) > B:
            b_m = b_m * (B / np.sum(b_m))

        b_m_safe = np.where(b_m > 0, b_m, b_min)

        # === Custos brutos ===
        compute_cost = np.sum((D_m * c_m / f_m) * p_c * a_m)
        transmission_cost = np.sum(b_m * p_tr * a_m)
        latency = np.max((D_m * c_m / f_m) + (d_m / b_m_safe))

        # === Normalizações ===
        MAX_COMP_COST = 10.0
        MAX_COMM_COST = 1e7
        MAX_LATENCY = round_deadline

        compute_cost_norm = compute_cost / MAX_COMP_COST
        transmission_cost_norm = transmission_cost / MAX_COMM_COST
        latency_norm = latency / MAX_LATENCY

        # === Recompensa ===
        slice_priority = np.array([3.0 if s == "uRLLC" else 1.5 if s == "eMBB" else 0.5 for s in slice_types])

        reward = - ((1 - rho) * (compute_cost_norm + transmission_cost_norm) + rho * latency_norm)
        reward += 0.5 * np.sum(a_m * slice_priority)

        # Penalização por excesso de banda
        if np.sum(b_m) > B:
            reward -= 5.0 * ((np.sum(b_m) - B) / B)

        # Penalidade por "RIC crítico"
        latency_per_ric = (D_m * c_m / f_m) + (d_m / np.where(b_m > 0, b_m, b_min))
        max_latency_penalty = np.max(np.maximum(0, latency_per_ric - round_deadline)) ** 2
        reward -= 2000 * max_latency_penalty

        # Bônus explícito por alocações acima do mínimo
        bonus = np.sum(np.where(b_m > b_min * 1.2, 0.1 * (b_m - b_min), 0))
        reward += bonus

        # === Atualização do estado ===
        self.state[-2] = np.sum(b_m) / B
        self.state[-1] = max(0, 1 - latency_norm)

        # === Métricas para plotagem ===
        self.compute_costs.append(compute_cost)
        self.transmission_costs.append(transmission_cost)
        self.latencies.append(latency)
        self.selected_counts.append(n_selected)

        selected_bandwidths = b_m[np.where(a_m == 1)]
        fairness_index = jain_fairness_index(selected_bandwidths)
        self.fairness_indices.append(fairness_index)

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000  # Convertendo para milissegundos
        self.execution_times_ms.append(elapsed_ms)

        done = latency > round_deadline
        return self.state.copy(), reward, done, False, {}

    def plot_metrics(self):
        plt.figure(figsize=(15, 18))

        # Custo de computação
        plt.subplot(3, 2, 1)
        plt.plot(self.compute_costs, color='blue')
        plt.title("Custo de Computação por Episódio")
        plt.xlabel("Episódios")
        plt.ylabel("Custo")
        plt.grid(True)

        # Custo de transmissão
        plt.subplot(3, 2, 2)
        plt.plot(self.transmission_costs, color='orange')
        plt.title("Custo de Transmissão por Episódio")
        plt.xlabel("Episódios")
        plt.ylabel("Custo")
        plt.axhline(y=B, color='red', linestyle='--', label='Banda Total Máxima')
        plt.legend()
        plt.grid(True)

        # Latência
        plt.subplot(3, 2, 3)
        plt.plot(self.latencies, color='green')
        plt.axhline(round_deadline, color='red', linestyle='--', label='Deadline')
        plt.title("Latência Máxima por Episódio")
        plt.xlabel("Episódios")
        plt.ylabel("Latência (s)")
        plt.legend()
        plt.grid(True)

        # Número de RICs selecionados
        plt.subplot(3, 2, 4)
        plt.plot(self.selected_counts, color='purple')
        plt.title("Quantidade de RICs Selecionados")
        plt.xlabel("Episódios")
        plt.ylabel("Número de RICs")
        plt.axhline(y=M, color='gray', linestyle='--', label='Total de RICs')
        plt.grid(True)
        plt.legend()

        # Fairness Index
        plt.subplot(3, 2, 5)
        plt.plot(self.fairness_indices, color='red')
        plt.title("Índice de Justiça (Jain) por Episódio")
        plt.xlabel("Episódios")
        plt.ylabel("Fairness Index")
        plt.axhline(y=1.0, color='black', linestyle='--', label='Máxima Justiça')
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.legend()

        # Tempo de Execução (em ms)
        plt.subplot(3, 2, 6)
        plt.plot(self.execution_times_ms, color='brown')
        plt.title("Tempo de Execução por Episódio (ms)")
        plt.xlabel("Episódios")
        plt.ylabel("Tempo (ms)")
        plt.grid(True)

        avg_time_ms = np.mean(self.execution_times_ms)
        plt.axhline(y=avg_time_ms, color='red', linestyle='--',
                    label=f'Média: {avg_time_ms:.2f}ms')

        # Configurações para melhor visualização em ms
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        max_time = np.max(self.execution_times_ms)
        plt.ylim(0, max_time * 1.2)
        plt.legend()

        plt.tight_layout()
        plt.show()


class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if len(self.locals['dones']) > 0 and self.locals['dones'][0]:
            self.episode_rewards.append(self.locals['rewards'][0])
            if len(self.episode_rewards) % 10 == 0:
                print(
                    f"Episódio {len(self.episode_rewards)} - Recompensa média: {np.mean(self.episode_rewards[-10:]):.2f}")
        return True


# Ambiente
env = ResourceAllocationEnv()

# Ruído
action_noise = NormalActionNoise(
    mean=np.zeros(2 * M),
    sigma=0.2 * np.ones(2 * M)
)

# Modelo DDPG
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=100000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=(1, "episode")
)

# Treinamento
print("Iniciando treinamento...")
model.learn(total_timesteps=2000, callback=MetricsCallback())

# Avaliação final
print("\nAvaliação final...")
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

# Plotagem de métricas
env.plot_metrics()

# Salvar o modelo treinado
model.save("ddpg_ran_model")

# Resultados finais
a_m = (action[:M] > 0.5).astype(int)
b_m = action[M:] * B / np.sum(action[M:])
b_m = np.where(a_m == 1, np.maximum(b_m, b_min), 0.0)

selected_indices = np.where(a_m == 1)[0]
selected_slices = slice_types[selected_indices]
slice_counts = Counter(selected_slices)
labels = ["uRLLC", "eMBB", "mMTC"]
values = [slice_counts.get(s, 0) for s in labels]

# Gráfico de RICs por slice
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color=["blue", "green", "orange"], width=0.5)
plt.title("Distribuição de RICs Selecionados por Tipo de Slice", fontsize=12)
plt.ylabel("Número de RICs", fontsize=10)
plt.xlabel("Tipo de Slice", fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Fairness index final
selected_bandwidths = b_m[np.where(a_m == 1)]
fairness_index = jain_fairness_index(selected_bandwidths)

print("\n=== RESULTADOS FINAIS ===")
print(f"► RICs Selecionados: {np.where(a_m == 1)[0]}")
print(f"► Banda Alocada: {b_m[np.where(a_m == 1)]}")
print(f"► Total de Banda Utilizada: {np.sum(b_m):.2f}/{B} Mbps ({np.sum(b_m) / B * 100:.1f}%)")
print(f"► Custo Computacional Total: {env.compute_costs[-1]:.2f}")
print(f"► Custo de Transmissão Total: {env.transmission_costs[-1]:.2f}")
print(f"► Latência Máxima Observada: {env.latencies[-1]:.4f}s (Deadline: {round_deadline}s)")
print(f"► Índice de Justiça (Jain): {fairness_index:.4f}")
print(f"► Tempo Médio de Execução por Episódio: {np.mean(env.execution_times_ms):.2f}ms")
print(f"► Índice de Justiça (Jain): {env.fairness_indices:}")
print(env.execution_times_ms)
print("=" * 30)
