import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Configurações dos parâmetros
M = 50  # Número total de treinadores
b_min = 1e5 # Banda mínima
B = 10e6 # Banda total
d_m = 20

# Simulação de parâmetros adicionais
D_m = np.random.uniform(5e6, 10e6, M)  # Tamanho dos dados locais
f_m = np.random.uniform(1e9, 1.6e9, M)  # Frequência da CPU
c_m = 15  # Constante do modelo de processamento
p_c = 1  # Custo de computação
p_tr = 1  # Custo de transmissão
round_deadline = np.random.uniform(0.001, 1)
#round_deadline = 0.003  # Deadline normalizado
slice_types = ['eMBB', 'uRLLC', 'mMTC']
local_epochs = 10

# Dados simulados: tráfego mensal em fatias
time_series_length = 4320  # 30 dias x 24 horas
traffic_data = {
    slice_type: np.random.uniform(100, 1000, time_series_length) for slice_type in slice_types
}


# Inicialização dos treinadores
near_rt_rics = [{'id': i, 'slice': random.choice(slice_types), 'dataset_size': D_m[i]} for i in range(M)]
print(near_rt_rics)
