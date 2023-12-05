import numpy as np
import matplotlib.pyplot as plt

def binary_to_decimal(binary_str):
    # Convierte una cadena binaria a decimal
    decimal = int(binary_str, 2)
    return decimal

def initialize_population(population_size):
    # Genera una población inicial de individuos
    population = np.random.randint(2, size=(population_size, 9))
    return population

def evaluate_fitness(population):
    # Evalúa la función de aptitud para cada individuo en la población
    fitness_values = []
    for individual in population:
        sign = -1 if individual[0] == 1 else 1
        binary_str = ''.join(map(str, individual[1:]))
        decimal_value = binary_to_decimal(binary_str) / 100.0  # Dividir por 100 para el punto decimal
        x = sign * decimal_value
        fitness = -(x**2 - 1) * (x - 35) * (x - 4)
        fitness_values.append(fitness)
    return np.array(fitness_values)

def select_parents(population, fitness_values):
    # Selecciona a los padres basándose en la ruleta ponderada acumulada
    total_fitness = np.sum(fitness_values)
    
    if total_fitness == 0:
        probabilities = np.ones(len(fitness_values)) / len(fitness_values)
    else:
        probabilities = fitness_values / total_fitness

    # Asegúrate de que las probabilidades sean no negativas
    probabilities = np.maximum(probabilities, 0)

    # Asegúrate de que las probabilidades sumen 1
    probabilities /= np.sum(probabilities)

    # Ruleta ponderada acumulada
    cumulative_probabilities = np.cumsum(probabilities)
    
    selected_indices = []
    for _ in range(len(population) // 2):
        random_number = np.random.rand()
        selected_index = np.searchsorted(cumulative_probabilities, random_number)
        selected_indices.append(selected_index)

    parents = population[selected_indices]
    return parents

def crossover(parents):
    # Cruce aleatorio entre pares específicos y mutación posterior
    children = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i + 1]

        # Cruzar los bits del 2 al 8
        crossover_point = np.random.randint(2, 8)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

        # Muta un bit aleatorio en cada hijo
        mutation_point_child1 = np.random.randint(0, 8)
        mutation_point_child2 = np.random.randint(0, 8)
        child1[mutation_point_child1] ^= 1
        child2[mutation_point_child2] ^= 1

        children.extend([child1, child2])

    return np.array(children)

# Parámetros del algoritmo genético
population_size = 8
num_generations = 50
mutation_rate = 0.01

# Inicializar población
population = initialize_population(population_size)

# Listas para almacenar la mejor aptitud en cada generación
best_fitness_history = []

# Ciclo de evolución
for generation in range(num_generations):
    # Evaluar la aptitud
    fitness_values = evaluate_fitness(population)

    # Seleccionar padres
    parents = select_parents(population, fitness_values)

    # Cruzar padres para generar descendencia
    children = crossover(parents)

    # Reemplazar la población anterior con la nueva generación
    population[:len(children)] = children
    
    # Mostrar el mejor individuo en cada generación
    best_index = np.argmax(fitness_values)
    best_fitness = fitness_values[best_index]
    best_individual = population[best_index]
    print(f"Mejor Aptitud = {best_fitness}, Mejor Individuo = {best_individual}")
    print("\n" + "-"*30 + "\n")

    # Almacenar la mejor aptitud en la historia
    best_fitness_history.append(best_fitness)
    
    # Mostrar el estado de la población y la aptitud en cada generación
    print(f"Generación {generation + 1}:")
    print("Población:")
    print(population)
    print("Función de Aptitud:")
    print(fitness_values)

    
    
# Graficar la evolución de la mejor aptitud
plt.plot(range(1, num_generations + 1), best_fitness_history, marker='o')
plt.xlabel('Generación')
plt.ylabel('Mejor Aptitud')
plt.title('Convergencia del Algoritmo Genético')
plt.grid(True)
plt.show()

# Mostrar el resultado final
final_fitness_values = evaluate_fitness(population)
best_index = np.argmax(final_fitness_values)
best_fitness = final_fitness_values[best_index]
best_individual = population[best_index]
print(f"\nResultado Final: Mejor Aptitud = {best_fitness}, Mejor Individuo = {best_individual}")
