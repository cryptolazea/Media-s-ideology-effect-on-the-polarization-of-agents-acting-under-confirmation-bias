import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

class SocialNetwork:
    def __init__(self, medias : int, medias_beliefs : np.ndarray , initial_agents : int = 40, total_agents : int =1000, confirmation_bias : float= 0.1):
        self.initial_agents = initial_agents
        self.total_agents = total_agents
        self.medias = medias
        self.confirmation_bias = confirmation_bias
        self.adj_matrix = np.zeros((self.total_agents+self.medias, self.total_agents+self.medias))
        self.communication_matrix = np.zeros((self.total_agents+self.medias, self.total_agents+self.medias))
        self.beliefs = np.random.uniform(0, 1, self.total_agents + self.medias) # Initial beliefs
        self.beliefs[total_agents:]=medias_beliefs
        self.polarization_history=[]
    def initialize_network(self):
        # Set up initial fully connected network with random direction
        for i in range(self.initial_agents):
            for j in range(i):
                if np.random.rand() < 0.5:
                    self.adj_matrix[i][j] = 1  # i -> j
                else:
                    self.adj_matrix[j][i] = 1  # j -> i

    def expand_network(self):
        m_r = 20
        p_r = 0.8
        p_n = 0.8

        for current_agent in range(self.initial_agents, self.total_agents):
            # Step 1: Randomly meet m_r existing nodes
            initial_links = np.random.choice(current_agent, m_r, replace=False)
            for node in initial_links:
                if np.random.rand() < p_r:
                    if np.random.rand() < 0.5:
                        self.adj_matrix[current_agent][node] = 1  # New node -> Existing node
                    else:
                        self.adj_matrix[node][current_agent] = 1  # Existing node -> New node

            # Step 2: Meet m_n neighbors of neighbors
            neighbors_of_neighbors = set()
            for node in np.nonzero(self.adj_matrix[current_agent, :current_agent])[0]:
                neighbors_of_neighbors.update(np.nonzero(self.adj_matrix[node, :current_agent])[0])

            neighbors_of_neighbors = list(neighbors_of_neighbors - {current_agent})
            if len(neighbors_of_neighbors) > 0:
                neighbor_links = np.random.choice(neighbors_of_neighbors, min(m_r, len(neighbors_of_neighbors)),
                                                  replace=True)
                for node in neighbor_links:
                    if np.random.rand() < p_n:
                        if np.random.rand() < 0.5:
                            self.adj_matrix[current_agent][node] = 1  # New node -> Existing node
                        else:
                            self.adj_matrix[node][current_agent] = 1  # Existing node -> New node


    def add_medias(self, relative_importance: float = 100.0):
        # Media nodes are added at the very end of the adjacency matrix
        start_index = self.total_agents
        connection_probability = 0.5  # Adjust as needed
        for i in range(start_index, self.total_agents+self.medias):
            for j in range(start_index):
                if np.random.rand() < connection_probability:
                    self.adj_matrix[j][i] = relative_importance  # Existing node -> Media node
    def normalize_matrix(self):
        for i in range(self.total_agents):
            out_links = np.sum(self.adj_matrix[i, :])
            if out_links > 0:
                self_weight = np.random.uniform(0.5, 1)
                self.adj_matrix[i][i] = self_weight
                for j in range(self.total_agents + self.medias):
                    if self.adj_matrix[i][j] != 0 and i != j:

                        self.adj_matrix[i][j] = (1 - self_weight) / (out_links) *self.adj_matrix[i][j]#distribute the rest equally

        for i in range(self.total_agents, self.total_agents + self.medias):
            self.adj_matrix[i][i] = 1  # Ensure media nodes' self-weight is 1
            for j in range(self.total_agents + self.medias):
                if i != j:
                    self.adj_matrix[i][j] = 0  # No influence to other nodes

    def create_communication_matrix(self):
        n=self.total_agents+self.medias
        q=self.confirmation_bias
        self.communication_matrix=self.adj_matrix.copy()
        for i in range(n):
            for j in range(n):
                if abs(self.beliefs[i] - self.beliefs[j]) > (1 - q):
                    self.communication_matrix[j, j] = self.communication_matrix[j, j] + self.communication_matrix[j, i]
                    self.communication_matrix[j, i] = 0
    def update_beliefs(self):
        # Placeholder for belief update logic
        self.beliefs = self.communication_matrix @ self.beliefs
        polarization=self.polarization()
        self.polarization_history.append(polarization)
    def polarization(self):
        return np.var(self.beliefs)

    def run_belief_updates(self, iterations: int = 10):
        for _ in range(iterations):
            self.update_beliefs()

    def plot_degree_distribution(self):
        degrees = np.count_nonzero(self.adj_matrix, axis=1)  # Count non-zero elements in each row
        degree_counts = np.bincount(degrees)  # Count occurrences of each degree
        degree_prob = degree_counts / len(degrees)  # Normalize to get probabilities
        ccdf = 1 - np.cumsum(degree_prob)  # Calculate CCDF

        plt.figure(figsize=(10, 6))
        plt.loglog(range(len(ccdf)), ccdf, marker='o', linestyle='-')
        plt.title('Complementary Cumulative Degree Distribution (CCDF)')
        plt.xlabel('Degree (d)')
        plt.ylabel('1 - F(d)')
        plt.grid(True, which="both", ls="--")
        plt.show()
    def run_simulation_with_belief_plot(self, iterations: int = 10):
        plt.figure(figsize=(10, 6))
        for i in range(0,iterations):
            self.update_beliefs()
            sns.kdeplot(self.beliefs, label=f'Iteration {i + 1}')
        plt.title('Belief Distribution Over Iterations')
        plt.xlabel('Belief Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()
    def get_polarization_history(self):
        return self.polarization_history
    def plot_polarization(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.polarization_history, marker='o', linestyle='-')
        plt.title('Polarization Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Polarization (Variance of Beliefs)')
        plt.grid(True)
        plt.show()

    def average_listening_count(self):
        listening_counts = np.count_nonzero(self.communication_matrix, axis=1) - 1  # Subtract 1 to exclude self-weights
        return np.mean(listening_counts)

    def average_media_in_degree(self):
        media_columns = self.communication_matrix[:, -self.medias:]
        in_degrees = np.count_nonzero(media_columns, axis=0) - 1  # Subtract 1 to exclude self-loops
        return np.mean(in_degrees)

    def percentage_higher_than_threshold(self, threshold):
        return np.sum(self.beliefs > threshold) / len(self.beliefs) * 100

def plot_degree_ccdf(adj_matrix):
    degrees = np.count_nonzero(adj_matrix, axis=1)  # Count non-zero elements in each row
    degree_counts = np.bincount(degrees)  # Count occurrences of each degree
    degree_prob = degree_counts / len(degrees)  # Normalize to get probabilities
    ccdf = 1 - np.cumsum(degree_prob)  # Calculate CCDF

    plt.figure(figsize=(10, 6))
    plt.loglog(range(len(ccdf)), ccdf, marker='o', linestyle='-')
    plt.title('Complementary Cumulative Degree Distribution (CCDF)')
    plt.xlabel('Degree (d)')
    plt.ylabel('1 - F(d)')
    plt.grid(True, which="both", ls="--")
    plt.show()

def run_single_simulation(media_beliefs, confirmation_bias, iterations=10):
    network = SocialNetwork(medias=len(media_beliefs), medias_beliefs=media_beliefs, confirmation_bias=confirmation_bias)
    network.initialize_network()
    network.expand_network()
    network.add_medias()
    network.normalize_matrix()
    network.create_communication_matrix()
    network.run_belief_updates(iterations)
    return network.get_polarization_history()

def run_multiple_simulations(media_beliefs, confirmation_bias, num_simulations=1000, iterations=10):
    all_histories = []
    for _ in range(num_simulations):
        history = run_single_simulation(media_beliefs, confirmation_bias, iterations)
        all_histories.append(history)
    all_histories = np.array(all_histories)
    return np.mean(all_histories, axis=0)


def run_simulation_and_test(media_beliefs, confirmation_bias, threshold, iterations_list):
    

    percentages = []
    for iterations in iterations_list:
        network = SocialNetwork(medias=len(media_beliefs), medias_beliefs=media_beliefs,
                            confirmation_bias=confirmation_bias)
        network.initialize_network()
        network.expand_network()
        network.add_medias()
        network.normalize_matrix()
        network.create_communication_matrix()
        network.run_belief_updates(iterations)
        percentage = network.percentage_higher_than_threshold(threshold)
        percentages.append(percentage)
        #network.plot_polarization()  # Optional: to visualize polarization at each step

    return percentages


    
def run_simulation_and_test_fixed_network(media_beliefs, confirmation_bias, threshold, iterations_list, network):
    network = network
    

    percentages = []
    for iterations in iterations_list:
        network_copy=network
        network_copy.run_belief_updates(iterations)
        percentage = network_copy.percentage_higher_than_threshold(threshold)
        percentages.append(percentage)
        #network.plot_polarization()  # Optional: to visualize polarization at each step

    return percentages

def best_response_function_miu_b(confirmation_bias, miu_a):
    confirmation_bias = confirmation_bias
    miu_a=miu_a
    threshold = 0.5
    iterations_list = [5]
    steps = 0.05
    num_runs = 50  # Number of runs to average over
    max_average_percentage = -1
    best_miu_b = None
    
    for miu_b in np.arange(0, 1 + steps, steps):
        media_beliefs = np.array([miu_a, miu_b])
        all_percentages = []

        for _ in range(num_runs):
            percentages = run_simulation_and_test(media_beliefs, confirmation_bias, threshold, iterations_list)
            all_percentages.append(percentages)

        array = np.array(all_percentages)
        average_percentages = np.mean(array, axis=0)
        average_of_all_iterations = np.mean(average_percentages)

        if average_of_all_iterations > max_average_percentage:
            max_average_percentage = average_of_all_iterations
            best_miu_b = miu_b


    print(
        f"For miu_a = {miu_a:.2f}, the best miu_b is {best_miu_b:.2f} with an average percentage of {max_average_percentage:.2f}%")
    return best_miu_b
def best_response_function_miu_b_fixed_network(confirmation_bias, miu_a):
    confirmation_bias = confirmation_bias
    miu_a=miu_a
    threshold = 0.5
    iterations_list = [5]
    steps = 0.05
    num_runs = 50  # Number of runs to average over
    max_average_percentage = -1
    best_miu_b = None
    

    for miu_b in np.arange(0, 1 + steps, steps):
        media_beliefs = np.array([miu_a, miu_b])
        all_percentages = []
        network = SocialNetwork(medias=len(media_beliefs), medias_beliefs=media_beliefs,
                            confirmation_bias=confirmation_bias)
        network.initialize_network()
        network.expand_network()
        network.add_medias()
        network.normalize_matrix()
        network.create_communication_matrix()
        

        for _ in range(num_runs):
            percentages = run_simulation_and_test_fixed_network(media_beliefs, confirmation_bias, threshold, iterations_list, network)
            all_percentages.append(percentages)

        array = np.array(all_percentages)
        average_percentages = np.mean(array, axis=0)
        average_of_all_iterations = np.mean(average_percentages)

        if average_of_all_iterations > max_average_percentage:
            max_average_percentage = average_of_all_iterations
            best_miu_b = miu_b


    print(
        f"For miu_a = {miu_a:.2f}, the best miu_b is {best_miu_b:.2f} with an average percentage of {max_average_percentage:.2f}%, under a fixed network")
    return best_miu_b

    
def best_response_function_plot(confirmation_bias) :
    confirmation_bias = confirmation_bias
    threshold = 0.5
    iterations_list = [5]
    steps = 0.05
    num_runs = 1  # Number of runs to average over
    

    miu_a_values = np.arange(0, 0.5 + steps, steps)
    best_miu_b_for_miu_a = {}

    for miu_a in miu_a_values:
        max_average_percentage = -1
        best_miu_b = None
        
        for miu_b in np.arange(0.5, 1 + steps, steps):
            media_beliefs = np.array([miu_a, miu_b])
            all_percentages = []

            for _ in range(num_runs):
                percentages = run_simulation_and_test(media_beliefs, confirmation_bias, threshold, iterations_list)
                all_percentages.append(percentages)
                

            array = np.array(all_percentages)
            average_percentages = np.mean(array, axis=0)
            average_of_all_iterations = np.mean(average_percentages)
            

            if average_of_all_iterations > max_average_percentage:
                max_average_percentage = average_of_all_iterations
                best_miu_b = miu_b

        best_miu_b_for_miu_a[miu_a] = best_miu_b
        print(
            f"For miu_a = {miu_a:.2f}, the best miu_b is {best_miu_b:.2f} with an average percentage of {max_average_percentage:.2f}%")
    miu_a_values = list(best_miu_b_for_miu_a.keys())
    best_miu_b_values = list(best_miu_b_for_miu_a.values())

    plt.figure(figsize=(10, 6))
    plt.plot(miu_a_values, best_miu_b_values, marker='o', linestyle='-')
    plt.title('Best miu_b for each miu_a')
    plt.xlabel('miu_a')
    plt.ylabel('Best miu_b')
    plt.grid(True)
    plt.show()
def best_response_function_plot_fixed_network(confirmation_bias) :
    confirmation_bias = confirmation_bias
    threshold = 0.5
    iterations_list = [5]
    steps = 0.05
    num_runs = 1  # Number of runs to average over


    miu_a_values = np.arange(0, 0.5 + steps, steps)
    best_miu_b_for_miu_a = {}
    
    for miu_a in miu_a_values:
        max_average_percentage = -1
        best_miu_b = None

        for miu_b in np.arange(0.5, 1 + steps, steps):
            media_beliefs = np.array([miu_a, miu_b])
            all_percentages = []
            network = SocialNetwork(medias=len(media_beliefs), medias_beliefs=media_beliefs,
                            confirmation_bias=confirmation_bias)
            network.initialize_network()
            network.expand_network()
            network.add_medias()
            network.normalize_matrix()
            network.create_communication_matrix()

            for _ in range(num_runs):
                percentages = run_simulation_and_test_fixed_network(media_beliefs, confirmation_bias, threshold, iterations_list, network)
                all_percentages.append(percentages)

            array = np.array(all_percentages)
            average_percentages = np.mean(array, axis=0)
            average_of_all_iterations = np.mean(average_percentages)

            if average_of_all_iterations > max_average_percentage:
                max_average_percentage = average_of_all_iterations
                best_miu_b = miu_b

        best_miu_b_for_miu_a[miu_a] = best_miu_b
        print(
            f"For miu_a = {miu_a:.2f}, the best miu_b is {best_miu_b:.2f} with an average percentage of {max_average_percentage:.2f}%, under a fixed network")
    miu_a_values = list(best_miu_b_for_miu_a.keys())
    best_miu_b_values = list(best_miu_b_for_miu_a.values())

    plt.figure(figsize=(10, 6))
    plt.plot(miu_a_values, best_miu_b_values, marker='o', linestyle='-')
    plt.title('Best miu_b for each miu_a, fixed network')
    plt.xlabel('miu_a')
    plt.ylabel('Best miu_b')
    plt.grid(True)
    plt.show()


def run_simulation_and_collect_polarization(media_beliefs, confirmation_bias, add_medias_func, iterations, num_runs):
    all_polarization_histories = []
    for _ in range(num_runs):
        network = SocialNetwork(medias=len(media_beliefs), medias_beliefs=media_beliefs,
                                confirmation_bias=confirmation_bias)
        network.initialize_network()
        network.expand_network()
        add_medias_func(network)
        network.normalize_matrix()
        network.create_communication_matrix()
        network.run_belief_updates(iterations)
        all_polarization_histories.append(np.array(network.get_polarization_history()))
    all_polarization_histories = np.array(all_polarization_histories)
    mean_polarization_history = np.mean(all_polarization_histories, axis=0)
    return mean_polarization_history


def run_simulations_and_plot_average_beliefs(media_beliefs, confirmation_bias, add_medias_func, iterations=10,
                                             num_runs=50):
    """
    Runs multiple simulations, averages the belief distributions over the num_runs simulations,
    and plots the averaged distribution.

    Parameters:
        media_beliefs (np.ndarray): Array of media beliefs.
        confirmation_bias (float): The confirmation bias parameter.
        add_medias_func (function): The function used to add media connections in the network (either add_medias or add_medias_under_Hotelling).
        iterations (int): Number of iterations to run in each simulation.
        num_runs (int): Number of simulations to average over.
    """
    all_beliefs = []  # List to store beliefs from all runs for each iteration

    for _ in range(num_runs):
        network = SocialNetwork(medias=len(media_beliefs), medias_beliefs=media_beliefs,
                                confirmation_bias=confirmation_bias)
        network.initialize_network()
        network.expand_network()
        add_medias_func(network)  # Dynamic dispatch of media adding function
        network.normalize_matrix()
        network.create_communication_matrix()
        for i in range(iterations):
            network.update_beliefs()
            if i == iterations - 1:  # Only collect the final beliefs for each run
                all_beliefs.append(network.beliefs)

    # Convert the list of arrays to a 2D numpy array for easier processing
    all_beliefs = np.array(all_beliefs)

    # Average beliefs across all runs
    average_beliefs = np.mean(all_beliefs, axis=0)

    # Plotting the averaged belief distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(average_beliefs, label=f'Average Distribution after {iterations} Iterations')
    plt.title('Averaged Belief Distribution Over Multiple Simulations')
    plt.xlabel('Belief Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
def main():
    media_beliefs_set1 = np.array([0.7, 0.8, 0.9])
    media_beliefs_set2 = np.array([0.5, 0.5, 0.5])
    confirmation_bias = 0.75
    num_simulations = 10
    iterations = 10
    
    avg_polarization_history_set1 = run_multiple_simulations(media_beliefs_set1, confirmation_bias, num_simulations,
                                                             iterations)
    avg_polarization_history_set2 = run_multiple_simulations(media_beliefs_set2, confirmation_bias, num_simulations,
                                                             iterations)
    
    plt.figure(figsize=(12, 6))
    plt.plot(avg_polarization_history_set1, marker='o', linestyle='-', label='Media Beliefs Set 1')
    plt.plot(avg_polarization_history_set2, marker='o', linestyle='-', label='Media Beliefs Set 2')
    plt.title('Average Polarization Over Iterations (1000 Simulations)')
    plt.xlabel('Iteration')
    plt.ylabel('Average Polarization (Variance of Beliefs)')
    plt.legend()
    plt.grid(True)
    plt.show()


    media_beliefs = np.array([0.4, 0.7])

    network = SocialNetwork(medias=2, medias_beliefs=media_beliefs, confirmation_bias=0.4)

    network.initialize_network()
    network.expand_network()
    network.add_medias()
    network.normalize_matrix()
    network.create_communication_matrix()
    plot_degree_ccdf(network.adj_matrix)
    network.run_simulation_with_belief_plot(10)
    network.run_belief_updates(100)
    network.plot_polarization()


    
    confirmation_bias = 0.3
    #best_response_function_plot(confirmation_bias)
    #best_response_function_plot_fixed_network(confirmation_bias)
    iterations = 50
    miu_a_values = [0.1, 0.25, 0.4]
    num_runs = 50
    for miu_a in miu_a_values:
        best_response_function_miu_b_fixed_network(confirmation_bias, miu_a)
    plt.figure(figsize=(12, 8))

    for miu_a in miu_a_values:
        miu_b = best_response_function_miu_b_fixed_network(confirmation_bias=confirmation_bias, miu_a=miu_a)
        media_beliefs = np.array([miu_a, miu_b])

        # For add_medias method
        polarization_history_add_medias = run_simulation_and_collect_polarization(
            media_beliefs, confirmation_bias, lambda network: network.add_medias(), iterations, num_runs)

        # Plot results
        plt.plot(polarization_history_add_medias, label=f'miu_a={miu_a}, add_medias')

   
    plt.title('Polarization Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Polarization (Variance of Beliefs)')
    plt.legend()
    plt.grid(True)
    plt.savefig('.pdf')
    plt.show()
    print("final")
if __name__ == "__main__":
    main()