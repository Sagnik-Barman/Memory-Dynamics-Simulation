import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from abc import ABC, abstractmethod
from itertools import combinations

class AbstractMemorySystem(ABC):
    @abstractmethod
    def __init__(self, alpha, beta, **kwargs):
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)
        self.n = len(alpha)
        self.bias = np.array(kwargs.get('bias', np.zeros(self.n)))
        self.threshold = kwargs.get('threshold', 0.05)

    @abstractmethod
    def derivatives(self, M, t):
        pass

    def simulate(self, M0, t):
        sol = odeint(self.derivatives, M0, t)
        return sol

    def predict_behavior(self, final_M):
        final_M = np.array(final_M)
        dominant = np.argmax(final_M)
        strengths = ', '.join([f'M{i+1}: {strength:.2f}' for i, strength in enumerate(final_M)])
        return f"Dominant memory is Memory {dominant+1}. (Strengths: {strengths})"

    def plot_all_phase_portraits(self, M0):
        """
        Draws a grid of 2D phase portraits for all unique pairs of memories.
        """
        pairs = list(combinations(range(self.n), 2))
        nplots = len(pairs)
        if nplots == 0:
            print("Not enough memories for pairwise phase portraits.")
            return
        
        cols = min(nplots, 3)
        rows = int(np.ceil(nplots / cols))
        plt.figure(figsize=(5 * cols, 5 * rows))
        for idx, (i, j) in enumerate(pairs):
            plt.subplot(rows, cols, idx+1)

            M1_vals = np.linspace(-1.5, 1.5, 20)
            M2_vals = np.linspace(-1.5, 1.5, 20)
            M1, M2 = np.meshgrid(M1_vals, M2_vals)
            # Slices for i, j
            alpha = self.alpha[[i, j]]
            beta = self.beta[np.ix_([i, j], [i, j])]
            bias = self.bias[[i, j]]

            dM1dt = alpha[0]*M1 + beta[0][0]*M1 + beta[0][1]*M2 + bias[0]
            dM2dt = alpha[1]*M2 + beta[1][0]*M1 + beta[1][1]*M2 + bias[1]
            N = np.sqrt(dM1dt**2 + dM2dt**2)
            N[N == 0] = 1
            plt.quiver(M1, M2, dM1dt/N, dM2dt/N, color='r')
            plt.xlabel(f'Memory {i+1} Strength')
            plt.ylabel(f'Memory {j+1} Strength')
            plt.title(f'Phase Portrait: M{i+1} vs M{j+1}')
            plt.plot(M0[i], M0[j], 'bo', label='Initial Condition')
            plt.legend()
            plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_results(self, t, sol, M0):
        """
        Plot time evolution for all memories and mark the initial condition on individual plots.
        - Left subplot: all memory strengths vs time
        - Right area: (optional) will be used by plot_all_phase_portraits for pairwise portraits
        """
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        for i in range(self.n):
            plt.plot(t, sol[:, i], label=f'Memory {i+1}')
          
            plt.plot(t[0], sol[0, i], 'o')
        plt.xlabel('Time')
        plt.ylabel('Memory Strength')
        plt.title('Memory Strengths over Time')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        final_vals = sol[-1]
        txt = "Final Memory Strengths:\n\n"
        for i, v in enumerate(final_vals):
            txt += f"Memory {i+1}: {v:.3f}\n"
        txt += "\n" + self.predict_behavior(final_vals)
        plt.axis('off')
        plt.text(0.01, 0.99, txt, va='top', ha='left', fontsize=12, family='monospace')
        plt.title('Summary')
        plt.tight_layout()
        plt.show()

class LinearMemorySystem(AbstractMemorySystem):
    def __init__(self, alpha, beta, **kwargs):
        super().__init__(alpha, beta, **kwargs)

    def derivatives(self, M, t):
        M = np.array(M)
        dMdt = self.alpha * M + self.beta @ M + self.bias
        return dMdt.tolist()

def get_float_list_input(prompt, size):
    floats = []
    while len(floats) < size:
        try:
            inputs = input(f"{prompt} (enter {size} numbers separated by spaces): ").strip().split()
            floats = [float(x) for x in inputs]
            if len(floats) != size:
                print(f"Please enter exactly {size} numbers.")
                floats = []
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")
    return floats

def get_full_matrix_input(prompt, size):
    mat = []
    print(f"{prompt}\nPlease enter each row as {size} numbers separated by spaces:")
    for i in range(size):
        row = get_float_list_input(f"Row {i+1}", size)
        mat.append(row)
    return mat

def main():
    print("USER INPUT FOR MEMORY SYSTEM PARAMETERS\n")
    while True:
        try:
            n = int(input("Enter number of memories (2 or more): "))
            if n < 2:
                print("Please enter at least 2.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    alpha = get_float_list_input("Enter self-dynamics for each memory (alpha)", n)
    beta = get_full_matrix_input("Enter the full beta interaction matrix (beta, rows)", n)
    while True:
        bias_inp = input(f"Enter bias for each memory (default all zeros, {n} numbers, optional, press Enter to skip): ").strip()
        if not bias_inp:
            bias = np.zeros(n)
            break
        try:
            bias = np.array([float(x) for x in bias_inp.split()])
            if len(bias) != n:
                print(f"Please enter exactly {n} numbers.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter numbers.")
    M0 = get_float_list_input("Enter initial value for each memory", n)
    while True:
        try:
            sim_time = float(input("Simulation end time (default 100): ") or 100)
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    t = np.linspace(0, sim_time, 400)
    while True:
        try:
            threshold = float(input("Behavior prediction threshold (default 0.05): ") or 0.05)
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
    mem_sys = LinearMemorySystem(alpha, beta, bias=bias, threshold=threshold)
    solution = mem_sys.simulate(M0, t)
    behavior = mem_sys.predict_behavior(solution[-1])
    print(f"\nFinal Memory Strengths: " + ", ".join([f"{val:.3f}" for val in solution[-1]]))
    print(f"Predicted Behavior: {behavior}\n")


    mem_sys.plot_results(t, solution, M0)
    
    mem_sys.plot_all_phase_portraits(M0)

if __name__ == "__main__":
    main()
