from shiny import render
import matplotlib.pyplot as plt
import numpy as np

def server(input, output, session):
    
    @output
    @render.text
    def txt():
        return f"Seleccionaste {input.n()} observaciones"
    
    @output
    @render.plot
    def plot():
        x = np.random.randn(input.n())
        plt.hist(x, bins=20)
        plt.title(f"Histograma con {input.n()} valores")
        return plt.gcf()