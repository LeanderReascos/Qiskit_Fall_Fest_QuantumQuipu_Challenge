# QuantumChallenge
Reto para el Qiskit Fall Fest Latino, Escuela en Español de Computación Cuántica

### Reto: Decodificar y Clasificar

Este proyecto se enfoca en codificar los archivos [challenge_train.csv](https://github.com/QuantumQuipu/QuantumChallenge/blob/main/challenge_train.csv) y [challenge_test.csv](https://github.com/QuantumQuipu/QuantumChallenge/blob/main/challenge_test.csv) en al menos dos formas diferentes en un circuito cuántico (estas podrían ser codificaciones basadas en ángulos, amplitud, kernel, aleatorias o personalizadas).

---


# Solución
## Analisis de los datos

Los datos proporcionados para este problema constan de dos conjuntos de datos: uno de entrenamiento y otro de prueba. Ambos conjuntos están compuestos por 4 características y un objetivo. Las características son variables continuas, y el objetivo es una variable discreta que puede tomar dos valores, 0 o 1.

El tamaño de los conjuntos de datos es de 800 datos para el conjunto de entrenamiento y 200 para el conjunto de prueba.

Se analizaron los datos de entrenamiento y prueba para observar la distribución de los datos y la correlación entre las variables. Para este análisis, se presenta un mapa de calor de correlación entre las variables y cómo se distribuyen para encontrar posibles patrones que faciliten la clasificación.

### Correlación entre variables

![Correlación entre variables](Images/Correlation_heatmap.png)

### Distribución de los datos

![Distribución de los datos](Images/distribution.png)

Basándonos en estos resultados, es fácil observar que la característica 2 es la que presenta mayor correlación con el objetivo y que su distribución facilita la clasificación. Además, se puede concluir que no existe correlación en las demás características.

De esta manera, se siguieron dos métodos para el uso de los datos. El primero, y el más simple, consistió en descartar simplemente las características que no aportan información aparente para la clasificación. El segundo método consistió en utilizar la técnica de Análisis de Componentes Principales (PCA) para facilitar la clasificación.

Estas técnicas se utilizaron para la codificación de los datos en el circuito cuántico. La primera técnica se empleó con el primer método de codificación, mientras que la segunda técnica se utilizó con el segundo método de codificación.

## Método 1: Data Reuploading

Para la clasificación de datos, se utilizó el algoritmo de Data Reuploading, que implica la repetición de capas $L(\theta_i, x)$, donde $\theta_i$ es un vector de parámetros y $x$ es el vector de características. Cada capa consta de un bloque de codificación $S(x)$ y un bloque de parámetros entrenables $W(\theta_i)$. Al final de varias capas, se añade un bloque adicional de entrenamiento para garantizar la alternancia entre un bloque de codificación y uno de entrenamiento.

$$L(\theta_i,x) = S(x) W(\theta_i)$$

La elección de este método se basa en resultados recientes, como los de Maria Schuld et al., 2021 [^1], y Adrián Pérez-Salinas et al., 2020 [^2]. En [^2], se demuestra que un solo qubit puede actuar como un clasificador universal con el número adecuado de capas y el observable correcto. Por otro lado, [^1] revela que Data Reuploading es un aproximador universal de funciones continuas, siempre que estas puedan aproximarse en series de Fourier. El número de frecuencias accesibles está limitado por el número de capas de codificación, y las amplitudes de las frecuencias están definidas por todo el circuito.

Basándonos en esto y considerando la simplicidad del problema, es factible concluir que el problema puede resolverse utilizando solo un qubit y una capa, resultados que se mostrarán a continuación.

[^1]: https://arxiv.org/abs/2008.08605
[^2]: https://arxiv.org/abs/1907.02085

### Resumen del Entrenamiento

Se realizaron entrenamientos de varios circuitos utilizando PyTorch con el optimizador Adam. Se empleó un learning rate variable de 0.2 y un batch size de 100 durante 100 epochs. Se implementó una reducción del learning rate en un factor de 0.5 cada 10 epochs si el loss no mejoraba, deteniendo el entrenamiento después de 3 reducciones del learning rate sin mejora.

#### Función de Pérdida

Para establecer la función de costo, se definió un observable $O$ medible en el circuito, utilizando el proyector de los estados $\ket{0}$ y $\ket{1}$ para las etiquetas 0 y 1, respectivamente. La función de pérdida se diseñó para optimizar la fidelidad entre el estado resultante del circuito variacional y el estado objetivo. Se formuló como la media de la diferencia al cuadrado entre la fidelidad del estado objetivo y su valor máximo posible de 1.

$$Loss = \frac{1}{N} \sum_{i=1}^N (1 - \braket{\psi_i|\psi(x_i)})^2$$

Donde $\ket{\psi(x_i)}$ representa el estado resultante del circuito variacional y $\ket{\psi_i}$ es el estado objetivo.

El código completo utilizado para el entrenamiento se encuentra en el archivo `python_utils/data_reuploading_pennylane_pytorch.py`.

### Resultados

Se entrenaron 5 circuitos para comparar la calidad de la solución, la cantidad de recursos utilizados y cómo estos afectan a la solución. Los circuitos se diferencian en la cantidad de capas de codificación y la cantidad de qubits utilizados. Los resultados se almacenaron en la carpeta `best_params_pytorch/`.

Los resultados se muestran en la siguiente tabla:

| Circuito | Número de Qubits | Número de Layers | Número de Parámetros | Accuracy Train | Accuracy Test |
|----------|-------------------|-------------------|----------------------|-----------------|---------------|
| 1 | 1 | 1 | 6 | 0.9875 | 0.995 |
| 2 | 1 | 2 | 9 | 0.9925 | 0.995 |
| 3 | 1 | 3 | 12 | 0.9925 | 0.995 |
| 4 | 2 | 1 | 12 | 0.99 | 0.99 |
| 5 | 2 | 2 | 18 | 0.9925 | 0.995 |

El mejor circuito encontrado en términos de recursos cuánticos utilizados, es decir, el número de compuertas y qubits, es el Circuito 1. Utiliza un solo qubit y una sola capa de codificación. Este circuito logra una precisión del 0.995 en el conjunto de entrenamiento y del 0.9875 en el conjunto de prueba.

Sin embargo, es posible observar que el aumento de capas lleva a un aumento en la precisión de los datos de entrenamiento, pero no en los datos de prueba. Esto se debe a que el Circuito 1 ya ha alcanzado la máxima precisión posible para los datos de prueba, que es 0.995, es decir, un único valor mal clasificado.

Con estos resultados tambien es posible observar que se verifica los resultados de los papres [^1] y [^2], ya que se puede observar que con un solo qubit se puede resolver el problema de clasificacion como fue observado en [^2]. Ademas que la cantidad de frecuencias que es posible acceder esta limitada por la cantidad de capas de codificacion, como se observa en [^1].

Esto también se puede observar en el siguiente gráfico, que muestra el valor esperado del operador Pauli Z $\langle Z\rangle$ así como las probabilidades de medir los estados $\ket{0}$ y $\ket{1}$.

#### Circuito 1 - 1 Qubit, 1 Capa

![Circuito 1 - 1 Qubit, 1 Capa](Images/one_freq.png)

En resumen, tras analizar varios enfoques utilizando diferentes configuraciones de capas y qubits, se observó que aumentar la cantidad de qubits y agregar entrelazamiento no necesariamente mejora la precisión del circuito para este problema específico. La precisión máxima lograda previamente no se superó al incrementar las capas y qubits, indicando que la complejidad adicional no se traduce en una mejora significativa en la precisión. Esto se debe a la simplicidad del problema. Es importante destacar que la complejidad del modelo no siempre garantiza mejoras en el rendimiento, y cada problema puede requerir un enfoque particular.

