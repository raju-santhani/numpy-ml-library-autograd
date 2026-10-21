# mlp_visual_test.py
import numpy as np
from my_ml_lib.nn.autograd import Value
from visualize import draw_dot

# ---- Example: One forward pass of a tiny MLP ----

# Input vector (1x2)
x = Value(np.array([[1.0, -2.0]]), label='x')

# Linear layer weights (2x3) and bias (1x3)
W = Value(np.random.randn(2, 3), label='W')
b = Value(np.zeros((1, 3)), label='b')

# Linear transformation: xW + b
z = x @ W + b
z.label = 'z'

# Activation: ReLU
a = z.relu()
a.label = 'a'

# Output node (sum of activations)
y = a.sum()
y.label = 'y'

# ---- Visualize the computation graph ----
dot_graph = draw_dot(y, format='png')
dot_graph.render('mlp_forward_pass', format='png', view=True)

