import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Neural Network Intelligence: Tanh App",
    page_icon="🧠",
    layout="wide"
)

# =============================
# Sidebar: Parameter Control
# =============================
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("### 1. Visualization Settings")
x_min = st.sidebar.slider("Min x value", -10.0, 0.0, -5.0)
x_max = st.sidebar.slider("Max x value", 0.0, 10.0, 5.0)
num_points = st.sidebar.slider("Number of points", 10, 500, 100)

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Model Complexity")
neurons = st.sidebar.slider("Hidden Layer Neurons", 1, 50, 20)
epochs = st.sidebar.slider("Epochs", 50, 1000, 400)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.001, 0.01, 0.05, 0.1], value=0.01)

# =============================
# Header & Introduction
# =============================
st.title("Neural Network Architecture: Tanh Activation")
st.write(
    """
    This application explores the **Hyperbolic Tangent (Tanh)** activation function. 
    Tanh is a zero-centered function that maps inputs to a range between -1 and 1. 
    We demonstrate how this helps a neural network learn data that oscillates around zero, 
    providing stronger gradients than the Sigmoid function.
    """
)

# =============================
# Section 1: Interactive Tanh Visualization
# =============================
st.markdown("---")
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📈 Interactive Tanh Curve")
    x_val = np.linspace(x_min, x_max, num_points)
    # Tanh formula
    y_val = np.tanh(x_val)

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(x_val, y_val, color='#ff7f0e', linewidth=2, label="Tanh(x)")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("Input", fontsize=7)
    ax.set_ylabel("Output", fontsize=7)
    ax.tick_params(labelsize=6)
    st.pyplot(fig, use_container_width=False)

with col2:
    st.subheader("⚙️ Mathematical Framework")
    st.latex(r"f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}")
    st.info("""
        **Core Technical Advantages:**
        * **Zero-Centered Output:** Since outputs range from -1 to 1, the mean of the hidden layer stays closer to zero, which speeds up convergence.
        * **Stronger Gradients:** The slope of Tanh is steeper than Sigmoid, meaning the network learns more aggressively during backpropagation.
        * **Symmetric Mapping:** Treats negative and positive inputs with equal structural weight.
        * **Bounded Range:** Prevents activations from "exploding" by squashing them into a fixed range.
        """)

# =============================
# Section 2: Data-Driven Neural Network Model (Centered Wave)
# =============================
st.markdown("---")
st.subheader("🤖 Live Model Training: Solving a Centered Wave")
st.write(
    "Below, we generate an **oscillating wave** centered at zero. This highlights Tanh's ability to handle balanced negative and positive outputs.")

# 1. Generate Wave Data (-1 to 1 range)
X_np = np.linspace(-4, 4, 60).reshape(-1, 1)
y_np = np.tanh(0.8 * X_np) + np.random.normal(0, 0.07, X_np.shape)

X_torch = torch.from_numpy(X_np).float()
y_torch = torch.from_numpy(y_np).float()

# 2. Define Model Structure
model = nn.Sequential(
    nn.Linear(1, neurons),
    nn.Tanh(),  # Core focused component
    nn.Linear(neurons, 1)
)

# 3. Training and Result Plotting
if st.button('🚀 Execute Training Process'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    progress_bar = st.progress(0)
    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction = model(X_torch)
        loss = criterion(prediction, y_torch)
        loss.backward()
        optimizer.step()
        progress_bar.progress((epoch + 1) / epochs)

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        fig_res, ax_res = plt.subplots(figsize=(3, 2))
        ax_res.scatter(X_np, y_np, s=8, color='gray', alpha=0.6, label='Noisy Wave')
        with torch.no_grad():
            y_pred = model(X_torch).numpy()
        ax_res.plot(X_np, y_pred, color='red', linewidth=1.5, label='Tanh-NN Fit')
        ax_res.set_title(f"Fit achieved using {neurons} Neurons", fontsize=8)
        ax_res.tick_params(labelsize=6)
        ax_res.legend(prop={'size': 5})
        st.pyplot(fig_res, use_container_width=False)

    with res_col2:
        st.success(f"Training Complete!")
        st.metric("Final Training Loss (MSE)", f"{loss.item():.4f}")

        st.info("""
        **Understanding the Training Process:**
        * **Model Complexity:** Neurons combine multiple Tanh "S-curves" that can span both positive and negative values to fit the balanced wave.
        * **Epochs:** Tanh typically converges faster than Sigmoid because its zero-centered nature prevents gradients from shifting too far in one direction.
        * **Learning Rate:** Essential for navigating the steep gradients of Tanh. If too high, the model may oscillate; if too low, it may get stuck in saturation.
        * **Optimization (Adam):** Efficiently updates weights by utilizing Tanh's stronger gradient signals compared to Sigmoid.
        """)

st.markdown("---")
st.caption("BSD3513 Introduction to Artificial Intelligence | Lab 4 – Neural Networks")