**Core Idea:** We'll structure the presentation logically: Motivation -> Basic RNN -> Challenges -> Advanced RNN (LSTM) -> Applications & Practical Aspects.

**Topic Division Among 4 Members:**

1.  **Member 1: Introduction & Motivation**
    *   What is Sequence Data? Why is it different?
    *   Limitations of Standard Models (CNNs, Feedforward Nets, potentially HMMs briefly).
    *   Introduction to RNNs: The Core Idea (Recurrence, Memory).
    *   High-level diagram of an RNN.
    *   *Implicitly touches on:* Theory (basic concept).

2.  **Member 2: Vanilla RNN Mechanics & Theory**
    *   Detailed RNN Architecture (Unrolling in Time).
    *   Forward Propagation (Formulas & Explanation).
    *   Activation Functions used in RNNs.
    *   Backward Propagation Through Time (BPTT - Concept & Formulas).
    *   Diagrams: Unrolled RNN, Formula visualization.
    *   *Covers:* Theory (Vanilla RNN), Formulas (Vanilla RNN), Diagrams, Activation Functions.

3.  **Member 3: Challenges & LSTM/GRU**
    *   The Vanishing and Exploding Gradient Problem (Explanation & Diagrams).
    *   Introduction to Long Short-Term Memory (LSTM) as a solution.
    *   LSTM Architecture: Cell State, Gates (Forget, Input, Output) - (Diagrams & Explanation).
    *   LSTM Forward Pass (Conceptual, maybe key formulas if time allows).
    *   Brief mention of GRU as a simpler alternative.
    *   *Covers:* Theory (Challenges, LSTM, GRU), Formulas (LSTM - conceptual), Diagrams.

4.  **Member 4: Applications, Forecasting vs. Pattern Recognition & Code**
    *   Distinction: Pattern Recognition vs. Forecasting in Sequences (using RNN examples).
    *   Key Applications of RNNs (Text Generation, Machine Translation, Speech Recognition, Handwriting Recognition, Video Analysis, etc.).
    *   Numerical/Conceptual Example (like Binary Addition from Hinton's slides - explain the *idea* rather than deep math).
    *   Simple Code Example (TensorFlow/Keras - showing RNN/LSTM layer usage).
    *   Conclusion & Summary.
    *   *Covers:* Pattern Recognition vs. Forecasting, Applications, Numerical (conceptual), Code.

---

**Presentation Slides Content (Based on PDFs):**

Here's a draft slide structure incorporating content from your PDFs.

**(Member 1: Introduction & Motivation)**

*   **Slide 1: Title Slide**
    *   Title: Recurrent Neural Networks: Understanding Sequential Data
    *   Group Members' Names
    *   Course Name/Number

*   **Slide 2: What is Sequential Data?**
    *   Data where order matters: Text, Speech, Time Series, Video Frames, Music.
    *   Goal: Model dependencies across time/sequence steps.
    *   *(Ref: Wang Slide 5, Hinton Slide 2)*
    *   Diagram: Show examples like a sentence, audio wave, stock chart. *(Ref: Wang Slide 5)*

*   **Slide 3: Why Not Standard Networks? (Limitations)**
    *   **Feedforward Nets/CNNs:** Fixed input/output size, assume independence between inputs/outputs (or limited local dependence for CNNs). Struggle with variable-length sequences and long-range dependencies.
    *   *(Ref: Wang Slides 4, 7, 8; Hinton Slide 3)*
    *   Diagram: Show FFNN/CNN structure vs. a sequence task. *(Ref: Hinton Slide 3, Wang Slide 8)*
    *   **Memoryless Models (Autoregressive):** Limited history window. *(Ref: Hinton Slide 3, Wang Slide 8)*
    *   **(Optional) HMMs/LDS:** Limited memory capacity (HMMs - log(N) bits), strict statistical assumptions. *(Ref: Hinton Slides 5, 6, 7)*

*   **Slide 4: Introducing Recurrent Neural Networks (RNNs)**
    *   Designed specifically for sequential data.
    *   Key Idea: Maintain an internal "memory" or "state" (`h_t`) that captures information from past steps.
    *   The same weights are applied at each time step (parameter sharing).
    *   *(Ref: Hinton Slide 8, Wang Slide 10, 11)*

*   **Slide 5: RNN Core Concept: Recurrence**
    *   Diagram: Simple RNN loop (input `x_t`, hidden state `h_t`, output `y_t`). Show the hidden state feeding back into itself.
    *   Explain: The output/hidden state at time `t` depends on input `x_t` AND the hidden state `h_{t-1}` from the previous step.
    *   `h_t = f(W_{hh} * h_{t-1} + W_{xh} * x_t)`
    *   `y_t = g(W_{hy} * h_t)` (Simplified notation)
    *   *(Ref: Wang Slide 10 diagram, Hinton Slide 8 diagram)*

**(Member 2: Vanilla RNN Mechanics & Theory)**

*   **Slide 6: RNN Architecture: Unrolling in Time**
    *   Diagram: Show the looped RNN unrolled across several time steps (t-1, t, t+1). Emphasize shared weights (W_hh, W_xh, W_hy).
    *   Explain: This visualization makes it look like a deep feedforward network, allowing standard backpropagation (modified).
    *   *(Ref: Hinton Slide 11, Wang Slide 16)*

*   **Slide 7: Forward Propagation in a Vanilla RNN**
    *   Goal: Compute hidden states and outputs for each time step.
    *   Formulas:
        *   `a_h^t = Σ_i (W_{ih} * x_i^t) + Σ_{h'} (W_{h'h} * b_{h'}^{t-1})` (Input to hidden)
        *   `b_h^t = θ_h(a_h^t)` (Hidden activation)
        *   `a_k^t = Σ_h (W_{hk} * b_h^t)` (Input to output)
        *   `y_k^t = θ_k(a_k^t)` (Output activation, e.g., softmax for classification)
    *   Explanation: Step-by-step calculation from t=1 to T.
    *   *(Ref: Wang Slides 13, 14)*
    *   Diagram: Highlight the flow in the unrolled diagram.

*   **Slide 8: Activation Functions in RNNs**
    *   Hidden Units (`θ_h`): Typically `tanh` (hyperbolic tangent) or sometimes `ReLU`. `tanh` is common as it's zero-centered (-1 to 1).
    *   Output Units (`θ_k`): Depends on the task.
        *   `Softmax` for predicting probabilities over classes (e.g., next word).
        *   `Sigmoid` for binary classification or gates (in LSTM/GRU).
        *   `Linear` for regression tasks.
    *   *(Ref: Hinton Slide 22 - mentions squashing functions, Wang Slides 13, 27, 28 - use θ, f, g)*

*   **Slide 9: Backward Propagation Through Time (BPTT)**
    *   Goal: Calculate gradients of the error with respect to weights.
    *   Concept: Apply chain rule back through the unrolled network.
    *   Key Challenge: Gradients flow back not just "vertically" from output to input, but also "horizontally" through the recurrent connections (`h_t` depends on `h_{t-1}`).
    *   *(Ref: Hinton Slide 13, Wang Slide 15, 16)*
    *   Diagram: Show error signal propagating backward in the unrolled diagram. *(Ref: Wang Slide 16)*

*   **Slide 10: BPTT Formulas (Conceptual)**
    *   Error at time `t` depends on the output at time `t` AND the error propagated from time `t+1` via the hidden state.
    *   Formula (Delta rule for hidden state): `δ_h^t = θ'(a_h^t) * [ Σ_k (δ_k^t * W_{hk}) + Σ_{h'} (δ_{h'}^{t+1} * W_{hh'}) ]`
    *   Weight updates are summed across all time steps: `ΔW = Σ_t (gradient at time t)`
    *   *(Ref: Wang Slides 16, 17 - Show the structure, maybe not full derivation)*
    *   *(Ref: Hinton Slide 12 for weight sharing/constraints)*

*   **(Optional) Slide 11: Bidirectional RNNs (BiRNNs)**
    *   Concept: Process sequence forward and backward using two separate hidden states. Combine states before final output.
    *   Benefit: Output at time `t` can depend on both past and future context. Useful for tasks like translation or named entity recognition.
    *   Diagram: Show BiRNN structure. *(Ref: Wang Slide 18, 19)*

**(Member 3: Challenges & LSTM/GRU)**

*   **Slide 12: The Challenge: Vanishing Gradients**
    *   Problem: During BPTT, gradients are repeatedly multiplied by the recurrent weight matrix (`W_hh`).
    *   If weights/gradients are small (<1), the gradient signal shrinks exponentially as it propagates back through time.
    *   Effect: Network cannot learn long-range dependencies; influence from early inputs is lost.
    *   *(Ref: Hinton Slide 23, Wang Slide 21, 22)*
    *   Diagram: Show gradient signal fading over time steps. *(Ref: Wang Slide 21, Hinton Slide 23 visualization, Wang Slide 26 Fig 4.1)*

*   **Slide 13: The Challenge: Exploding Gradients**
    *   Problem: If weights/gradients are large (>1), the gradient signal grows exponentially.
    *   Effect: Leads to unstable training (NaN values).
    *   Solution: Gradient Clipping (limit the max value of the gradient).
    *   *(Ref: Hinton Slide 23, 24, Wang Slide 21, 22)*

*   **Slide 14: Solution: Long Short-Term Memory (LSTM)**
    *   Hochreiter & Schmidhuber (1997).
    *   Goal: Explicitly designed to combat vanishing gradients and model long-range dependencies.
    *   Key Idea: Introduce a separate "cell state" (`C_t`) and "gates" to control information flow.
    *   *(Ref: Hinton Slide 25, 26, Wang Slide 24)*

*   **Slide 15: LSTM Architecture: Gates & Cell State**
    *   Diagram: Standard LSTM cell diagram. *(Ref: Hinton Slide 27, Wang Slide 25)*
    *   **Cell State (`C_t`):** The "memory highway". Information flows along it with minimal changes, regulated by gates.
    *   **Forget Gate (`f_t`):** Decides what info to throw away from the cell state (Sigmoid).
    *   **Input Gate (`i_t`):** Decides which new info to store in the cell state (Sigmoid + tanh).
    *   **Output Gate (`o_t`):** Decides what part of the cell state to output (Sigmoid + tanh applied to filtered cell state).
    *   *(Ref: Hinton Slides 26, 27, 28, Wang Slides 25, 27)*

*   **Slide 16: LSTM Information Flow (Conceptual Forward Pass)**
    *   1. Decide what to forget (`f_t * C_{t-1}`).
    *   2. Decide what new info to store and calculate candidate values (`i_t * \tilde{C}_t`).
    *   3. Update cell state (`C_t = f_t * C_{t-1} + i_t * \tilde{C}_t`).
    *   4. Decide what to output (`o_t * tanh(C_t)`).
    *   Explain how this gating mechanism allows gradients to flow better (identity path through cell state when gates are set appropriately).
    *   *(Ref: Hinton Slide 28 shows backprop flow)*
    *   *(Ref: Wang Slide 27 shows forward formulas)*

*   **Slide 17: Gated Recurrent Unit (GRU)**
    *   A simpler alternative to LSTM (Cho et al., 2014).
    *   Combines forget and input gates into an "Update Gate".
    *   Merges cell state and hidden state.
    *   Fewer parameters, computationally slightly cheaper. Often performs similarly to LSTM.
    *   Diagram: Show GRU cell. *(Ref: Wang Slide 33)*

**(Member 4: Applications, Forecasting vs. Pattern Recognition & Code)**

*   **Slide 18: Pattern Recognition vs. Forecasting with RNNs**
    *   **Forecasting:** Predicting future elements in a sequence based on past elements.
        *   Examples: Next word prediction (Language Modeling), Stock price prediction, Weather forecasting.
        *   *(Ref: Hinton Slide 2 - "predict the next term")*
    *   **Pattern Recognition:** Identifying patterns or classifying sequences/sub-sequences.
        *   Examples: Speech Recognition (sequence -> text), Sentiment Analysis (sequence -> class), Action Recognition in Videos (sequence -> action label), Handwriting Recognition (sequence -> text).
        *   *(Ref: Hinton Slide 2 - "turn input sequence into output sequence in different domain", Hinton Slides 29, 30)*
    *   RNNs excel at both due to their ability to model temporal dependencies.

*   **Slide 19: Application: Language Modeling & Text Generation**
    *   Task: Predict the next word/character given previous ones.
    *   RNN reads sequence, hidden state summarizes context, output layer predicts next element.
    *   Can be used to generate new text.
    *   *(Ref: Hinton Slides 39-50 - Character RNN, Wang Slide 5)*
    *   Example: Show generated text snippet *(Ref: Hinton Slide 48, 49)*

*   **Slide 20: Application: Machine Translation**
    *   Sequence-to-Sequence (Seq2Seq) models often use two RNNs (Encoder-Decoder).
    *   Encoder RNN reads source sentence into a context vector.
    *   Decoder RNN generates target sentence from the context vector.
    *   *(Ref: Wang Slide 5)*

*   **Slide 21: Application: Speech & Handwriting Recognition**
    *   Input: Sequence of audio features or pen coordinates.
    *   Output: Sequence of characters or words.
    *   BiRNNs and LSTMs are crucial here for context.
    *   *(Ref: Hinton Slides 29, 30)*
    *   Diagram/Video Link: Alex Graves' handwriting demo *(Ref: Hinton Slide 31)*

*   **Slide 22: Application: Video Analysis / Sports Analytics**
    *   Input: Sequence of video frames or player trajectory data.
    *   Output: Action classification, event detection, trajectory prediction.
    *   *(Ref: Wang Slides 31, 32)*
    *   Diagram: Show example from Wang's slides. *(Ref: Wang Slide 31 or 32)*

*   **Slide 23: Numerical/Conceptual Example: Binary Addition**
    *   Task: Add two binary numbers, processed bit-by-bit from right to left.
    *   RNN needs to remember the "carry" bit.
    *   Hidden state represents the state (e.g., "no carry" or "carry").
    *   Demonstrates how hidden state acts as memory.
    *   *(Ref: Hinton Slides 17-21)*
    *   Diagram: Finite state automaton for binary addition. *(Ref: Hinton Slide 18)*

*   **Slide 24: Code Example: Simple RNN/LSTM Layer (TensorFlow/Keras)**
    *   Show 5-6 lines calling the relevant function.
    *   **TensorFlow Example (from Wang's slides):**
        ```python
        import tensorflow as tf
        # Define LSTM cell
        lstm_cell = tf.keras.layers.LSTMCell(units=128) # Or SimpleRNNCell
        # Initial state (example for one batch item)
        initial_state = lstm_cell.get_initial_state(batch_size=1, dtype=tf.float32)
        # Input at one time step (example)
        inputs = tf.random.normal([1, input_features])
        # Run one step
        output, state = lstm_cell(inputs, states=initial_state)
        ```
    *   **Keras Sequential API Example:**
        ```python
        from tensorflow import keras
        from tensorflow.keras import layers
        model = keras.Sequential()
        # Add an LSTM layer
        model.add(layers.LSTM(64, input_shape=(timesteps, features))) # Or SimpleRNN
        # Add output layer
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.summary() # Show model structure
        ```
    *   *(Ref: Wang Slides 35, 36 - adapt for simplicity)*

*   **Slide 25: Conclusion**
    *   RNNs are powerful tools for sequential data.
    *   They use internal memory (hidden state) to capture dependencies.
    *   Vanilla RNNs suffer from gradient issues.
    *   LSTMs and GRUs provide solutions using gating mechanisms.
    *   Wide range of applications across various domains.

*   **Slide 26: Q & A / Thank You**
    *   Title: Questions?
    *   Thank you!
    *   *(Ref: Wang Slide 41)*

---
This structure provides a logical flow, covers all the requested topics, divides the work relatively evenly, and leverages the content from both provided PDF sources. Remember to add speaker notes and practice the transitions between members. Good luck!
