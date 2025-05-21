# Sequence-to-Sequence Models (With vs. Without Attention)

This report compares two encoder–decoder (seq2seq) models trained for transliteration using the Dakshina Hindi dataset. One model uses a **basic RNN encoder–decoder** (without attention), and the other uses the same architecture augmented with **Bahdanau (additive) attention**. We describe the data and preprocessing, detail each model’s architecture, training setup, and then compare their performance.

---

##  Dataset and Preprocessing

We use the **Dakshina transliteration corpus (Hindi)**, which contains parallel text: Latin-script (Roman) transliterations and native-script Hindi. Each line in the dataset follows:

```
<Devangari_Hindi>\t<Latin_Transliteration>\t<attestation_flag>
```

* The **native-script Hindi** is the target, and the **Latin transliteration** is the input.
* Target sequences are wrapped with start (`\t`) and end (`\n`) tokens.
* Vocabulary is constructed at the character level for both input and target.
* Padding or one-hot encoding is used depending on the model.

### No-Attention Preprocessing:

* Uses one-hot encoded 3D arrays for encoder and decoder inputs.
* Decoder targets are shifted by one timestep.

### Attention Model Preprocessing:

* Uses index-mapped padded sequences.
* Character indices are later embedded via an `Embedding` layer.
* Decoder outputs are one-hot encoded for loss calculation.

---

##  Model Architectures

###  Encoder–Decoder without Attention

* Implements a **vanilla RNN-based seq2seq model**.
* Recurrent cells can be **LSTM**, **GRU**, or **SimpleRNN** (configurable).
* Encoder provides the final hidden states to initialize the decoder.
* Decoder uses a dense layer to produce character probabilities.

#### Simplified Structure:

```python
# Encoder
x = OneHot(input) → RNN Layers → encoder_states

# Decoder
y = OneHot(target_input)
for i in decoder_layers:
    y = RNN(y, initial_state=encoder_states[i])
decoder_output = Dense(y)
```

* No dynamic context—decoder uses fixed encoder state only.
* Uses beam search during inference for better predictions.

---

###  Encoder–Decoder with Bahdanau Attention

* Adds **Bahdanau attention** to dynamically focus on encoder outputs.
* Uses embeddings instead of one-hot encodings.
* At each decoding step, computes a weighted sum of encoder outputs.

#### Simplified Structure:

```python
# Encoder
x = Embedding(input) → RNN Layers → encoder_outputs

# Decoder
y = Embedding(target_input) → RNN Layers (init from encoder)
attention_output = Attention([encoder_outputs, decoder_outputs])
decoder_concat = Concatenate([decoder_outputs, attention_output])
decoder_output = TimeDistributed(Dense(decoder_concat))
```

* Enables better alignment between input and output.
* Achieves better results for longer sequences or variable alignment patterns.


## Evaluation and Metrics

* **Character-Level Accuracy** (Keras built-in).
* **Word-Level Accuracy** via beam search decoding.
  * Measures n-gram overlap with reference.


## ✅ Conclusion

We implemented and compared two seq2seq models on the Hindi transliteration task:

* A **vanilla encoder–decoder**.
* An **attention-based model** using Bahdanau attention.

The attention mechanism improves training and inference by allowing the model to focus on relevant input parts. This results in better convergence and higher transliteration accuracy, especially for longer or more complex sequences.

---

## 🛠️ How to Run

1. **Open Jupyter Notebooks:**
   Use Google Colab or Jupyter Lab.

2. **Install Dependencies:**

   ```bash
   pip install wandb
   ```

3. **Mount Google Drive (if required):**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Run Preprocessing Cells:**
   Prepares data and encodes sequences.

5. **Train the Model:**

   * For the no-attention model, call:

     ```python
     model = Seq2SeqModel(...)
     model.build_and_train()
     ```
   * For the attention model, start the sweep or run:

     ```python
     model = build_model(...)
     train_model(...)
     ```

6. **Evaluate:**
   Use `decode_sequence` or beam search methods to infer outputs.

7. **View Logs:**
   Metrics like accuracy and word accuracy are logged via `wandb`.

---

