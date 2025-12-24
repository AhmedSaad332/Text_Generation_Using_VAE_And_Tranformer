# Shakespeare Text Generation using VAE + Transformer

This project implements an AI-powered text generation system that learns the writing style of William Shakespeare and generates new, coherent literary text using a hybrid Variational Autoencoder (VAE) and Transformer architecture.

The goal of the project is to explore generative language models and understand how probabilistic latent representations can be combined with attention-based models to produce creative and stylistically rich text.

---

## Project Overview

- Built a text generation model using a Variational Autoencoder combined with a Transformer
- Trained on a Shakespeare text dataset
- Applied extensive text preprocessing and normalization
- Used word embeddings to capture semantic and syntactic relationships
- Learned latent representations of text to generate stylistically consistent outputs

---

## Key Concepts

- Natural Language Processing (NLP)
- Deep Learning
- Variational Autoencoders (VAE)
- Transformers and Attention Mechanisms
- Text preprocessing and tokenization
- Generative language models

---

## Model Architecture

- Encoder: Encodes input text into a latent space using a VAE
- Latent Space: Captures probabilistic representations of writing style
- Decoder: Uses Transformer-based attention to generate text sequences
- Embeddings: Word embeddings are used to improve semantic understanding

---

## Dataset

- Shakespeare text corpus
- Preprocessed by:
  - Lowercasing
  - Removing unnecessary symbols
  - Tokenization
  - Vocabulary construction

---

## Results

- The model is capable of generating Shakespeare-like text
- Maintains a balance between creativity and coherence
- Demonstrates the effectiveness of combining VAEs with Transformers for text generation

---

## GUI Access

GUI for interacting with the model:  
https://huggingface.co/spaces/ahmed3746887278237832/TextGeneration_Using_VAE
---

## Team Members

- Marwan Megahed  
- Aml Ashraf  
- Raghad Hany  

---

## Future Improvements

- Improve text coherence with larger datasets
- Experiment with different latent space dimensions
- Fine-tune Transformer hyperparameters
- Add temperature and sampling controls in the GUI

---

## License

This project is for educational and research purposes.
