# MoICE
Official implementation for "Mixture of In-Context Experts Enhance LLMs’ Awareness of Long Contexts"

# Getting Started
Let’s take Llama2-7b-chat as an example.
1. Create a virtural environment from requirements.txt.
   ```pip install -r requirements.txt```
3. Replace original modeling_llama.py with our modeling_llama.py which use MoICE.
4. Replace paths in train.sh and train Llama2-7b-chat with MoICE.
   ```bash train.sh``
