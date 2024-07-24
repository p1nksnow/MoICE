# MoICE
Official implementation for "Mixture of In-Context Experts Enhance LLMs’ Awareness of Long Contexts"

# Getting Started
Let’s take Llama2-7b-chat as an example.
1. Create a virtural environment from requirements.txt.
   
   ```pip install -r requirements.txt```
3. Replace original modeling_llama.py with our modeling_llama.py with MoICE.
4. Replace paths in train.sh and train Llama2-7b-chat with MoICE.
   
   ```bash train.sh```


# Test
we take the open long-context benchmark [Leval](https://github.com/OpenLMLab/LEval) as our main evaluation.



## Citation
```
@article{lin2024mixture,
  title={Mixture of In-Context Experts Enhance LLMs' Long Context Awareness},
  author={Lin, Hongzhan and Lv, Ang and Chen, Yuhan and Zhu, Chen and Song, Yang and Zhu, Hengshu and Yan, Rui},
  journal={arXiv preprint arXiv:2406.19598},
  year={2024}
}
```
