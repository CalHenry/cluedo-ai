# Cluedo for AI 🕵️‍♂️

This branch is a version of the code to make it runnable on a kaggle notebook.  Please go to the [main branch README](https://github.com/CalHenry/cluedo-ai/blob/main/README.md) for project presentation.  
It has minor changes compared to the main branch (mostly fixes - 1 actually)

The branch exist so i can keep 2 versions of the workflow independant.  
Kaggle uses GPU with cuda and require a GGUF quantized model and not the MLX i am using locally.  
Turns out that GGUF and MLX are different processus that can impact behavior even for the same model and quantization.  

I had to fix an issue with the GGUF model not responding with the correct structured output.  
Fix: Allow the supervisor agent to retry its anwer if incorrect.
