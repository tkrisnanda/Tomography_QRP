This GitHub repository contains codes and data used in our article 'Experimental demonstration of enhanced quantum tomography via quantum reservoir processing'. 
Available online: https://arxiv.org/abs/2412.11015

Here are the step-by-step explanations for use
1. Download all files 
2. The main codes are labelled aa0GH...py - aa7GH...py. TK_basics.py contains helful functions. The rest of the files are theoretically generated data and experimental data.
3. aa0GH...py creates optimized displacements via gradient-descent method, given a truncation dimension D.
4. aa1GH...py computes the idealised maps, given a truncation dimension, from the optimized displacements generated before. 
5. aa2GH...py simulates the target states for our system parameters with decoherence
6. aa3GH...py computes the learnt map, from experimental data
7. aa4GH...py computes the learnt map, from experimental data, with bootstrap (resampling the experimental data 20 times)
8. aa5GH...py reproduces the results presented in Fig. 3
9. aa6GH...py reproduces the results presented in Fig. 4
10. aa7GH...py reproduces the results presented in Fig. 5

Email: tanjungkrisnanda@gmail.com for any questions.
