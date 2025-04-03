This GitHub repository contains codes and data used in our article 'Experimental demonstration of enhanced quantum tomography via quantum reservoir processing'. 
Available online: https://arxiv.org/abs/2412.11015

Here are the step-by-step explanations for use
1. Download all files 
2. Move all the files in Data/D6_additional to Data/D6 (Due to file limit we could not upload them in the same folder)
3. The main codes are labelled aa0GH...py - aa7GH...py. TK_basics.py contains helful functions. The rest of the files are theoretically generated data and experimental data.
4. aa0GH...py creates optimized displacements via gradient-descent method, given a truncation dimension D.
5. aa1GH...py computes the idealised maps, given a truncation dimension, from the optimized displacements generated before. 
6. aa2GH...py simulates the target states for our system parameters with decoherence
7. aa3GH...py computes the learnt map, from experimental data
8. aa4GH...py computes the learnt map, from experimental data, with bootstrap (resampling the experimental data 20 times)
9. aa5GH...py reproduces the results presented in Fig. 3
10. aa6GH...py reproduces the results presented in Fig. 4
11. aa7GH...py reproduces the results presented in Fig. 5

Email: tanjungkrisnanda@gmail.com for any questions.
