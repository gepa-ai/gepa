# Plot Script `real_cost.ipynb`

## LaTeX Dependencies (for high-quality text rendering in plots)
If you want to use LaTeX for text rendering in plots (via `plt.rcParams['text.usetex'] = True`), you'll need to install the following LaTeX packages:

```bash
sudo apt-get install texlive-latex-base texlive-latex-recommended \
                     texlive-science texlive-latex-extra \
                     texlive-fonts-recommended cm-super dvipng
```

Alternatively, you can disable LaTeX rendering by setting `plt.rcParams['text.usetex'] = False` in your code.