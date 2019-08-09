# Representational similarity analysis
### Tutorial for Methods in Neuroscience at Dartmouth (MIND) 2018

Representational similarity analysis (RSA) is statistical technique based on analyzing second-order isomorphisms. That rather than directly analyzing the relationship between one measure and another, RSA instead computes some measure of similarity within each measure and then compares these similarities to each other. RSA was pioneered by [Kriegeskorte, Mur, and Bandettini (2008, Frontiers in System Neuroscience)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2605405/) and has since become a popular method for analyzing neuroimaging data. Much of this popularity is driven by the fact that - because RSA focuses on second-order isomorphisms (i.e., similarities) - it is an incredibly flexible analytic technique, capable linking disparate measures of brain and behavior.

![Kriegeskorte, Mur, and Bandettini (2008)](http://www.mrc-cbu.cam.ac.uk//personal/nikolaus.kriegeskorte/fig5_kriegeskorte_RSA_FNS.gif)

In the context of fMRI, RSA usually takes the form of a correlation or regression between neural pattern similarity and a task, rating, or model. In this tutorial we will learn how to conduct these confirmatory RSAs as well as how
to perform complementary exploratory analyses.

### Installation

First, if you have not already done so, please install [R](https://cran.r-project.org/) and [RStudio](https://www.rstudio.com/products/rstudio/download/#download).

Clone this repository to your local machine, and open "Representational_similarity_analysis.Rmd" with RStudio. From the "Run" dropdown menu in the upper left pane, select "Run All". When the script completes, save the .Rmd file, and then select "Preview" to view the R Notebook. To make changes, edit the code in the .Rmd file, re-run the corresponding code-chunk(s), and then save changes - the preview will update automatically.

Alternatively, a static version of the R Notebook can be viewed directly through your web browser by downloading the file ending in ".nb.html" and then opening it with the browser of your choice. Viewing this version does not require R/RStudio, but changes cannot be made to the code.



