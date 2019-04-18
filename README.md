# CoxGWAS
## Genomewide Cox regression for deriving weights of polygenic hazard scores

## Intro

This is the core function to derive weights for polygenic hazard score (PHS). This is to put the polygenic prediction in the context of age-related disease process. The intuition is to achieve better age-dependent risk prediction by taking age-at-onset into account. The hazard, or the age dependent risk function, is estimated using survival analysis. This main core function is used for efficiently performing Cox GWAS, then the resulting spreadsheet can be used in standard pipelines of polygenic scores.


## Required input

The main input is the genotype data in plink binary and a spreadsheet with corresponding individual identifier. Please see help function to check the necessary input. Right now it only support plink binary file with keep functionality.
```
PyPHS --help
```

## Output

A spreadsheet with variant ID, hazard weights, and correponding P values from score tests

## Limitation 
 
Right now the algorithm depends on n by n risk matrix to calculate the p values based on score tests. This imposed a great memory constraint on the process so it is not biobank supportable unless you have a gigantic machine. It is possible to toggle this option off if the PHS is all you want but not individual p values for Cox GWAS. For future release, this feature will be improved.

## License

Under GNU GPL v3.0

