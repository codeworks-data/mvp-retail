# Use Case #1 : A Glimpse Into The World of Retail

## In a nutshell
In this project, we tried to understand the consumers behaviour and anticipate it, using a combination of data analysis and machine learning.


This code details the work done on the retail dataset above, obtained from kaggle.
It contains the data exploration and the customer segmentation analysis as well as recommendation systems.

We have also written an article on it: [https://medium.com/codeworksparis/use-case-1-a-glimpse-into-the-world-of-retail-821bea55d5c0][1]


## Table of contents
* [Data source](#data-source)
* [Methodology](#methodology)
* [Technologies](#technologies)
* [Files](#Files)
* [Contributors](#contributors)

## Data source
We use a Kaggle [retail store dataset].

It describes the day-to-day transactions for a set of customers over a period of three years.

## Methodology
To build our recommender, we used and compared two recommendations systems to our baseline,
that made the most sense to us.
- Content-based
- Collaborative-filtering
	
## Technologies
|  Technologies | Version  |
|---|---|
|  python + pytest |  3.7 |
|  scikit-learn |  0.23.2  |
|  seaborn |  3.3.4  |
|  implicit |  0.4.4 |
|  ml-metrics | 0.1.4 |
|  pandas | 1.2.1 |
|  numpy | 1.20.0 | 
| recmetrics | 0.0.12 |

## Files

### Notebooks
- Retail_Case_study.ipynb: is the notebook that contains the EDA we have done, and where insights in the part 1 of the article were taken from.
- Evaluation.ipynb: is the notebook where we build all our recommenders (baselines included), and where we evaluated them on the test test.
### Utils
This directory contains all recommenders (baselines included) code base, one recommender by file. Files names are self-descriptive:
- content_based_recommender.py
- collaborative_filtering_matrix_factorization.py
- baseline_last_sold_recommender.py
- baseline_most_sold_recommender.py
### Tests
This directory contains tests on utils codebase.



## Contributors
* [Adnene Tekaya](https://github.com/atekaya)
* [Denise NGUYEN](https://github.com/nise2)
* [Hajar AIT EL KADI](https://github.com/HAEKADI)
* [Koffi Cornelis](https://github.com/CorKof)

[1]: https://medium.com/codeworksparis/use-case-1-a-glimpse-into-the-world-of-retail-821bea55d5c0
[2]: https://www.kaggle.com/darpan25bajaj/retail-case-study-data




