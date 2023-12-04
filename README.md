# amazonOptimiser -- Amazon Price Optimizer
Final project for my Data Science bootcamp

This is a linear regression project to help set the price of a listing on Amazon.
It is realized that revenue losses can come just from the action of setting a price. This tool takes the costs of similar items to best find a price for your listing.

The model is trained on high performing Amazon UK products - those that have high ratings and receive high numbers of reviews.

The accuracy is dependant on how many words in the title are converted into numbers, and how many numbers.
More words -> more numbers = more accuracy, but comes at the risk of over-fitting. Therefore a balance must be struck.


Please find the Kaggle dataset attached below. This is data scraped from 2.2 million+ products on Amazon UK.
There has been no similar public projects that use this data; therefore, this is a unique project.  

https://www.kaggle.com/datasets/asaniczka/amazon-uk-products-dataset-2023/download?datasetVersionNumber=4
