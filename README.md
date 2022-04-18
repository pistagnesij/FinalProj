Credit to: Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks.

Run on two different data sets:
https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset?datasetId=615374&sortBy=voteCount
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images?datasetId=7415&sortBy=voteCount

Data set structure edited for use on our classifier. Edited data found here:
https://drive.google.com/drive/folders/1HozK_S8a6Qzx1a9Cd_2SxLxzN_uyTNEL?usp=sharing

To run classifier, download all libraries included in both py files, unzip whichever dataset you want to run in the same 
directory as both .py files. Run FeaturesExtractionLayer.py first (takes a few minutes), then run xDNN.py.

Possible issue with library versions depending on the version of pytorch needed for your device. Past versions of a 
few libraries may need to be installed depending on your pytorch version.
