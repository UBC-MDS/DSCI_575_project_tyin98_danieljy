# DSCI 575 Group Project
This App is designed to help musical enthusiasts find the most appropriate suggestions for shopping on Amazon. 

## Getting Started
1. Clone the repo to your local device
```bash
git clone https://github.com/UBC-MDS/DSCI_575_project_tyin98_danieljy
cd DSCI_575_project_tyin98_danieljy
```
2. Create the environment to run the app
```bash
conda env create -f environment.yml
conda activate dsci575_td
```
3. Setup the index for running the models. Depending on your RAM, this could take a few minutes. You can increase the speed by reducing the number after '--max-products'. 
```bash
python src/utils.py --rebuild --max-products 5000
```
4. Run the Streamlit App locally.
```bash
streamlit run app/app.py
```

## Authors
- Daniel Yorke
- Tiantong Yin 

## Additional Notes
From your terminal within the project root:
-   You can run `python src/semantic.py -q "<your search query>" -k 10` to perform a test search of top 10 matching products using semantic search.
-   You can run `python src/bm25.py -q "<your search query>" -k 10` to perform a test search of top 10 matching products using BM25 search.

>*Some design choices: Semantic search included the reviews as part of the corpus but not BM25 because with basic keyword matching including the review might skew the model too much. We sorted the reviews by helpful_vote and only kept the good reviews (rating \>= 3).*

