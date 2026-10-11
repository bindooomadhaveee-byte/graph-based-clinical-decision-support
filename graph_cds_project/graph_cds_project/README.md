# Graph-Based Clinical Decision Support

A simple base project that models medical knowledge as a graph (NetworkX):

    Symptom -> Disease -> Test
                 |
                 +-> Drug -> Drug (interactions)
                       |
                       +-> Condition (contraindications)

For each patient it ranks likely diseases, suggests tests, and recommends
treatments after filtering out unsafe drugs (allergy, contraindication,
major drug interaction).

## Run
    pip install -r requirements.txt
    python graph_cds.py            # CLI demo with 3 patients
    python graph_cds.py --plot     # also draws the knowledge graph

## Disclaimer
Educational demo only. Not for real clinical use.
