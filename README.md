# Biodiversity-Exploratories-Knowledge-Graph
This repository contains files for the knowledge graph developed for the Biodiversity Exploratories (https://www.biodiversity-exploratories.de/en/), that consists of publication and dataset metadata of research endeavors related to the BE.

The Mappings and Outputs folder contains the R2RML mappings to create the knowledge graph, and the resulting KG triples. 

The Code folder contains a notebook to pull metadata from the BExIS (Biodiversity Exploratories Information System) API, and project folders for LLM applications on the metadata.

First, we investigate whether embedding models can be utilized to extract latent information from publication and dataset titles and abstracts by clustering the embedded documents.
Then, we provide anchor concepts to the embedding space in the form of the NASA EARTHDATA GCMD earth science keywords (https://gcmd.earthdata.nasa.gov/KeywordViewer/scheme/all?gtm_scheme=all), and investigate which concepts clusters of documents form around.
Finally, we embed the research goals of the BE (https://www.biodiversity-exploratories.de/en/about-us/research-objectives-and-background/) as anchors in the embedding space and assign documents to them.

The second LLM application is concerned with the extraction of metadata categories from publication and dataset titles and abstracts to investigate whether structured information can be extracted without further human effort and which metadata categories are well suited for this task.
