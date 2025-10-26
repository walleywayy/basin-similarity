.PHONY: caravan_demo caravan_similarity

caravan_demo:
	python scripts/fetch_caravan_data.py

caravan_similarity:
	jupyter nbconvert --execute notebooks/caravan_similarity.ipynb