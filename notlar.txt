# Makale
https://www.sciencedirect.com/science/article/pii/S2352340925008637#bib0015

# Veriseti
https://data.mendeley.com/datasets/t9hgvk2h9p/1

# Conda kurulum
conda create -n pamuk_projesi -c conda-forge python=3.11 tensorflow opencv scikit-learn matplotlib pillow tqdm

# C++ çökme hatası sonucu tensorflow sürüm düşürme 2.15.0 sürümünde çalıştı
conda install -n pamuk_projesi -c conda-forge tensorflow=2.15.0