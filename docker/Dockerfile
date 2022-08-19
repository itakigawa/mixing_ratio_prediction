FROM jupyter-cuda/base-notebook

USER root

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    build-essential \
    less \
    gcc \
    g++ \
    gfortran \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}
WORKDIR "${HOME}"


# from scipy-notebook
RUN mamba install --quiet --yes -c rdkit rdkit && \
    mamba install --quiet --yes \
    'altair' \
    'beautifulsoup4' \
    'bokeh' \
    'bottleneck' \
    'cloudpickle' \
    'conda-forge::blas=*=openblas' \
    'cython' \
    'dask' \
    'dill' \
    'h5py' \
    'ipympl'\
    'ipywidgets' \
    'matplotlib-base' \
    'numba' \
    'numexpr' \
    'pandas' \
    'patsy' \
    'protobuf' \
    'pytables' \
    'scikit-image' \
    'scikit-learn' \
    'scipy' \
    'seaborn' \
    'sqlalchemy' \
    'statsmodels' \
    'sympy' \
    'widgetsnbextension'\
    'xlrd' && \
    mamba clean --all -f -y


# user install
RUN pip install --quiet --no-cache-dir --upgrade \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu115

RUN pip install --quiet --no-cache-dir --upgrade \
    tensorboard \
    nbconvert \
    nbdime \
    plotly \
    kaleido \
    openpyxl \
    shap \
    flake8 \
    black \
    isort \
    autopep8 \
    pydocstyle \
    pylint \
    rope \
    yapf \
    scikit-optimize \
    hydra-core mlflow \
    opencv-python \
    opencv-rolling-ball \
    xgboost \
    lightgbm \
    nltk \
    line_profiler \
    memory_profiler \
    humanfriendly \
    more-itertools \
    py3dmol

RUN pip install --quiet --no-cache-dir \
  git+https://github.com/rwightman/pytorch-image-models.git \
  albumentations

RUN pip install -U numpy
