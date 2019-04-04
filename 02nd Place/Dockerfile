FROM jupyter/minimal-notebook:59b402ce701d

USER root

RUN conda update -n base conda
COPY environment.yml .
RUN conda env update -f environment.yml

USER $NB_USER
