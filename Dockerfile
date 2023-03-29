# Copyright Adrien Laurent 2022
# All rights reserved.

FROM pytorch/pytorch
RUN apt-get update && apt-get -y upgrade && apt-get -y install sudo vim git screen locales wget 
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8
RUN pip install transformers==4.8.1; pip install  sentencepiece==0.1.95; pip install  textwrap3==0.9.2; pip install --quiet nltk==3.2.5; pip install --quiet ipython-autotime
RUN pip install --quiet git+https://github.com/boudinfl/pke.git@dc4d5f21e0ffe64c4df93c46146d29d1c522476b && pip install --quiet flashtext==2.7
RUN pip install keybert==0.2.0
RUN pip install strsim==0.0.3
#spacy must be 2.3.7
RUN pip install spacy==2.3.7
#sense2vec install spacy dependency but it installs 3.x
RUN pip install sense2vec==1.0.2
RUN pip install ebooklib
RUN echo "import nltk; nltk.download('punkt')" | python3
RUN echo "import nltk; nltk.download('brown')" | python3
RUN echo "import nltk; nltk.download('wordnet')" | python3
RUN echo "import nltk; nltk.download('stopwords')" | python3
RUN python -m spacy download en_core_web_md
RUN pip install firebase-admin
RUN pip install better_profanity

ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8 

WORKDIR /app
COPY . .