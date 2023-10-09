FROM ubuntu:latest

#https://stackoverflow.com/questions/23935141/how-to-copy-docker-images-from-one-host-to-another-without-using-a-repository
#save docker images

#why use docker: 
#https://towardsdatascience.com/why-using-docker-for-machine-learning-74c927ceb6c4


#this may be the best option besides steps from git to execution


RUN apt-get update && apt-get install -y python3 \
    python3-pip


WORKDIR /home/python_files

#install all packages for both.py files
RUN pip3 install scipy
RUN pip3 install scikit-learn
RUN pip3 install pandas
RUN pip3 install nltk
RUN pip3 install ordered_set



#do we really need to copy to the container???
COPY transform.py .
COPY  analysis.py .
COPY both.sh .

COPY tf_matrix.csv .
COPY mbti_1.csv .
COPY results.csv .


#note this seems to run when the image is being built
#not just when the image is run 

# RUN ./both.sh

#this might be the only way to force these scripts to run only when the image is run ratehr than being built

CMD ["./both.sh"]





