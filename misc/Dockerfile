FROM ubuntu:latest


# Note: An older version of python is specified to mitigate the chance that a newer verison of python runs into an issue with the notebook code.
# Python 3.10.7 worked on personal machine
RUN apt-get update && apt-get install -y python3.10 \
    python3-pip


#Install packages rquried for both python files
RUN pip3 install scipy
RUN pip3 install scikit-learn
RUN pip3 install pandas
RUN pip3 install nltk
RUN pip3 install ordered_set


WORKDIR /home/files

#These are the files to run inside the container (They don't change):
COPY transform.py .
COPY analysis.py .
COPY both_scripts.sh .

#This file is raw data that is transformed into tf_matrix.csv by the transform.py program 
#It also doesn't change
COPY mbti_1.csv .


#Run the script that runs transform.py and analysis.py in order.
CMD ["./both_scripts.sh"]





