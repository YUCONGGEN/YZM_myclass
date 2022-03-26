FROM  conda/miniconda3-centos7
COPY . ./code
WORKDIR ./code
RUN  yum -y upgrade
RUN  yum install python3 -y
RUN yum install python3-pip -y
RUN pip3 install --upgrade pip
RUN yum install libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 -y
RUN  pip3 install Flask==1.1.2 -i https://pypi.douban.com/simple
RUN  pip3 install opencv-python-headless -i https://pypi.douban.com/simple
RUN  pip3 install numpy -i https://pypi.douban.com/simple
RUN  pip3 install tensorflow==1.15 -i https://pypi.douban.com/simple
CMD ["python","flasktest.py"]