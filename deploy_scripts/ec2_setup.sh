#!/usr/bin/env bash

# This is not an automatic script for server deployment
# It is rather a reference or example for deployment

# install packages
sudo yum update
sudo yum search "xxx"
sudo yum update python
sudo yum install python3.x86_64
sudo yum install git-core.x86_64
sudo amazon-linux-extras install nginx1

# install python packages
pip3 install --user aiohttp==3.6.1
pip3 install --user Django==2.1
pip3 install --user dnspython==1.16.0
pip3 install --user gunicorn==19.9.0
pip3 install --user motor==2.0.0
pip3 install --user pymongo==3.9.0
pip3 install --user requests==2.22.0
pip3 install --user numpy==1.16.4
pip3 install --user pandas==0.24.2
pip3 install --user sklearn==0.0
pip3 install --user tensorflow==2.0.0rc0

# User specific aliases and functions
alias ngconfig="vim /etc/nginx/nginx.conf"

# clone source code
cd ~
mkdir stock_recommender
cd stock_recommender/
git clone "https://github.com/ryansu2011/stock_recommender.git"

# modify configuration file for work dir and securate key
# set up nginx configuration file
# set inbound rules for ec2

# start gunicorn and django
cd ~
cd stock_recommender/back_end_web/nindex
sudo gunicorn nindex.wsgi &

# start tf-serve aiohttp server
cd ~
cd stock_recommender
sudo python3 stock_analysis_tool.serving.main &

# start nginx web server
sudo nginx &

# to quit nginx
nginx -s quit
