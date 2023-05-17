#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pdvm_spark 
@File    ：flask_web.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2023/1/5 13:16 
'''

from flask import Flask, redirect

app=Flask(__name__)

#路由装饰器
@app.route('/index')
def index():
    return '<!DOCTYPE html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>pdvm-login</title><link href=/static/css/app.cbda802602fcc0e7ce7bd7aa1cb7885a.css rel=stylesheet></head><body><div id=app></div><script type=text/javascript src=/static/js/manifest.2ae2e69a05c33dfc65f8.js></script><script type=text/javascript src=/static/js/vendor.616306a1a98bd1a8ee7f.js></script><script type=text/javascript src=/static/js/app.19cc68b1fee9550f8b74.js></script></body></html>'

@app.route('/register')
def register():
    return "<a href='/redirect'>跳转重定向页面redirect</a>"

@app.route('/redirect')
def get_redirect():
    return redirect('/index',code=302,Response=None)

if __name__ == '__main__':
    app.run(port=8080,debug=True)

    exit()
    
  
  