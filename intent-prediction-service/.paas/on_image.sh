source /opt/app/.venv/bin/activate && pip install -U --ignore-installed uvicorn==0.33.0 -i http://mirror.release.ctripcorp.com/repository/pypigroup/simple/ --trusted-host mirror.release.ctripcorp.com

rm -f /etc/supervisord.d/gunicorn.ini
cp -f /opt/app/.paas/uvicorn.ini /etc/supervisord.d/uvicorn.ini
