import os
from multiprocessing import cpu_count


# Socket path
bind = 'unix:/run/txt2img_lighting/gunicorn.sock'

# Worker Options
# workers = cpu_count() + 1
workers = 1
worker_class = 'uvicorn.workers.UvicornWorker'

# Worker timeout
timeout = 300

# Logging Options
loglevel = 'debug'
accesslog = '/var/log/txt2img_lighting/access.log'
errorlog = '/var/log/txt2img_lighting/error.log'
