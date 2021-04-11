#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Date: 2021/4/11 上午10:07
# @Filename: redis_func
# @Author：zyt
import redis
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

if FLAGS.mid != 'null':
    r = redis.Redis(host=FLAGS.ip_PS, port=6379, decode_responses=True)
else:
    r = None

def incr_iter_times(uid, mid):
    if uid.startswith('W'):
        r.incr("contribute-{}-{}".format(uid[1:], mid), amount=1)

def update_validate(mid, iter, accu, loss):
    r.lpush("model-{}-iter".format(mid), iter)
    r.lpush("model-{}-accu".format(mid), accu)
    r.lpush("model-{}-loss".format(mid), loss)