.. code:: python

    import malaya

Disable file validation
-----------------------

If you deployed some of Malaya models on persist, short-life
(auto-restart to reduce memory consumption) and async / multiprocess
workers, you might get errors related to file checking. You can skip
this error as long you able to persist malaya models.

download model
^^^^^^^^^^^^^^

So, first you need to download the model into your local machine /
environment, run this on different script,

.. code:: python

    model = malaya.zero_shot.classification.transformer(model = 'tiny-albert')


.. parsed-literal::

    INFO:tensorflow:loading sentence piece model


load model
^^^^^^^^^^

Load model without need to check model, run this on top of fastapi /
flask / gunicorn.

.. code:: python

    model = malaya.zero_shot.classification.transformer(model = 'tiny-albert', validate = False)


.. parsed-literal::

    INFO:tensorflow:loading sentence piece model


This loaded model able to share among multi-workers / multi-threads.

disable type checking
---------------------

Make sure you already install latest version herpetologist,

.. code:: bash

   pip install herpetologist -U

If you check Malaya source code, you can see we check parameters on
function / method definition,
https://github.com/huseinzol05/Malaya/blob/master/malaya/model/bert.py#L232

We use herpetologist to check passed variables,
https://github.com/huseinzol05/herpetologist

.. code:: python

   @check_type
   def predict(self, strings: List[str], add_neutral: bool = True):
       """
       classify a string.
       Parameters
       ----------
       strings: List[str]
       add_neutral: bool, optional (default=True)
           if True, it will add neutral probability.
       Returns
       -------
       result: List[str]
       """

``@check_type`` will check ``strings`` is a ``List[str]`` or not, if
not, it will throw an error.

But this ``@check_type`` will become expensive if you have massive list
of strings. So you can disable to this type checking by simply set bash
environment.

Some of our environments we want to enable it, some of it also we want
to disable, and we do not want herpetologist to keep check the
variables. So to disable it, simply set bash environment,

.. code:: bash

   export ENABLE_HERPETOLOGIST=false

Or, using python,

.. code:: python

   import os
   os.environ['ENABLE_HERPETOLOGIST'] = 'false'

You can see impact of time execution in this
`example <https://github.com/huseinzol05/herpetologist/blob/master/example.ipynb>`__.

Use smaller model
-----------------

Stacking multiple smaller models much faster than a single big model.
But this cannot ensure the accuracy will be same as the big model.

docker example
--------------

You can check some docker examples and benchmarks at here,
https://github.com/huseinzol05/Malaya/tree/master/misc/deployment.

The purpose of these benchmarks, how fast and how much requests for a
model able to serve on perfect minibatch realtime, let say live
streaming data from social media to detect sentiment, whether a text is
a negative or a positive. Tested on ALBERT-BASE sentiment model.

These are my machine specifications,

1. Intel(R) Core(TM) i7-8557U CPU @ 1.70GHz
2. 16 GB 2133 MHz LPDDR3

And I use same wrk command,

.. code:: bash

   wrk -t15 -c600 -d1m --timeout=15s http://localhost:8080/?string=husein%20sangat%20comel%20dan%20handsome%20tambahan%20lagi%20ketiak%20wangi

Some constraints,

1. ALBERT BASE is around 43MB.
2. Limit memory is 2GB, set by Docker itself.
3. batch size of 50 strings, duplicate 50 times of
   ``husein sangat comel dan handsome tambahan lagi ketiak wangi``, can
   check every deployment in app.py or main.py.
4. No limit on CPU usage.
5. no caching.

fast-api
^^^^^^^^

workers automatically calculated by fast-api,
https://github.com/huseinzol05/Malaya/tree/master/misc/deployment/fast-api

.. code:: text

   Running 1m test @ http://localhost:8080/?string=husein%20sangat%20comel%20dan%20handsome%20tambahan%20lagi%20ketiak%20wangi
     15 threads and 600 connections
     Thread Stats   Avg      Stdev     Max   +/- Stdev
       Latency     0.00us    0.00us   0.00us     nan%
       Req/Sec     0.24      1.16     9.00     95.52%
     68 requests in 1.00m, 8.96KB read
     Socket errors: connect 364, read 293, write 0, timeout 68
   Requests/sec:      1.13
   Transfer/sec:     152.75B

Gunicorn Flask
^^^^^^^^^^^^^^

5 sync workers,
https://github.com/huseinzol05/Malaya/tree/master/misc/deployment/gunicorn-flask

.. code:: text

   Running 1m test @ http://localhost:8080/?string=husein%20sangat%20comel%20dan%20handsome%20tambahan%20lagi%20ketiak%20wangi
     15 threads and 600 connections
     Thread Stats   Avg      Stdev     Max   +/- Stdev
       Latency     7.98s     3.25s   12.71s    41.67%
       Req/Sec     0.49      1.51     9.00     90.91%
     59 requests in 1.00m, 9.10KB read
     Socket errors: connect 364, read 39, write 0, timeout 47
   Requests/sec:      0.98
   Transfer/sec:     155.12B

UWSGI Flask + Auto scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^

Min 2 worker, Max 10 workers, spare2 algorithm,
https://github.com/huseinzol05/Malaya/tree/master/misc/deployment/uwsgi-flask-cheaper

.. code:: text

   Running 1m test @ http://localhost:8080/?string=husein%20sangat%20comel%20dan%20handsome%20tambahan%20lagi%20ketiak%20wangi
     15 threads and 600 connections
     Thread Stats   Avg      Stdev     Max   +/- Stdev
       Latency     8.80s     4.16s   14.73s    62.50%
       Req/Sec     0.75      2.60     9.00     91.67%
     12 requests in 1.00m, 0.90KB read
     Socket errors: connect 364, read 105, write 0, timeout 4
   Requests/sec:      0.20
   Transfer/sec:      15.37B

UWSGI Flask
^^^^^^^^^^^

4 Workers,
https://github.com/huseinzol05/Malaya/tree/master/misc/deployment/uwsgi-flask-fork

.. code:: text

   Running 1m test @ http://localhost:8080/?string=husein%20sangat%20comel%20dan%20handsome%20tambahan%20lagi%20ketiak%20wangi
     15 threads and 600 connections
     Thread Stats   Avg      Stdev     Max   +/- Stdev
       Latency     8.79s     4.13s   14.87s    53.33%
       Req/Sec     1.06      3.16    20.00     92.59%
     56 requests in 1.00m, 4.21KB read
     Socket errors: connect 364, read 345, write 0, timeout 41
   Requests/sec:      0.93
   Transfer/sec:      71.74B

Learn different deployment techniques
-------------------------------------

Eg, Change concurrent requests into mini-batch realtime processing to
speed up text classification,
`repository <https://github.com/huseinzol05/Gather-Deployment/tree/master/tensorflow/26.fastapi-batch-streamz>`__

**This can reduce time taken up to 95%!**

