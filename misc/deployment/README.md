## How-to

1. Install `wrk`, https://github.com/wg/wrk

2. Run any `docker-compose.yaml` inside any folder.

3. Execute stress test,

```bash
wrk -t15 -c600 -d1m --timeout=15s http://localhost:8080/?string=husein%20sangat%20comel%20dan%20handsome%20tambahan%20lagi%20ketiak%20wangi
```

## Benchmarks

These are my machine specifications,

- Intel(R) Core(TM) i7-8557U CPU @ 1.70GHz
- 16 GB 2133 MHz LPDDR3

And I use same `wrk` command,

```bash
wrk -t15 -c600 -d1m --timeout=15s http://localhost:8080/?string=husein%20sangat%20comel%20dan%20handsome%20tambahan%20lagi%20ketiak%20wangi
```

The purpose of these benchmarks, how fast and how much requests for a model able to serve on perfect minibatch realtime, let say live streaming data from social media to detect sentiment, whether a text is a negative or a positive. Tested on ALBERT-BASE sentiment model,

```python
import malaya
model = malaya.sentiment.transformer(model = 'albert', size = 'base')
```

Others,

- ALBERT BASE is around 43MB.
- Limit memory is 2GB, set by Docker itself.
- batch size of 50 strings, duplicate 50 times of `husein sangat comel dan handsome tambahan lagi ketiak wangi`, can check every deployment in `app.py` or `main.py`.
- No limit on CPU usage.

1. Fast-api

- workers automatically calculated by fast-api.

```text
Running 1m test @ http://localhost:8080/?string=husein%20sangat%20comel%20dan%20handsome%20tambahan%20lagi%20ketiak%20wangi
  15 threads and 600 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     0.00us    0.00us   0.00us     nan%
    Req/Sec     0.24      1.16     9.00     95.52%
  68 requests in 1.00m, 8.96KB read
  Socket errors: connect 364, read 293, write 0, timeout 68
Requests/sec:      1.13
Transfer/sec:     152.75B
```

2. Gunicorn Flask

- 5 sync workers.

```text
Running 1m test @ http://localhost:8080/?string=husein%20sangat%20comel%20dan%20handsome%20tambahan%20lagi%20ketiak%20wangi
  15 threads and 600 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     7.98s     3.25s   12.71s    41.67%
    Req/Sec     0.49      1.51     9.00     90.91%
  59 requests in 1.00m, 9.10KB read
  Socket errors: connect 364, read 39, write 0, timeout 47
Requests/sec:      0.98
Transfer/sec:     155.12B
```

3. UWSGI Flask + Auto scaling

- Min 2 worker, Max 10 workers, spare2 algorithm.

```text
Running 1m test @ http://localhost:8080/?string=husein%20sangat%20comel%20dan%20handsome%20tambahan%20lagi%20ketiak%20wangi
  15 threads and 600 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     8.80s     4.16s   14.73s    62.50%
    Req/Sec     0.75      2.60     9.00     91.67%
  12 requests in 1.00m, 0.90KB read
  Socket errors: connect 364, read 105, write 0, timeout 4
Requests/sec:      0.20
Transfer/sec:      15.37B
```

4. UWSGI Flask

- 4 Workers.

```text
Running 1m test @ http://localhost:8080/?string=husein%20sangat%20comel%20dan%20handsome%20tambahan%20lagi%20ketiak%20wangi
  15 threads and 600 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     8.79s     4.13s   14.87s    53.33%
    Req/Sec     1.06      3.16    20.00     92.59%
  56 requests in 1.00m, 4.21KB read
  Socket errors: connect 364, read 345, write 0, timeout 41
Requests/sec:      0.93
Transfer/sec:      71.74B
```
