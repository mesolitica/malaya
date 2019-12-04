## How-to

1. Install `wrk`, https://github.com/wg/wrk

2. Run any `docker-compose.yaml` inside any folder.

3. Execute stress test,

```bash
wrk -t15 -c600 -d1m --timeout=10s http://localhost:8080/?string=comel
```

## Benchmarks

These are my machine specifications,

- Intel(R) Core(TM) i7-8557U CPU @ 1.70GHz
- 16 GB 2133 MHz LPDDR3

And I use same `wrk` command,

```bash
wrk -t15 -c600 -d1m --timeout=10s http://localhost:8080/?string=comel
```

The purpose of these benchmarks, how fast and how much requests for a model able to serve on perfect realtime, let say live streaming data from social media to detect sentiment, whether a text is a negative or a positive. Tested on ALBERT-BASE sentiment model,

```python
import malaya
model = malaya.sentiment.transformer(model = 'albert', size = 'base')
```

Others,

- ALBERT BASE is around 43MB.
- Limit memory is 2GB, set by Docker itself.
- No limit on CPU usage.

1. Fast-api

- workers automatically calculated by fast-api.

```text
Running 1m test @ http://localhost:8080/?string=comel
  15 threads and 600 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     4.09s     2.02s   10.00s    67.02%
    Req/Sec     7.04      5.94    60.00     74.41%
  3268 requests in 1.00m, 702.11KB read
  Socket errors: connect 364, read 80, write 0, timeout 21
Requests/sec:     54.41
Transfer/sec:     11.69KB
```

2. Gunicorn Flask

- 5 sync workers.

```text
Running 1m test @ http://localhost:8080/?string=comel
  15 threads and 600 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     3.35s   726.67ms   9.47s    86.62%
    Req/Sec     8.64      6.43    40.00     72.52%
  2400 requests in 1.00m, 581.25KB read
  Socket errors: connect 364, read 104, write 0, timeout 30
Requests/sec:     39.95
Transfer/sec:      9.68KB
```

3. UWSGI Flask + Auto scaling

- Min 1 worker, Max 10 workers, spare2 algorithm.