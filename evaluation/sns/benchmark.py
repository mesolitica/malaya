import aiohttp
import asyncio
import time
import json
import click
import os

async def stress_test(index, url, model, start_token, start_token_length):
    json_data = {
        'model': model,
        'prompt': start_token * start_token_length,
        'max_tokens': 384,
        'stream': True,
        'ignore_eos': True,
    }

    start_time = time.time()
    count = 0

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=json_data) as response:
            first_token_time = None

            async for line in response.content:
                elapsed = time.time() - start_time

                l = line.decode()
                if len(l) and 'data: ' in l:
                    try:
                        l = json.loads(l.split('data: ')[1])
                        if first_token_time is None:
                            first_token_time = elapsed
                        count += 1
                    except Exception:
                        pass

    total_time = time.time() - start_time
    return {
        "first_token": first_token_time,
        "total_response": total_time,
        'total_token': count,
    }

async def run_stress_test(url, model, start_token, start_token_length, requests_per_second):
    tasks = [
        stress_test(i, url, model, start_token, start_token_length)
        for i in range(requests_per_second)
    ]
    results = await asyncio.gather(*tasks)
    return results

@click.command()
@click.option('--url', default='http://localhost:8000/v1/completions', help='The inference endpoint URL.')
@click.option('--model', required=True, help='Model name to use.')
@click.option('--start_token', default='<|im_start|>', help='start token.')
@click.option('--start_token_length', default = 4096, help='start token length.')
@click.option('--save', required=True, help='save folder name.')
@click.option('--rps-list', default='10,20,30,40,50,60,70,80,90,100',
              help='Comma-separated list of RPS (requests per second) values to test.')
def main(url, model, start_token, start_token_length, save, rps_list):
    os.makedirs(save, exist_ok = True)
    rps_values = [int(rps) for rps in rps_list.split(',')]

    async def run_all():
        for rps in rps_values:
            print(f"\nTesting with {rps} RPS")
            results = await run_stress_test(url, model, start_token, start_token_length, rps)
            first_token_avg = sum(res["first_token"] for res in results if res["first_token"] is not None) / len(results)
            total_response_avg = sum(res["total_response"] for res in results) / len(results)
            total_token = sum(res["total_token"] for res in results)

            results_data = {
                "first_token_avg": first_token_avg,
                "total_response_avg": total_response_avg,
                'total_token': total_token,
                'average_token': total_token / len(results),
            }

            with open(f"{save}/{rps}RPS.json", "w") as outfile:
                json.dump(results_data, outfile, indent=4)

    asyncio.run(run_all())

if __name__ == '__main__':
    main()
