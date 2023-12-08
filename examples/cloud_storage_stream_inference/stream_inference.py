"""
This module reads amazon_review_polarity_csv.tar.gz file using S3 API from minio
and calls inference REST API on a model hosted by torchserve.
Prerequisite for successfully running this example -
    1) MINIO running and amazon_review_polarity_csv.tar.gz file uploaded in pytorch-data minio bucket.
    2) distilbert-base-uncased model running with torchserve
Please follow instructions in README.md.
"""
import asyncio
import datetime
import tarfile

import aiohttp
import s3fs

MINIO_URL = "http://127.0.0.1:9000"
# Torchserver model URL
MODEL_URL = "http://127.0.0.1:8080/predictions/my_tc"
# Total number of inference calls to send in a single batch
BATCH_TOTAL_TASKS = 1000
# Stop sending infrence request after making TOTAL_INFERENCE_CALLS
TOTAL_INFERENCE_CALLS = 2000


async def read_from_s3():
    aiohttp_client_session = aiohttp.ClientSession()
    s3 = s3fs.S3FileSystem(
        anon=True,
        use_ssl=False,
        client_kwargs={
            "endpoint_url": "http://127.0.0.1:9000",
        },
    )
    # Open tar file from minio using s3 API
    with s3.open("pytorch-data/amazon_review_polarity_csv.tar.gz", "rb") as tf:
        # Read tar file in streaming mode
        with tarfile.open(fileobj=tf, mode="r|gz") as tar:
            # Iterate through each item in the tar. Tar contins train.csv, test.csv... files
            for item in tar:
                # Check if an item is a file and the file is test file
                if item.isfile() and "test" in item.name:
                    # Extract the file
                    with tar.extractfile(item) as f:
                        # Read each line from the file, take the comment which is 3rd column. Create an async task
                        # to call torchserver model inference REST API and add it to task array.
                        tasks = []
                        total_completed_tasks = 0
                        for row in f:
                            row_str = row.decode("utf-8")
                            text = row_str.split(",")[2]
                            if text:
                                if len(tasks) <= BATCH_TOTAL_TASKS:
                                    tasks.append(
                                        send_inference_request(
                                            aiohttp_client_session, text
                                        )
                                    )
                                else:
                                    # Once we've TOTAL_TASKS send all inference request in short time to allow
                                    # torchserver to do batch inference
                                    print(
                                        f"Sending {len(tasks)} inference calls. Total sent so far {TOTAL_INFERENCE_CALLS}."
                                    )
                                    result = await asyncio.gather(*tasks)
                                    if total_completed_tasks > TOTAL_INFERENCE_CALLS:
                                        print(
                                            f"Finished sending maximum number of inference calls {TOTAL_INFERENCE_CALLS}. Stopping..."
                                        )
                                        break
                                    tasks = []
                                    tasks.append(
                                        send_inference_request(
                                            aiohttp_client_session, text
                                        )
                                    )
                                total_completed_tasks = total_completed_tasks + 1
                            else:
                                print("Error - Retrieved text is None - ", text)
    await aiohttp_client_session.close()


async def send_inference_request(aiohttp_session, text):
    """
    This functions call inference REST API with data.
    """
    lines = text.splitlines()[0]
    sentence = lines.split(".")[0].replace('"', "")
    msg = f'{{"text":"{sentence}"}}'
    print(datetime.datetime.now(), "- Calling model inference with data - ", msg)
    async with aiohttp_session.post(MODEL_URL, data=f"{msg}") as resp:
        resp_text = await resp.text()
        print(datetime.datetime.now(), "- Model prediction: ", resp_text)
        return resp_text


if __name__ == "__main__":
    asyncio.run(read_from_s3())
