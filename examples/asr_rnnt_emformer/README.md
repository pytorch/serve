### ASR (Automated Speech Recognition) Example

In this example we use torchserve to serve a ASR model that convert wav to text.  There are four steps in this process. First we download a pretrained emformer model and save it to JIT format; Second we start model server, create the model archive; Third we configure the model server with 1 worker; Last we send a wav file to the model endpoint to get text prediction.

#### Steps to run:
- 1. Save asr model to jit format. 
```bash
./00_save_jit_model.sh 
```
- 2. Create model archive
```bash
./01_create_model_archive.sh

output:
2023-01-10T20:46:39,660 [INFO ] pool-3-thread-2 TS_METRICS - MemoryUtilization.Percent:3.2|Level:Host|hostname:ip-172-31-15-90,timestamp:1673383599
```
- 3. Configure model server. register model and add workers. 
```bash
./02_configure_server.sh

Output:
{
  "status": "Model \"rnnt\" Version: 1.0 registered with 0 initial workers. Use scale workers API to add workers for the model."
}
{
  "status": "Processing worker updates..."
}

```

- 4. Get prediction results
```
python3 03_predict.py

output:
he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fat and sauce
```
