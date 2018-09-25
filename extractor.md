# Extractor
Image Retrieval 시스템에 붙일 Extractor 모듈입니다. 

1.  docker
```bash
nvidia-docker run -it -p [host]:[local] -v [host]:[local] --name [name] floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3.31 /bin/bash
```
도커 컨테이너 생성해 줍니다.

2.  clone 한 경로에서
```bash
pip install -r requirements.txt && apt-get install rabbitmq-server -y  && sudo service rabbitmq-server restart
```
3.  extractor 클래스 작성
```bash
cd Modules 
```
```python
#../Modules/Dummy/main.py 
import os
from Modules.dummy.example import test

class Dummy:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
        model_path = os.path.join(self.path, "model.txt")
        self.model = open(model_path, "r")

    def inference_by_path(self, image_path):
        result = []
        # TODO
        #   - Inference using image path
        import time
        time.sleep(2)
        result = [{'descriptor':"[[0.122312]]"}]
        self.result = result

        return self.result        
```
init 에서 모델을 미리 로드합니다. 
inference_by_path 함수는 실제로 인풋이 들어올때 호출되어 [{'descriptor': 'value'}] 를 반환합니다. 
descriptor 는 각자 사용하는 feature의 이름, mac, rmac 이런 이름을 적어주시면 됩니다. value는 이미지에 대한 feature인 n차원 np.array를 리스트로 변환 후 스트링으로 바꾸어주면 됩니다.  
실제 사용한 코드를 ../Modules/Extractor 에 같이 올려 놓았습니다.


작성한 Extractor 클래스는 
 ```python
 #../WebAnalyzer/task.py


@worker_process_init.connect
def module_load_init(**__):
    global analyzer
    worker_index = current_process().index

    print("====================")
    print(" Worker Id: {0}".format(worker_index))
    print("====================")

    # TODO:
    #   - Add your model
    #   - You can use worker_index if you need to get and set gpu_id
    #       - ex) gpu_id = worker_index % TOTAL_GPU_NUMBER
    # from Modules.dummy.main import Dummy
    # analyzer=Dummy()
    from Modules.Extractor.main import Extractor
    analyzer = Extractor()


@app.task
def analyzer_by_path(image_path):
    result = analyzer.inference_by_path(image_path)
    return result
                  
 ```
 에서 적용해주면 됩니다.

4. TEST
여기까지 하셨으면 다시 원래 경로로 돌아가셔서 다음을 실행합니다.
```bash
sh server_initialize.sh
sh server_start.sh
```
정상적으로 실행되었을 경우,  
![img](https://i.imgur.com/oIf9WiK.png)

처럼 이미지를 POST 하면 위와 결과를 볼 수 있습니다.

만약 실행이 안될 경우
```bash
vim celery.log
vim django.log 
```
로그를 확인해보시면 됩니다.


