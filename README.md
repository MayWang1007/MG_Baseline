**MG_BASELINE**

</div> 

## Installation:

```bash
pip3 install -r docker/requirements.txt
```


Installation of ONNX runtime:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz \
    && tar -xf onnxruntime-linux-x64-1.14.1.tgz \
    && cp onnxruntime-linux-x64-1.14.1/lib/* /usr/lib/ && cp onnxruntime-linux-x64-1.14.1/include/* /usr/include/
```

Optionally, you could use the Dockerfile to build the image:
```bash
cd docker && sh build.sh
```

## Inference Example:

To execute the **Follower** algorithm and produce an animation using pre-trained weights, use the following command:

```bash
python mgmapf_infer.py --map_name maze-32-32-4 --num_agents 10 --num_goals 10
```

The animation will be stored in the `renders` folder.The results will be stored in the `results` folder.

